"""Подставной провайдер слота: характеры отказа merge на проде (LSBN-T-35, G2).

На проде merge-путь падал не «сетью», а ответами модели: (а) текст вместо
JSON-объекта, (б) пустой `content` при непустом рассуждении (бюджет ответа съеден
`thinking`), (в) ответ, обрезанный по лимиту вывода (`finish_reason=length`).
Ни один из этих характеров не воспроизводился тестом — только наблюдался на бою.

Здесь подставной провайдер говорит ровно эти три ответа на уровне ПРОВОДА
(`httpx.MockTransport`): настоящий `LLMClient`, настоящий `SummaryService`,
настоящая merge-петля воркера — подменён только ответ модели. Проверяется
контракт отказа на паре ОБЫЧНЫХ заметок (`origin_note_id IS NULL`, то есть
барьер «один исходник» не при чём): данные не теряются, пара встаёт в
ретрай/attention по существующему контракту, повторы ограничены и замирают.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.summary import SummaryService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

SLOT_MAX_ATTEMPTS = 3  # SLOT_MAX_ATTEMPTS теста: быстро и без пауз слота
SLOT_PAUSE_AFTER = 2  # пауза слота наступает раньше attention (порог внимания выше)

#: Характер отказа → ожидаемый исход слота (контракт #39/#50).
EXPECTED_OUTCOME = {
    "prose": "off_contract",  # ответ не JSON-объект контракта merge
    "empty-reasoning": "truncated",  # content пуст, рассуждение съело бюджет
    "cut": "off_contract",  # JSON обрезан по лимиту: разбор не проходит
}

TEXTS = [
    "Ежедневный бэкап кластера: проверяем снимок и журнал репликации утром.",
    "Ежедневный бэкап кластера: проверяем снимок и журнал репликации вечером.",
]


class BrokenMergeProvider:
    """Подставной провайдер summary-слота: ответы, которые давала сломанная модель.

    Один и тот же экземпляр отвечает на все chat-запросы слота (транспорт общий
    для операций summarize/title/merge/split). Журнал `requests` — тела запросов:
    тест проверяет, что повторы ограничены и после attention слот молчит.
    """

    def __init__(self, mode: str) -> None:
        if mode not in EXPECTED_OUTCOME:
            raise ValueError(f"неизвестный режим отказа: {mode}")
        self.mode = mode
        self.requests: list[dict] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json=self._answer())

    def _answer(self) -> dict:
        """Ответ ollama-совместимого chat-роута (единственный путь слота)."""
        if self.mode == "prose":
            return _chat(
                "Конечно! Вот объединённая заметка: оба текста говорят об одном "
                "и том же бэкапе кластера, поэтому я собрал их вместе."
            )
        if self.mode == "empty-reasoning":
            return _chat(
                "",
                thinking=(
                    "Сначала разберу обе части, выделю общие факты и соберу "
                    "самодостаточные части..."
                ),
            )
        # cut: JSON оборван на середине — как обрезанный по лимиту ответ модели
        return _chat(
            (
                '{"parts": [{"title": "Объединённая заметка", '
                '"text": "Ежедневный бэкап кластера: проверяем снимок'
            ),
            done_reason="length",
        )


def _chat(
    content: str, *, thinking: str | None = None, done_reason: str = "stop"
) -> dict:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": "broken-merge",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 120,
        "eval_count": 40,
    }


@pytest.fixture
def slot_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    monkeypatch.setenv("SLOT_MAX_ATTEMPTS", str(SLOT_MAX_ATTEMPTS))
    monkeypatch.setenv("SLOT_PAUSE_AFTER", str(SLOT_PAUSE_AFTER))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _merge_job(worker: BackgroundWorker, settings, note_id: int, older_id: int) -> int:
    """Merge-работа в очереди summary — как её ставит judge-петля."""
    worker._create_job(
        "summary", "merge", note_id, payload=json.dumps({"older_id": older_id})
    )
    with session(settings) as conn:
        row = conn.execute(
            "SELECT id FROM worker_jobs WHERE slot = 'summary' AND kind = 'merge' "
            "AND note_id = ? ORDER BY id DESC",
            (note_id,),
        ).fetchone()
    assert row is not None
    return int(row["id"])


def _job_state(settings, job_id: int) -> dict:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT status, attention_at, attempts, last_outcome "
            "FROM worker_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else {}


def _alive_texts(settings) -> list[str]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT text FROM notes WHERE deleted_at IS NULL ORDER BY id"
        ).fetchall()
    return [row["text"] for row in rows]


def _events(caplog, event: str) -> list:
    return [
        record for record in caplog.records if getattr(record, "event", None) == event
    ]


@pytest.mark.parametrize("mode", sorted(EXPECTED_OUTCOME))
def test_broken_merge_answer_keeps_the_pair_and_stops(slot_env, mode, caplog) -> None:
    """Сломанный ответ модели: обе заметки живы, пара в attention, повторы конечны."""
    provider = BrokenMergeProvider(mode)
    notes = NoteService(slot_env, FailingEmbedder())
    older_id = int(notes.save(TEXTS[0])["id"])
    note_id = int(notes.save(TEXTS[1])["id"])
    summarizer = SummaryService(slot_env, transport=provider.transport)
    worker = BackgroundWorker(slot_env, HashEmbedder(8), summarizer)
    job_id = _merge_job(worker, slot_env, note_id, older_id)

    with caplog.at_level(logging.WARNING, logger="app"):
        for _ in range(SLOT_MAX_ATTEMPTS):
            # 0 — ни одна работа не ЗАКРЫТА: пара осталась pending на повтор.
            assert worker.process_merge_pending() == 0
            # NFR-3: сломанный ответ не портит данные — обе заметки живы.
            assert _alive_texts(slot_env) == TEXTS
        calls_at_attention = len(provider.requests)

        state = _job_state(slot_env, job_id)
        assert state["status"] == "pending"  # работа осталась в очереди
        assert state["attention_at"] is not None  # но ждёт человека
        assert state["attempts"] == SLOT_MAX_ATTEMPTS
        assert state["last_outcome"] == EXPECTED_OUTCOME[mode]

        # Пара вморожена: следующие циклы не берут её и не зовут модель.
        for _ in range(3):
            assert worker.process_merge_pending() == 0
        assert len(provider.requests) == calls_at_attention

    # Повтор конечен: одна попытка на цикл (авто-повтор бюджета — не больше двух).
    assert SLOT_MAX_ATTEMPTS <= calls_at_attention <= 2 * SLOT_MAX_ATTEMPTS
    failures = _events(caplog, "dedup_merge_failed")
    assert len(failures) == SLOT_MAX_ATTEMPTS
    assert {record.outcome for record in failures} == {EXPECTED_OUTCOME[mode]}
    assert {record.older_id for record in failures} == {older_id}
    assert {record.note_id for record in failures} == {note_id}

    # Данные целы: две заметки, тексты не изменены, частей слияния не появилось.
    assert _alive_texts(slot_env) == TEXTS
    with session(slot_env) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 2
        origins = conn.execute(
            "SELECT DISTINCT origin_note_id FROM notes"
        ).fetchall()
    assert [row[0] for row in origins] == [None]  # обычная пара, не части нарезки

    # Attention виден и оператору, и /health — с причиной отказа слота.
    attention = worker.attention_jobs()
    assert len(attention) == 1
    assert attention[0]["job_id"] == job_id
    assert attention[0]["outcome"] == EXPECTED_OUTCOME[mode]
    assert attention[0]["attempts"] == SLOT_MAX_ATTEMPTS
    health = worker.slots_health()["summary"]
    assert health["attention"] == 1
    assert worker.queues_health()["summary"]["pending"] == 0
    if mode == "cut":
        # Причина видна без журнала: ответ модели обрезан по лимиту вывода.
        assert health["last_finish_reason"] == "length"
    else:
        assert health["last_finish_reason"] != "length"
