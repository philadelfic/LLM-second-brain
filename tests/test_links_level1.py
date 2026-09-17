"""lsb-0010-02: связи заметок, уровень 1 — таблица `links`, расчёт без LLM.

Проверяется механика расчёта (джоба/backfill — постановка 8, выдача —
постановка 10): три вида связи (`cosine`/`entities`/`mention`), приоритет
вида на пару, идемпотентный пересчёт одной заметки, канонический порядок
пары (`note_a < note_b`) и маркер `notes.links_at`, отсечения при расчёте
(сама заметка, soft-deleted; «свой неймспейс» здесь НЕ отсекается), сбросы
маркера в `NoteService`, `purge_orphans`.

Ни одного вызова модели: векторы задаются явно (`app.storage.vectors`) —
косинусы детерминированы; виды `entities`/`mention` работают и на заметке
без вектора (`vector_status='pending'`), потому что их пул — FTS.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import Settings, get_settings
from app.services.links import LinksService
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.storage import vectors
from app.storage.db import init_db, session

DIM = 8  # маленькая размерность: косинусы задаются вектором явно
SOURCE = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _vec(cosine: float) -> list[float]:
    """Единичный вектор с заданным косинусом к SOURCE (второе измерение)."""
    return [cosine, (1.0 - cosine * cosine) ** 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def dim8(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — вектора задаются литерально."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _seed(settings: Settings, rows: list[tuple]) -> None:
    """Заметки прямым SQL: (id, title, text, summary, vector, deleted_at)."""
    with session(settings) as conn:
        for note_id, title, text, summary, vector, deleted in rows:
            conn.execute(
                "INSERT INTO notes "
                "(id, title, text, summary, namespace, vector_status, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    title,
                    text,
                    summary,
                    "work" if note_id % 2 == 0 else "default",
                    "ok" if vector is not None else "pending",
                    deleted,
                ),
            )
            if vector is not None:
                vectors.upsert(conn, note_id, vector)


def _links(monkeypatch: pytest.MonkeyPatch, **env: str) -> LinksService:
    """LinksService на переопределённом окружении (пороги/пул)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    return LinksService(settings, search=SearchService(settings))


def _rows() -> list[tuple[int, int, str, float | None]]:
    """Строки `links` — (note_a, note_b, kind, score), по каноническому порядку."""
    with session(get_settings()) as conn:
        return [
            (row["note_a"], row["note_b"], row["kind"], row["score"])
            for row in conn.execute("SELECT * FROM links ORDER BY note_a, note_b")
        ]


def _pairs() -> set[tuple[int, int]]:
    return {(note_a, note_b) for note_a, note_b, _, _ in _rows()}


def _set_links_at(note_id: int, stamp: str | None) -> None:
    with session(get_settings()) as conn:
        conn.execute(
            "UPDATE notes SET links_at = ? WHERE id = ?", (stamp, note_id)
        )


def _links_at(note_id: int) -> str | None:
    with session(get_settings()) as conn:
        return conn.execute(
            "SELECT links_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()[0]


class TestCosine:
    """Вид `cosine`: KNN по вектору заметки, порог LINK_COSINE_THRESHOLD."""

    def test_close_candidate_makes_a_link(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Кандидат выше порога (0.70) — связь, `score` = косинус."""
        _seed(
            dim8,
            [
                (1, "Источник", "исходный текст", "", SOURCE, None),
                (2, "Близкая", "второй текст", "", _vec(0.75), None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        assert _rows() == [(1, 2, "cosine", pytest.approx(0.75, abs=1e-4))]

    def test_threshold_boundary_is_inclusive(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Порог включительный: кандидат ровно на пороге остаётся.

        Порог берётся измеренным (`similar_notes`), а не литеральным 0.70:
        vec0 хранит float32, и 0.70 по обратном пути даёт 0.6999999881.
        """
        _seed(
            dim8,
            [
                (1, "Источник", "исходный текст", "", SOURCE, None),
                (2, "Близкая", "второй текст", "", _vec(0.70), None),
            ],
        )
        _, cosine = _links(monkeypatch)._search.similar_notes(1, 20, 0.0)[0]
        service = _links(monkeypatch, LINK_COSINE_THRESHOLD=str(cosine))
        assert service.compute_for_note(1) == 1
        assert _rows() == [(1, 2, "cosine", cosine)]

    def test_below_threshold_is_not_a_link(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Кандидат ниже порога (0.69) — связи нет (порог ленивого 0.50 ниже)."""
        _seed(
            dim8,
            [
                (1, "Источник", "исходный текст", "", SOURCE, None),
                (2, "Ниже порога", "второй текст", "", _vec(0.69), None),
            ],
        )
        service = _links(monkeypatch)
        _, cosine = service._search.similar_notes(1, 20, 0.0)[0]
        assert service.compute_for_note(1) == 0
        assert _rows() == []
        # Кандидат виден ленивому уровню — порог связи строже поискового.
        assert service._search.similar_notes(1, 20, 0.50)[0][0] == 2
        assert cosine < 0.70


class TestEntities:
    """Вид `entities`: общие значимые слова title+summary."""

    def test_two_shared_words_make_a_link(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2 общих значимых слова → связь; `score` = общих / max(слов_a, слов_b)."""
        _seed(
            dim8,
            [
                # Вектора нет: пул `entities` — FTS, KNN не нужен.
                (1, "Архитектура решений", "исходный текст", "", None, None),
                (2, "Архитектура системы", "второй текст", "решений", None, None),
                (3, "Архитектура системы", "третий текст", "сводка", None, None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        # Общие: «архитектура» + «решений» (2 из 3 слов кандидата).
        assert _rows() == [(1, 2, "entities", pytest.approx(2 / 3, abs=1e-6))]

    def test_one_shared_word_is_not_a_link(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """1 общее слово < LINK_ENTITIES_MIN_COMMON (2) — связи нет."""
        _seed(
            dim8,
            [
                (1, "Архитектура решений", "исходный текст", "", None, None),
                (3, "Архитектура системы", "третий текст", "сводка", None, None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 0
        assert _rows() == []

    def test_short_and_stop_words_do_not_participate(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Слова < LINK_ENTITIES_MIN_WORD_CHARS и стоп-слова не считаются общими.

        Источник и кандидат делят 5 коротких/стоп-слов и 2 значимых: связь
        возникает ровно по значимым (score = 1.0), а короткие и стоп-слова
        счёт не меняют — иначе третий кандидат (только они) тоже дал бы связь.
        """
        _seed(
            dim8,
            [
                (
                    1,
                    "Архитектура сервера",
                    "первый текст",
                    "полностью механически этот план тут",
                    None,
                    None,
                ),
                (
                    2,
                    "Сводка решений",
                    "второй текст",
                    "полностью механически только можно",
                    None,
                    None,
                ),
                (
                    3,
                    "Отдельная сводка",
                    "третий текст",
                    "сводка этот план тут только можно",
                    None,
                    None,
                ),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        # Общие значимые: «полностью» + «механически» (2 из 4); короткие
        # («этот», «план», «тут») и стоп-слова («только», «можно») не считаются.
        assert _rows() == [(1, 2, "entities", pytest.approx(0.5, abs=1e-6))]


class TestMention:
    """Вид `mention`: название другой заметки встречается в title/text этой."""

    def test_title_in_text_makes_a_link(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Название другой заметки в тексте → связь, `score` = NULL."""
        _seed(
            dim8,
            [
                (
                    1,
                    "Обзорная заметка",
                    "смотри Проект Альфа и аб, а ещё #4",
                    "",
                    None,
                    None,
                ),
                (2, "Проект Альфа", "посторонний текст", "", None, None),
                (3, "аб", "короткое название", "", None, None),
                (4, "Отдельная заметка", "четвёртый текст", "", None, None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        assert _rows() == [(1, 2, "mention", None)]

    def test_short_title_is_not_a_candidate(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Название короче 3 символов (предел триграммы) связью не становится."""
        _seed(
            dim8,
            [
                (1, "Обзорная заметка", "смотри аб", "", None, None),
                (3, "аб", "короткое название", "", None, None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 0
        assert _rows() == []

    def test_id_reference_is_not_a_link(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Упоминание по `id` (#4) связью не становится — правило про название."""
        _seed(
            dim8,
            [
                (1, "Обзорная заметка", "смотри заметку #4 и номер 4", "", None, None),
                (4, "Отдельная заметка", "четвёртый текст", "", None, None),
            ],
        )
        # Кандидат в пуле (общее слово «заметка»), но название в тексте не
        # встречается: `mention` не срабатывает, `entities` — только 1 общее слово.
        assert _links(monkeypatch).compute_for_note(1) == 0
        assert _rows() == []


class TestKindPriority:
    """Приоритет вида на пару: mention > entities > cosine, одна строка на пару."""

    def test_mention_wins_over_cosine(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Обе связи есть — хранится `mention` с score NULL, строка одна."""
        _seed(
            dim8,
            [
                (1, "Источник", "см. Проект Альфа", "", SOURCE, None),
                (2, "Проект Альфа", "второй текст", "", _vec(0.95), None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        assert _rows() == [(1, 2, "mention", None)]

    def test_entities_wins_over_cosine(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Совпали entities и cosine — побеждает entities (score ≠ косинус)."""
        _seed(
            dim8,
            [
                (1, "Отчёт", "первый текст", "архитектуры сервера", SOURCE, None),
                (2, "Сводка", "второй текст", "архитектуры сервера", _vec(0.90), None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        assert _rows() == [(1, 2, "entities", pytest.approx(2 / 3, abs=1e-6))]

    def test_all_three_rules_give_one_row(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Совпали все три правила — ровно одна строка, вид `mention`."""
        _seed(
            dim8,
            [
                (
                    1,
                    "Источник",
                    "см. Проект Альфа",
                    "архитектуры сервера",
                    SOURCE,
                    None,
                ),
                (
                    2,
                    "Проект Альфа",
                    "второй текст",
                    "архитектуры сервера",
                    _vec(0.95),
                    None,
                ),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 1
        assert _rows() == [(1, 2, "mention", None)]


class TestStorage:
    """Хранение: канонический порядок, идемпотентность, отсечения, маркер."""

    def test_pair_is_stored_in_canonical_order(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Пара хранится как note_a < note_b, даже когда расчёт идёт по большей."""
        _seed(
            dim8,
            [
                (3, "Малая", "первый текст", "", _vec(0.90), None),
                (5, "Исходная", "второй текст", "", SOURCE, None),
            ],
        )
        assert _links(monkeypatch).compute_for_note(5) == 1
        assert _pairs() == {(3, 5)}

    def test_recompute_is_idempotent_and_refreshes_marker(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Повторный расчёт не плодит строк и переписывает маркер `links_at`."""
        _seed(
            dim8,
            [
                (1, "Источник", "первый текст", "", SOURCE, None),
                (2, "Близкая", "второй текст", "", _vec(0.90), None),
            ],
        )
        service = _links(monkeypatch)
        assert service.compute_for_note(1) == 1
        first_rows = _rows()
        first_stamp = _links_at(1)
        assert first_stamp is not None

        _set_links_at(1, "2000-01-01T00:00:00Z")
        assert service.compute_for_note(1) == 1
        assert _rows() == first_rows  # ни новых строк, ни дублей
        assert _links_at(1) != "2000-01-01T00:00:00Z"

    def test_stale_rows_are_replaced(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Строки заметки перезаписываются целиком: устаревшая пара исчезает."""
        _seed(
            dim8,
            [
                (1, "Источник", "первый текст", "", SOURCE, None),
                (2, "Прочее", "второй текст", "", _vec(0.10), None),
            ],
        )
        with session(dim8) as conn:
            conn.execute(
                "INSERT INTO links (note_a, note_b, kind, score) "
                "VALUES (1, 2, 'cosine', 0.99)"
            )
        assert _links(monkeypatch).compute_for_note(1) == 0
        assert _rows() == []

    def test_soft_deleted_candidate_is_excluded(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Soft-deleted заметка (даже с близким вектором) в связи не идёт."""
        _seed(
            dim8,
            [
                (1, "Источник", "первый текст", "", SOURCE, None),
                (2, "Корзина", "второй текст", "", _vec(0.99), "2026-01-01T00:00:00Z"),
            ],
        )
        assert _links(monkeypatch).compute_for_note(1) == 0
        assert _rows() == []

    def test_own_namespace_is_not_cut_off_at_compute(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Отсечение «своего» узла — при выдаче (arch §3.5), не при расчёте."""
        _seed(
            dim8,
            [
                (1, "Источник", "первый текст", "", SOURCE, None),
                (2, "Сосед", "второй текст", "", _vec(0.90), None),
            ],
        )
        with session(dim8) as conn:  # одна пара — один узел
            conn.execute("UPDATE notes SET namespace = 'work' WHERE id IN (1, 2)")
        assert _links(monkeypatch).compute_for_note(1) == 1
        assert _pairs() == {(1, 2)}

    def test_trashed_source_keeps_rows(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Расчёт по trash-заметке — 0 и без изменений: строки связей живут."""
        _seed(
            dim8,
            [
                (1, "Источник", "первый текст", "", SOURCE, None),
                (2, "Близкая", "второй текст", "", _vec(0.90), None),
            ],
        )
        service = _links(monkeypatch)
        assert service.compute_for_note(1) == 1
        with session(dim8) as conn:
            conn.execute(
                "UPDATE notes SET deleted_at = '2026-01-01T00:00:00Z' WHERE id = 1"
            )
        assert service.compute_for_note(1) == 0
        assert _pairs() == {(1, 2)}

    def test_unknown_note_gives_zero(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Несуществующий id — 0 без ошибки и без строк."""
        assert _links(monkeypatch).compute_for_note(999) == 0
        assert _rows() == []


class TestPurgeOrphans:
    """`purge_orphans` — гигиена строк физически удалённых заметок."""

    def test_orphan_row_is_deleted_and_normal_rows_kept(
        self, dim8: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(
            dim8,
            [
                (1, "Источник", "первый текст", "", SOURCE, None),
                (2, "Близкая", "второй текст", "", _vec(0.90), None),
            ],
        )
        with session(dim8) as conn:
            conn.execute(
                "INSERT INTO links (note_a, note_b, kind, score) "
                "VALUES (1, 2, 'cosine', 0.9)"
            )
            conn.execute(
                "INSERT INTO links (note_a, note_b, kind, score) "
                "VALUES (1, 999, 'cosine', 0.9)"
            )
        assert _links(monkeypatch).purge_orphans() == 1
        assert _pairs() == {(1, 2)}


class TestMarkerResets:
    """Сброс `links_at` в `NoteService` возвращает заметку в очередь расчёта."""

    @pytest.fixture
    def notes(self, dim8: Settings) -> NoteService:
        return NoteService(dim8, HashEmbedder(dim8.embedding_dim))

    def test_compute_sets_marker(
        self, dim8: Settings, notes: NoteService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Расчёт проставляет маркер (даже когда связей не нашлось)."""
        note_id = notes.save("заметка про сервер и реестр")["id"]
        assert _links_at(note_id) is None
        assert _links(monkeypatch).compute_for_note(note_id) == 0
        assert _links_at(note_id) is not None

    def test_text_update_resets_marker(
        self, dim8: Settings, notes: NoteService
    ) -> None:
        note_id = notes.save("заметка про сервер и реестр")["id"]
        _set_links_at(note_id, "2026-01-01T00:00:00Z")
        notes.update(note_id, text="правленый текст заметки про реестр")
        assert _links_at(note_id) is None

    def test_title_update_resets_marker(
        self, dim8: Settings, notes: NoteService
    ) -> None:
        """Название меняет и вектор, и правила entities/mention — маркер сброшен."""
        note_id = notes.save("заметка про сервер и реестр")["id"]
        _set_links_at(note_id, "2026-01-01T00:00:00Z")
        notes.update(note_id, title="Новое название заметки")
        assert _links_at(note_id) is None

    def test_summary_update_keeps_marker(
        self, dim8: Settings, notes: NoteService
    ) -> None:
        note_id = notes.save("заметка про сервер и реестр")["id"]
        _set_links_at(note_id, "2026-01-01T00:00:00Z")
        notes.update(note_id, summary="краткая сводка заметки")
        assert _links_at(note_id) == "2026-01-01T00:00:00Z"

    def test_namespace_move_keeps_marker(
        self, dim8: Settings, notes: NoteService
    ) -> None:
        """Переезд между узлами маркер не сбрасывает: связи не от узла зависят."""
        NamespaceService(dim8).create("work", "Рабочие заметки.")
        note_id = notes.save("заметка про сервер и реестр")["id"]
        _set_links_at(note_id, "2026-01-01T00:00:00Z")
        notes.update(note_id, namespace="work")
        assert _links_at(note_id) == "2026-01-01T00:00:00Z"

    def test_merge_pair_resets_marker(
        self, dim8: Settings, notes: NoteService
    ) -> None:
        older = notes.save("заметка про сервер и реестр")["id"]
        newer = notes.save("вторая заметка про реестр сервера")["id"]
        _set_links_at(older, "2026-01-01T00:00:00Z")
        _set_links_at(newer, "2026-01-01T00:00:00Z")
        notes.merge_pair(older, "объединённый текст заметки", newer)
        assert _links_at(older) is None
