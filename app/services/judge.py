"""JudgeService — LLM-судья дедупа (Фаза 8, Этап 3.1; решение Олега 2026-08-30).

Третий внешний вызов системы: chat слота judge, модель JUDGE_MODEL
(`ornith-1.5:35b`). Решает по паре текстов «дубль/не дубль» — та же мысль,
пересказанная другими словами. Фоновый дедуп (Фаза 8, Этап 2) остаётся
предфильтром: судью спрашивают только про косинус-кандидатов (топ-N,
DEDUP_CANDIDATE_*), а не про все заметки (бриф Фазы 8, решение Олега
«косинус как предфильтр»).

Фаза 11: HTTP-транспорт вынесен в `LLMClient` (app/services/llm_client.py)
— один клиент на слот judge (провайдер `ollama` | `openai`, per-slot
адрес/модель/ключ из Settings, решение №1; блок env переименован
DEDUP_JUDGE_* → JUDGE_*). Сервис больше не держит httpx и не собирает
пейлоады сам: system+user, non-stream, `judge_num_predict` и think уходят
через `LLMClient.chat`; очередь F1 (ollama_slot — только ollama-провайдер),
Bearer при заданном ключе и разбор ответа (`message.content`, reasoning-
поля отбрасываются) — внутри клиента. keep_alive не отправляется никому
(решение №6: моделью управляет сервер, OLLAMA_KEEP_ALIVE).

Модель вернула проверку (2026-08-30, бриф Фазы 8): вердикт в `message.content`
маркированным жирным — `**ДУБЛЬ**` / `**НЕ ДУБЛЬ**` (markdown `**` стрипается,
далее ищется подстрока; сначала «НЕ ДУБЛЬ» — иначе попадёт в «ДУБЛЬ»).
Ответы без вердикта — отказ (безопаснее «не сводить»): воркер оставит обе
заметки и повторит по back-off (NFR-3).

`think` в запросе — история решений: на Этапе 3.1 судья не думал
(быстро); после сквозного теста Фазы 8 (2026-08-30) Олег включил
размышления — бездумный вердикт давал ложные ДУБЛИ на длинных
однопроектных заметках (думающий судья на той же паре ответил
корректно, 46 с). При JUDGE_THINK=false (дефолт кода) в тело идёт
явное `"think": false`; при true (прод) — поле не отправляется, а
пришедшее `message.thinking` всё равно отбрасывается (читаем только
content — как в суммаризаторе); бюджет thinking+вердикта —
JUDGE_NUM_PREDICT (прод 1024: замер ~1200 токенов, done_reason=stop).

Ключевые решения (по образцу SummaryService, Этап 2.2):
- Синхронный httpx-транспорт внутри клиента — воркер исполняет его в
  `asyncio.to_thread`; connect 2 с (LAN), чтение — JUDGE_TIMEOUT_SEC.
  Дефолт кода (30 с из брифа) устарел: E2E Фазы 8 показал 46 с думающего
  судьи на длинной паре (промпт ~2.9k токенов), худший живой вызов —
  ~121 с — прод ставит 300 с (решение О. 2026-08-30).
  Очередь ollama_slot (F1) гарантирует: read-таймаут тикает только с
  реальной отправки запроса — ожидание за merge-генерацией не съедает
  бюджет.
- Ретраев нет — повтор решает воркер (back-off очереди, NFR-3).
- Отказ — `JudgeError` и только она: HTTP не-200, не-JSON, пустой content,
  вердикт не распознан, транспорт недоступен. Текст ошибки клиента
  сохраняется (включая hint про {SLOT}_API_KEY у auth-отказа, решение №4).
- `last_attempt_ok` (NFR-4, как у суммаризатора) — источник статуса в /health;
  стартовая проверка слота (main.py, решение №5) инициализирует его до
  первого реального вызова.
- Промпты — из реестра (Фаза 11, решение №7): системный `judge_system` —
  редактируемый (маркеры «ДУБЛЬ»/«НЕ ДУБЛЬ» валидируются при старте —
  вердикты парсятся именно по ним), user-шаблон `judge_user` — зашитая
  константа; `registry` инъектируется сборкой (build_services), без DI —
  встроенные тексты реестра.
- DI: юнит-тесты — httpx.MockTransport через `transport`; фейковые судьи
  (tests/fakes.py) — для тестов интеграции в дедуп (Задача 3.2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError, SlotSpec
from app.services.prompts import PromptRegistry


class JudgeError(RuntimeError):
    """Судья не дал вердикта: сервер недоступен или ответ некорректен."""


class Judge(Protocol):
    """Контракт судьи дедупа: природы реализации он не знает.

    Реализации: JudgeService (живой слот judge) и тестовые фейки
    (tests/fakes.py) — воркер зависит от протокола, а не от класса клиента.
    """

    def judge(self, text_new: str, text_candidate: str) -> bool:
        """Вердикт по паре текстов: True — дубль, False — не дубль."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с остальными клиентами)."""
        ...


class JudgeService:
    """Вердикт «дубль/не дубль» через chat слота judge (non-stream)."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._prompts = registry if registry is not None else PromptRegistry()
        # DI сборки (общий клиент слота judge) или свой клиент с
        # transport-инъекцией для юнит-тестов (MockTransport).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(SlotSpec.for_judge(settings), transport=transport)
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None

    def judge(self, text_new: str, text_candidate: str) -> bool:
        """Спросить судью про пару текстов; True — «ДУБЛЬ», False — «НЕ ДУБЛЬ».

        Любой отказ — JudgeError: воркер оставит пару нетронутой и повторит
        по back-off (NFR-3) — лучше обе заметки, чем ошибочное сведение.
        """
        if not text_new or not text_new.strip():
            raise ValueError("judge: пустая заметка недопустима")
        if not text_candidate or not text_candidate.strip():
            raise ValueError("judge: пустая заметка недопустима")
        try:
            content = self._llm.chat(
                self._prompts.judge_system,
                self._prompts.judge_user.format(
                    text_new=text_new, text_candidate=text_candidate
                ),
                num_predict=self._settings.judge_num_predict,
            )
        except LLMError as exc:
            # Текст клиента сохраняется: HTTP-статус, адрес слота, hint
            # про {SLOT}_API_KEY у auth-отказа (решение №4).
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise JudgeError(str(exc)) from exc
        try:
            verdict = self._verdict(content)
        except JudgeError:
            self.last_attempt_ok = False  # нераспознанный вердикт — тоже отказ
            raise
        self.last_attempt_ok = True
        return verdict

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._llm.close()

    # --- внутреннее ---------------------------------------------------------

    @staticmethod
    def _verdict(content: str) -> bool:
        """Парсинг вердикта из content (формат брифа: `**ДУБЛЬ**/…`).

        Markdown-жирный стрипается, регистр не учитывается; сначала ищем
        «НЕ ДУБЛЬ» (подстрока содержит «ДУБЛЬ»), затем «ДУБЛЬ». Ответ без
        вердикта — отказ: неопределённость не превращаем в «не дубль».
        """
        normalized = " ".join(content.replace("*", " ").upper().split())
        if "НЕ ДУБЛЬ" in normalized:
            return False
        if "ДУБЛЬ" in normalized:
            return True
        raise JudgeError(f"судья не дал вердикт ДУБЛЬ/НЕ ДУБЛЬ: {content[:120]}")