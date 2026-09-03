"""PromotionService — триггер домена и авто-создание листа (Фаза 10, Шаг 5).

«Модель для моделей» (§5.7): структура растёт системой. Конвейер одного
прогона `run()`:

1. **Триггер — SQL-агрегация, не LLM**: `GROUP BY (domain_hint, subdomain_hint)`
   среди default-заметок; группа с счётчиком ≥ NAMESPACE_PROMOTION_THRESHOLD
   (15) при confidence каждой заметки ≥ NAMESPACE_PROMOTION_MIN_CONFIDENCE
   (0.60) — кандидат на авто-создание листа. `candidates()` — та же агрегация
   для `memory_namespaces.promotion_candidates` (актуальная карта для моделей).
2. **Генерация описания**: LLM по 2–3 суммари группы (Describer — модель
   суммаризации, маленький num_predict, think:false — паттерн классификатора
   Шага 4); контракт описаний ≤2 предложений держит обрезка до 2 предложений.
3. **Косинус-предфильтр антисинонимии**: эмбеддинг описания кандидата против
   описаний всех узлов реестра; косинус ≥ NAMESPACE_SYNONYM_SIMILARITY (0.85)
   → слияние БЕЗ LLM (паттерн Фазы 8: косинус — предфильтр, LLM — пограничную
   зону).
4. **Судья структуры — LLM-гейт перед созданием** (StructureJudge — модель
   судьи дедупа, отдельный промпт): (1) пограничная антисинонимия —
   вердикт «СЛИТЬ <path>»; (2) осмысленность слага/описания — «ОТКЛОНИТЬ»;
   иначе «СОЗДАТЬ». Вердиктует модель, не человек.
5. **Действие**: create → provisional-лист (только листья внутри
   существующих корней — `NamespaceService.create` корни не создаёт);
   merge → ретро-перекладка в канонический узел одним UPDATE с канонизацией
   hint; reject → запись вердикта, заметки остаются в default (честно-общие).
   Ретро-перекладка — один UPDATE (+vector_status='pending' — пере-кодировка
   векторов в партицию нового узла штатной очередью).

**Cooldown** (бриф «лимиты/cooldown»): три механизма — (а) запись вердикта
в `promotions` (merged/rejected): группа больше не дёргает describer/судью
(иначе отклонённый кандидат зациклил бы LLM-вызовы при каждом прогоне);
(б) существование узла: группа с уже зарегистрированным путём — не кандидат;
(в) лимиты защиты от шторма: NAMESPACE_AUTO_MAX_PER_DAY (provisional-узлы,
созданные за сегодня по UTC) и NAMESPACE_MAX_LEAVES_PER_DOMAIN (потолок
листов в корне) — кандидаты с превышением пропускаются с логом до вызова
LLM. Созданный узел записи в `promotions` не требует: ретро-перекладка
уводит группу из default, а новые высокоуверенные default-заметки с тем же
hint переезжают в узел штатной причёской Шага 4 (узел уже существует).

Границы автономии (§5.7): домен hint'а обязан быть зарегистрированным корнем
(новые корни — только оператор; незарегистрированный hint — сигнал в логах
«в default копится контент вне известных корней»); авто — только листья.
Отказ любого шага (DescriberError/StructureJudgeError/EmbeddingError) данные
не портит: кандидат остаётся, повтор — при следующем прогоне (NFR-3), группу
будит следующая классификация default-заметки в воркере.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.namespaces import (
    NamespaceService,
    count_sentences,
    normalize_slug,
)
from app.services.ollama_gate import ollama_slot
from app.storage.db import DEFAULT_NAMESPACE, session, transaction

# Описания кандидата строятся по суммари: топ по confidence, детерминированно.
SUMMARIES_PER_CANDIDATE = 3

# Кусок тела ответа в тексте ошибки (логи не захламляем).
_ERROR_BODY_CHARS = 120

# Таймауты: connect 2 с (LAN), чтение 30 с — короткие ответы (как
# классификатор Шага 4: маленький num_predict, reasoning не нужен).
CONNECT_TIMEOUT_SEC = 2.0
PROMOTION_TIMEOUT_SEC = 30

# Параметры вызова, не настраиваемые env: маленькие бюджеты, нулевая
# температура — детерминированные вердикты; keep_alive — модель живёт между
# операциями (описания и вердикты редки).
DESCRIBE_NUM_PREDICT = 128
TEMPERATURE = 0.1
KEEP_ALIVE = "15m"


class DescriberError(RuntimeError):
    """Генератор описания не дал текста: сервер недоступен или ответ пуст."""


class StructureJudgeError(RuntimeError):
    """Судья структуры не дал вердикта: сервер недоступен или ответ некорректен."""


class Verdict:
    """Вердикт судьи структуры: action create|merge|reject + цель слияния."""

    __slots__ = ("action", "target")

    def __init__(self, action: str, target: str | None = None) -> None:
        self.action = action
        self.target = target


class Describer(Protocol):
    """Контракт генератора описаний узлов: природы реализации он не знает."""

    def describe(
        self, summaries: list[str], slug: str, domain: str
    ) -> str:
        """Описание нового узла по суммари группы (≤2 предложений)."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с остальными клиентами)."""
        ...


class StructureJudge(Protocol):
    """Контракт судьи структуры (гейт перед созданием, §5.7)."""

    def review(
        self,
        description: str,
        slug: str,
        domain: str,
        existing: list[dict[str, Any]],
        nearest_path: str | None,
        nearest_cosine: float | None,
    ) -> Verdict:
        """Вердикт по кандидату: СОЗДАТЬ / СЛИТЬ <path> / ОТКЛОНИТЬ."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с остальными клиентами)."""
        ...


    # --- генератор описания (модель суммаризации, паттерн классификатора) ------

DESCRIBE_SYSTEM_PROMPT = (
    "Ты генератор описаний разделов иерархической памяти. По примерам "
    "заметок напиши описание раздела: какие заметки в нём живут. Строго "
    "1–2 коротких предложения, без списков и пояснений — только описание."
)

DESCRIBE_USER_TEMPLATE = (
    "Новый подраздел: {domain}/{slug}\n\n"
    "Примеры заметок раздела (краткие содержания):\n{summaries}"
)


class DescriptionService:
    """Описание узла через Ollama `POST /api/chat` (модель суммаризации).

    Отдельный маленький вызов (think:false, num_predict 128) — паттерн
    классификатора Шага 4: та же модель, что суммаризация, слот общий
    (ollama_gate сериализует вызовы к одному base_url). Слишком длинный
    ответ обрезается до 2 предложений — контракт описаний (решение О.)
    соблюдается механикой, а не надеждой на модель.
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
    ) -> None:
        self._settings = settings
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            transport = httpx.MockTransport(transport)
        self._client = httpx.Client(
            base_url=settings.summary_ollama_base_url,
            timeout=httpx.Timeout(PROMOTION_TIMEOUT_SEC, connect=CONNECT_TIMEOUT_SEC),
            transport=transport,
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None

    def describe(self, summaries: list[str], slug: str, domain: str) -> str:
        """Описание нового узла по суммари группы; отказ — DescriberError."""
        if not summaries:
            raise ValueError("describe: ожидается непустой список суммари")
        try:
            content = self._chat(
                DESCRIBE_SYSTEM_PROMPT,
                DESCRIBE_USER_TEMPLATE.format(
                    domain=domain,
                    slug=slug,
                    summaries="\n".join(f"- {s}" for s in summaries),
                ),
            )
        except DescriberError:
            self.last_attempt_ok = False
            raise
        self.last_attempt_ok = True
        return self._trim(content)

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    # --- внутреннее ---------------------------------------------------------

    def _chat(self, system_prompt: str, user_text: str) -> str:
        """Один вызов /api/chat + проверки контракта ответа (без ретраев)."""
        payload = {
            "model": self._settings.summary_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "num_predict": DESCRIBE_NUM_PREDICT,
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
            "think": False,  # короткое описание без рассуждений
        }
        try:
            # Очередь F1: один запрос к серверу в момент времени (та же
            # модель, что суммаризация — делим слот, не гоняем параллельно).
            with ollama_slot(self._settings.summary_ollama_base_url):
                response = self._client.post("/api/chat", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise DescriberError(
                "сервер генерации описаний недоступен "
                f"({self._settings.summary_ollama_base_url}): {exc}"
            ) from exc
        if response.status_code != 200:
            body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
            raise DescriberError(f"HTTP {response.status_code} от /api/chat: {body}")
        try:
            data = response.json()
        except ValueError as exc:
            raise DescriberError(f"не-JSON ответ от /api/chat: {exc}") from exc
        if not isinstance(data, dict):
            raise DescriberError("неожиданный формат ответа /api/chat (не объект)")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise DescriberError("пустой content в ответе генератора описаний")
        return content.strip()

    @staticmethod
    def _trim(content: str) -> str:
        """Нормализовать и обрезать до 2 предложений (контракт описаний).

        Модель может проигнорировать «1–2 предложения» — контракт держим
        обрезкой: первые два предложения, остальное отбрасывается. Пустой
        результат — DescriberError (узла без описания не бывает).
        """
        text = " ".join(content.split())
        if not text:
            raise DescriberError("пустое описание узла")
        sentences = [part.strip() for part in re.split(r"[.!?]+(?:\s|$)", text)]
        sentences = [part for part in sentences if part]
        if not sentences:
            raise DescriberError(f"описание без предложений: {content[:120]}")
        if len(sentences) > 2:
            sentences = sentences[:2]
        trimmed = ". ".join(sentences)
        if not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        if count_sentences(trimmed) > 2:  # страховка: контракт ≤2 предложений
            raise DescriberError(f"описание не обрезается до 2 предложений: {trimmed[:120]}")
        return trimmed


# --- судья структуры (модель судьи дедупа, паттерн Фазы 8) -----------------

JUDGE_SYSTEM_PROMPT = (
    "Ты судья структуры иерархической памяти. Проверь кандидата на новый "
    "подраздел. Правила: (1) если смысл кандидата совпадает с существующим "
    "тематическим узлом (та же тема другими словами) — это слияние, а не "
    "новый узел; (2) слаг и описание должны быть содержательными: "
    "бессмысленный, мусорный или пустой по смыслу кандидат — отклонить. "
    "Ответь строго одной отметкой без пояснений: СОЗДАТЬ — кандидат новый "
    "и осмысленный; СЛИТЬ <path> — кандидат дублирует существующий узел, "
    "в качестве path укажи ТОЛЬКО тематический путь из списка «Существующие "
    "узлы» (никогда — путь кандидата; default — системный своп, слияние с "
    "ним не бывает); ОТКЛОНИТЬ — кандидат бессмысленный."
)

JUDGE_USER_TEMPLATE = (
    "Кандидат: {domain}/{slug} — {description}\n\n"
    "Существующие узлы:\n{nodes}\n\n"
    "Ближайший по векторному сходству: {nearest}"
)


# Путь-цель вердикта СЛИТЬ: слаги латиница/цифры/дефис, максимум 2 уровня.
VERDICT_PATH_RE = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)?"
)


class StructureJudgeService:
    """Вердикт судьи структуры через Ollama `POST /api/chat` (non-stream).

    Модель судьи дедупа (DEDUP_JUDGE_MODEL, REQUIREMENTS §5.7 «та же модель
    судьи дедупа, отдельный промпт»); параметры вызова — как у JudgeService
    (Фаза 8): think из DEDUP_JUDGE_THINK, бюджет DEDUP_JUDGE_NUM_PREDICT.
    Отказ судьи (транспорт) — StructureJudgeError: кандидат остаётся без
    вердикта и повторяется при следующем прогоне (NFR-3). Плохой вердикт
    (СЛИТЬ без узла) — тоже StructureJudgeError: недоопределённое решение
    не превращаем в создание.
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
    ) -> None:
        self._settings = settings
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            transport = httpx.MockTransport(transport)
        self._client = httpx.Client(
            base_url=settings.dedup_judge_ollama_base_url,
            timeout=httpx.Timeout(
                settings.dedup_judge_timeout_sec, connect=CONNECT_TIMEOUT_SEC
            ),
            transport=transport,
        )
        self.last_attempt_ok: bool | None = None

    def review(
        self,
        description: str,
        slug: str,
        domain: str,
        existing: list[dict[str, Any]],
        nearest_path: str | None,
        nearest_cosine: float | None,
    ) -> Verdict:
        """Вердикт по кандидату; любой отказ — StructureJudgeError."""
        try:
            content = self._chat(
                JUDGE_SYSTEM_PROMPT,
                JUDGE_USER_TEMPLATE.format(
                    domain=domain,
                    slug=slug,
                    description=description,
                    nodes="\n".join(
                        f"- {node['path']}: {node['description']}"
                        for node in existing
                    )
                    or "(нет)",
                    nearest=(
                        f"{nearest_path} (косинус {nearest_cosine:.2f})"
                        if nearest_path is not None
                        else "нет близких узлов"
                    ),
                ),
            )
        except StructureJudgeError:
            self.last_attempt_ok = False
            raise
        self.last_attempt_ok = True
        return self._parse(content)

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    # --- внутреннее ---------------------------------------------------------

    def _payload(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        """Тело вызова /api/chat (параметры — как у судьи дедупа, Фаза 8)."""
        payload: dict[str, Any] = {
            "model": self._settings.dedup_judge_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "num_predict": self._settings.dedup_judge_num_predict,
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
        }
        if not self._settings.dedup_judge_think:
            payload["think"] = False
        return payload

    def _chat(self, system_prompt: str, user_text: str) -> str:
        """Один вызов /api/chat + проверки контракта ответа (без ретраев)."""
        try:
            # Слот общий с судьёй дедупа (тот же base_url) — не гоняем
            # параллельные вызовы на один Ollama-сервер.
            with ollama_slot(self._settings.dedup_judge_ollama_base_url):
                response = self._client.post(
                    "/api/chat", json=self._payload(system_prompt, user_text)
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise StructureJudgeError(
                "сервер судьи структуры недоступен "
                f"({self._settings.dedup_judge_ollama_base_url}): {exc}"
            ) from exc
        if response.status_code != 200:
            body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
            raise StructureJudgeError(
                f"HTTP {response.status_code} от /api/chat: {body}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise StructureJudgeError(f"не-JSON ответ от /api/chat: {exc}") from exc
        if not isinstance(data, dict):
            raise StructureJudgeError("неожиданный формат ответа /api/chat (не объект)")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise StructureJudgeError("пустой content в ответе судьи структуры")
        return content.strip()

    @staticmethod
    def _parse(content: str) -> Verdict:
        """Разбор отметки СОЗДАТЬ / СЛИТЬ <path> / ОТКЛОНИТЬ.

        Markdown-жирный стрипается, регистр не учитывается. «СЛИТЬ» требует
        узла-цели в ответе: путь извлекается из исходного content (не из
        upper) регэкспом слагов; без пути — отказ (недоопределённый вердикт
        не превращаем в создание). Ответ без отметки — StructureJudgeError.
        """
        upper = " ".join(content.replace("*", " ").upper().split())
        if "ОТКЛОНИТЬ" in upper:
            return Verdict("reject")
        if "СЛИТЬ" in upper:
            match = VERDICT_PATH_RE.search(content.replace("*", ""))
            if match is None:
                raise StructureJudgeError(
                    f"вердикт СЛИТЬ без узла-цели: {content[:120]}"
                )
            return Verdict("merge", match.group(0))
        if "СОЗДАТЬ" in upper:
            return Verdict("create")
        raise StructureJudgeError(
            f"судья структуры не дал вердикт СОЗДАТЬ/СЛИТЬ/ОТКЛОНИТЬ: {content[:120]}"
        )


# --- триггер домена ----------------------------------------------------------


class PromotionService:
    """Триггер + авто-создание листа: агрегация hint-групп → гейт → действие.

    Зависимости инъектируются (DI): embedding — косинус-предфильтр,
    describer — генерация описания, judge — гейт перед созданием,
    namespaces — реестр (общий экземпляр с воркером/NoteService).
    describer/judge None — триггер не запускается (тестовый режим, как
    classifier=None у воркера).
    """

    def __init__(
        self,
        settings: Settings,
        embedding: Any,
        describer: Describer | None = None,
        judge: StructureJudge | None = None,
        namespaces: NamespaceService | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._describer = describer
        self._judge = judge
        self._namespaces = namespaces if namespaces is not None else NamespaceService(settings)

    # --- чтение: кандидаты (SQL-агрегация, §5.7 «триггер — не LLM») ----------

    def candidates(self) -> list[dict[str, Any]]:
        """Группы default-заметок, доросшие до порога и без вердикта.

        SQL-агрегация: счётчик ≥ NAMESPACE_PROMOTION_THRESHOLD при confidence
        каждой заметки ≥ NAMESPACE_PROMOTION_MIN_CONFIDENCE; далее фильтры
        cooldown: домен hint'а зарегистрирован (корни — оператор), узел ещё
        не создан, вердикта merged/rejected нет. Сортировка по убыванию
        счётчика — большие группы первыми, детерминированно.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.domain_hint AS domain_hint, n.subdomain_hint AS subdomain_hint, "
                "COUNT(*) AS cnt, ROUND(AVG(n.confidence), 2) AS avg_confidence "
                "FROM notes n "
                "WHERE n.namespace = 'default' AND n.deleted_at IS NULL "
                "AND n.domain_hint IS NOT NULL AND n.subdomain_hint IS NOT NULL "
                "AND n.confidence >= ? "
                "GROUP BY n.domain_hint, n.subdomain_hint "
                "HAVING COUNT(*) >= ? "
                "ORDER BY cnt DESC, n.domain_hint, n.subdomain_hint",
                (
                    self._settings.namespace_promotion_min_confidence,
                    self._settings.namespace_promotion_threshold,
                ),
            ).fetchall()
        decided = self._decided_hints()
        result: list[dict[str, Any]] = []
        for row in rows:
            domain = normalize_slug(row["domain_hint"])
            slug = normalize_slug(row["subdomain_hint"])
            if domain is None or slug is None:
                continue  # мусорный hint классификатора — не кандидат
            if (domain, slug) in decided:
                continue  # cooldown: вердикт судьи уже вынесен
            if not self._namespaces.exists(domain):
                continue  # новые корни — оператор (§5.7); сигнал — в run()
            path = f"{domain}/{slug}"
            if self._namespaces.exists(path):
                continue  # узел уже есть — группа разберётся причёской
            result.append(
                {
                    "domain": domain,
                    "subdomain": slug,
                    "count": int(row["cnt"]),
                    "avg_confidence": float(row["avg_confidence"]),
                }
            )
        return result

    # --- прогон конвейера ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Прогнать всех готовых кандидатов; сводка {created, merged, rejected}.

        Вызывается воркером после классификации default-заметки (разметка
        могла докинуть группу до порога). Отказ на одном кандидате не
        отменяет остальных (NFR-3): кандидат пропускается до следующего
        прогона (в сводку не попадает). Лимиты дня/листов проверяются до
        LLM-вызовов.
        """
        report: dict[str, Any] = {"created": [], "merged": [], "rejected": []}
        if self._describer is None or self._judge is None:
            return report  # триггер отключён (тестовый режим)
        logger = logging.getLogger("app")
        self._signal_root_orphans()
        day_limit = self._day_limit_reached()
        for candidate in self.candidates():
            domain, slug = candidate["domain"], candidate["subdomain"]
            path = f"{domain}/{slug}"
            if day_limit:
                logger.warning(
                    "promotion: daily limit reached — candidate skipped",
                    extra={"event": "promotion_skipped", "path": path,
                           "reason": "daily_limit"},
                )
                continue
            if self._leaves_limit_reached(domain):
                logger.warning(
                    "promotion: leaves limit reached — candidate skipped",
                    extra={"event": "promotion_skipped", "path": path,
                           "reason": "leaves_limit"},
                )
                continue
            action = self._promote_one(domain, slug)
            if action in ("created", "merged", "rejected"):
                report[action].append(path)
            if action == "created":
                day_limit = self._day_limit_reached()
        return report

    def _signal_root_orphans(self) -> None:
        """Сигнал оператору: в default копится контент вне известных корней (§5.7).

        Новые корни система сама не создаёт (границы автономии): если
        разметка причёски стабильно указывает на незарегистрированный домен
        и группа доросла до порога — оператор решает, быть ли такому корню
        (REST Шаг 6). Сигнал — по порогу триггера (шум отсечён), в логах
        (event=root_orphans); в memory_namespaces не выносим — структурная
        сигнализация остаётся операторской.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.domain_hint AS domain_hint, COUNT(*) AS cnt "
                "FROM notes n WHERE n.namespace = 'default' AND n.deleted_at IS NULL "
                "AND n.domain_hint IS NOT NULL "
                "AND n.confidence >= ? "
                "GROUP BY n.domain_hint HAVING COUNT(*) >= ?",
                (
                    self._settings.namespace_promotion_min_confidence,
                    self._settings.namespace_promotion_threshold,
                ),
            ).fetchall()
        orphans = [
            {"domain": row["domain_hint"], "count": int(row["cnt"])}
            for row in rows
            if not self._namespaces.exists(normalize_slug(row["domain_hint"]) or "")
        ]
        if orphans:
            logging.getLogger("app").warning(
                "promotion: default accumulates content outside known roots",
                extra={"event": "root_orphans", "orphans": orphans},
            )

    def _promote_one(self, domain: str, slug: str) -> str | None:
        """Полный цикл одного кандидата: created|merged|rejected|None.

        None — отказ describer/судьи (транспорт): кандидат остаётся БЕЗ
        вердикта, повтор при следующем прогоне (NFR-3); в сводке не
        считается. Вернувшийся вердикт всегда записывается (cooldown):
        merged/rejected — в promotions, created — узлом реестра.
        """
        logger = logging.getLogger("app")
        summaries = self._group_summaries(domain, slug)
        try:
            description = self._describer.describe(summaries, slug, domain)  # type: ignore[union-attr]
        except DescriberError:
            logger.warning(
                "promotion: description generation failed — candidate kept",
                extra={"event": "promotion_failed", "domain": domain, "slug": slug,
                       "reason": "describer"},
            )
            return None  # без описания кандидата нет: повтор — следующий прогон
        nearest_path, nearest_cosine = self._nearest_node(description)
        if nearest_cosine is not None and (
            nearest_cosine >= self._settings.namespace_synonym_similarity
        ):
            # Косинус-предфильтр: слияние без LLM (паттерн Фазы 8).
            self._merge(domain, slug, nearest_path)  # type: ignore[arg-type]
            return "merged"
        try:
            verdict = self._judge.review(
                description,
                slug,
                domain,
                self._thematic_nodes(),
                nearest_path,
                nearest_cosine,
            )
        except StructureJudgeError:
            logger.warning(
                "promotion: structure judge failed — candidate kept",
                extra={"event": "promotion_failed", "domain": domain, "slug": slug,
                       "reason": "judge"},
            )
            return None
        if verdict.action == "merge":
            target = self._namespaces.validate_path(verdict.target or "")
            if target == f"{domain}/{slug}":
                # Судья «слил» кандидата с ним самим — вердикт некорректен:
                # кандидата не создаём и не запрещаем навсегда (записи нет),
                # повтор — следующий прогон; стабильно мусорные ответы видны
                # в логах (promotion_failed, reason=judge_self_merge).
                logger.warning(
                    "promotion: judge merged candidate into itself — bad verdict",
                    extra={"event": "promotion_failed", "domain": domain,
                           "slug": slug, "reason": "judge_self_merge"},
                )
                return None
            if target == DEFAULT_NAMESPACE:
                # Слияние с default бессмысленно: кандидат в нём и лежит.
                # Вердикт не записываем (путаница, а не решение); повтор —
                # следующий прогон, мусор виден в логах.
                logger.warning(
                    "promotion: judge merged candidate into default — bad verdict",
                    extra={"event": "promotion_failed", "domain": domain,
                           "slug": slug, "reason": "judge_merge_default"},
                )
                return None
            if not self._namespaces.exists(target):
                # Судья назвал незарегистрированный узел: вердикт ненадёжен,
                # но это его РЕШЕНИЕ (не отказ транспорта) — фиксируем как
                # reject, чтобы группа не дёргала судью повторно.
                logger.warning(
                    "promotion: judge merge target unknown — recorded as rejected",
                    extra={"event": "promotion_rejected", "domain": domain,
                           "slug": slug, "target": target},
                )
                self._record(domain, slug, "rejected")
                return "rejected"
            self._merge(domain, slug, target)
            return "merged"
        if verdict.action == "reject":
            self._record(domain, slug, "rejected")
            logger.info(
                "promotion: structure judge rejected candidate",
                extra={"event": "promotion_rejected", "domain": domain, "slug": slug,
                       "description": description},
            )
            return "rejected"
        self._create(domain, slug, description)
        return "created"

    def _create(self, domain: str, slug: str, description: str) -> None:
        """Создать provisional-лист и переложить группу (один UPDATE)."""
        path = f"{domain}/{slug}"
        self._namespaces.create(path, description, status="provisional")
        moved = self._retro_move(domain, slug, path)
        logging.getLogger("app").info(
            "promotion: provisional leaf created",
            extra={"event": "node_created", "path": path, "moved": moved,
                   "status": "provisional"},
        )

    def _merge(self, domain: str, slug: str, canonical: str) -> None:
        """Слияние кандидата с каноническим узлом (один UPDATE + вердикт).

        Заметки группы переехали — hint канонизируется (subdomain_hint =
        слаг канонического листа; корень → NULL: «общая для домена»).
        """
        moved = self._retro_move(domain, slug, canonical)
        self._record(domain, slug, "merged", canonical)
        logging.getLogger("app").info(
            "promotion: candidate merged into existing node",
            extra={"event": "node_merged", "canonical": canonical,
                   "hint": f"{domain}/{slug}", "moved": moved},
        )

    # --- SQL-механика --------------------------------------------------------

    def _thematic_nodes(self) -> list[dict[str, Any]]:
        """Тематические узлы реестра (без системного свопа default).

        Косинус-предфильтр и судья сравнивают кандидата только с ними:
        слияние с default бессмысленно (кандидат в нём и лежит), а своп в
        списке кандидатов-на-слияние только путает вердикт.
        """
        return [
            node
            for node in self._namespaces.list_all()["namespaces"]
            if node["path"] != DEFAULT_NAMESPACE
        ]

    def _retro_move(self, domain: str, slug: str, canonical: str) -> int:
        """Ретро-перекладка группы в канонический узел ОДНИМ UPDATE (§5.7).

        Канонизация hint: лист → subdomain_hint = слаг листа; корень →
        NULL (общая для домена). vector_status='pending' — штатная
        пере-кодировка векторов в партицию нового узла (воркер). Возврат —
        число переложенных заметок.
        """
        segments = canonical.split("/")
        target_subdomain = segments[1] if len(segments) == 2 else None
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET namespace = ?, subdomain_hint = ?, "
                "vector_status = 'pending' "
                "WHERE namespace = 'default' AND deleted_at IS NULL "
                "AND domain_hint = ? AND subdomain_hint = ?",
                (canonical, target_subdomain, domain, slug),
            )
            return cursor.rowcount

    def _record(
        self, domain: str, slug: str, status: str, canonical: str | None = None
    ) -> None:
        """Записать вердикт судьи (cooldown: группа больше не кандидат)."""
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO promotions (domain, subdomain, status, canonical_path) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(domain, subdomain) DO UPDATE SET "
                "status = excluded.status, canonical_path = excluded.canonical_path, "
                "decided_at = excluded.decided_at",
                (domain, slug, status, canonical),
            )

    def _decided_hints(self) -> set[tuple[str, str]]:
        """Группы с вынесенным вердиктом (cooldown: не дёргаем судью)."""
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT domain, subdomain FROM promotions"
            ).fetchall()
        return {(row["domain"], row["subdomain"]) for row in rows}

    def _group_summaries(self, domain: str, slug: str) -> list[str]:
        """Топ-3 суммари группы (по confidence, затем id) — вход описания."""
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT summary FROM notes "
                "WHERE namespace = 'default' AND deleted_at IS NULL "
                "AND domain_hint = ? AND subdomain_hint = ? "
                "AND summary != '' "
                "ORDER BY confidence DESC, id LIMIT ?",
                (domain, slug, SUMMARIES_PER_CANDIDATE),
            ).fetchall()
        return [row["summary"] for row in rows]

    def _nearest_node(self, description: str) -> tuple[str | None, float | None]:
        """Косинус-предфильтр: ближайший узел по описанию (эмбеддинги).

        Описание кандидата + описания тематических узлов реестра — одним
        батчем embed_texts (описаний мало — узлов 3–7). default исключён:
        слияние кандидата со свопом бессмысленно, он в нём и лежит. Возврат
        (path, cosine) ближайшего или (None, None), если реестр пуст/
        эмбеддинг отказал. Отказ кодирования (EmbeddingError) — предфильтр
        просто не находит ближайшего: гейт остаётся судье (деградация, не
        отказ конвейера).
        """
        nodes = self._thematic_nodes()
        if not nodes:
            return None, None
        try:
            vectors = self._embedding.embed_texts(
                [description] + [node["description"] for node in nodes]
            )
        except Exception:
            logging.getLogger("app").warning(
                "promotion: embedding failed — cosine prefilter skipped",
                extra={"event": "promotion_prefilter_skipped"},
            )
            return None, None
        candidate_vec, node_vecs = vectors[0], vectors[1:]
        best_index = max(
            range(len(node_vecs)),
            key=lambda i: sum(a * b for a, b in zip(candidate_vec, node_vecs[i])),
        )
        cosine = sum(
            a * b for a, b in zip(candidate_vec, node_vecs[best_index])
        )
        return nodes[best_index]["path"], cosine

    def _day_limit_reached(self) -> bool:
        """NAMESPACE_AUTO_MAX_PER_DAY: provisional-узлы, созданные сегодня (UTC)."""
        with session(self._settings) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM namespaces "
                "WHERE status = 'provisional' "
                "AND created_at >= strftime('%Y-%m-%dT00:00:00Z','now')"
            ).fetchone()[0]
        return int(count) >= self._settings.namespace_auto_max_per_day

    def _leaves_limit_reached(self, domain: str) -> bool:
        """NAMESPACE_MAX_LEAVES_PER_DOMAIN: потолок листов в корне."""
        with session(self._settings) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM namespaces WHERE path LIKE ? || '/%'",
                (domain,),
            ).fetchone()[0]
        return int(count) >= self._settings.namespace_max_leaves_per_domain