"""PromptRegistry — реестр промптов (Фаза 11, решение №7).

Все тексты промптов собираются в одном месте. Десять промптов делятся на
две группы (решение О. 2026-09-04: «часть промптов — наша задача довести
до ума, их не правит никто»):

- **3 редактируемых** (`summary_system`, `summary_merge_system`,
  `judge_system`): при заданном `prompts_dir` выносятся в файлы
  (seed-if-missing — создаются с встроенным дефолтом как стартовым
  текстом) и правятся оператором без пересборки образа; существующие
  файлы НЕ перезаписываются; пустой файл = встроенный дефолт; непустой
  файл побеждает;
- **7 зашитых** (`merge_user`, `judge_user`, `classifier_system`,
  `describe_system`, `describe_user`, `structure_judge_system`,
  `structure_judge_user`): только константы в коде, файлами не создаются
  никогда.

Контрактная защита: финальный `judge_system` (встроенный или из файла)
обязан содержать маркеры «ДУБЛЬ» и «НЕ ДУБЛЬ» — вердикты дедупа парсятся
по ним (app/services/judge.py::_verdict), и файл без маркеров молча уронил
бы дедуп на отказах парсера. Нарушение — фатальный `ConfigError` при
конструировании реестра (старт сервиса падает с понятным сообщением).

`prompts_dir` не задан (None) — файловая механика не активна вовсе:
работают только встроенные константы.

Потребитель — пул 4 (клиенты слотов summary/judge берут системные промпты
из реестра, user-шаблоны — зашитые константы реестра).
"""

from __future__ import annotations

import os
from pathlib import Path

from app.config import ConfigError

# --- Встроенные тексты -------------------------------------------------------
# Перенесены 1-в-1 из app/services/summary.py, judge.py, classifier.py,
# promotion.py (константы-источники остаются на месте до перевода клиентов
# на реестр — пул 4; см. test_builtin_texts_match_current_constants).

# Редактируемые 3 (seed-if-missing при заданном prompts_dir).

# summary.py::SYSTEM_PROMPT — пересказ заметки.
SUMMARY_SYSTEM_PROMPT = (
    "Summarize the note in 1–2 short, dense sentences, no more than 30 "
    "words in total. Convey the main idea so that this condensed version "
    "alone makes the text perfectly clear. No intros, quotes, or "
    "explanations. Respond in the language of the note."
)

# summary.py::MERGE_SYSTEM_PROMPT — слияние дубликатов (системный).
SUMMARY_MERGE_SYSTEM_PROMPT = (
    "You have two versions of the same note. Merge them into a single "
    "text: combine all information from both — facts, names, numbers, "
    "dates, statuses, configs and paths; state each fact once, omit "
    "repetitions. Do not drop anything and do not add anything of your "
    "own. Write coherently, no headings, intros, quotes, or explanations. "
    "Respond in the language of the notes."
)

# judge.py::JUDGE_SYSTEM_PROMPT — определение одинаковости заметок
# (ДУБЛЬ/НЕ ДУБЛЬ).
JUDGE_SYSTEM_PROMPT = (
    "You check long-term memory for duplicates. Determine whether two texts "
    "are duplicates: the same thought restated in other words (matching "
    "details matter more than wording). Answer with exactly one marker: "
    "DUPLICATE or NOT DUPLICATE. No explanations."
)

# Зашитые 7 (только константы, файлами не создаются).

# summary.py::MERGE_USER_TEMPLATE ({text_a}, {text_b}).
SUMMARY_MERGE_USER_TEMPLATE = "TEXT 1:\n{text_a}\n\nTEXT 2:\n{text_b}"

# judge.py::JUDGE_USER_TEMPLATE ({text_new}, {text_candidate}).
JUDGE_USER_TEMPLATE = "TEXT 1:\n{text_new}\n\nTEXT 2:\n{text_candidate}"

# classifier.py::CLASSIFY_SYSTEM_PROMPT — JSON-контракт разметки (§5.7).
CLASSIFIER_SYSTEM_PROMPT = (
    "You are a note classifier for hierarchical memory. Determine which "
    "section the note belongs to. Known nodes are listed in the request. "
    "Rules: if the note belongs to an existing node — return its full path "
    "as hint_path (1..3 segments, latin letters, digits, hyphens); if the "
    "note is generic and not bound to a section — return null. Answer with "
    "exactly one JSON object, no explanations: "
    '{"hint_path": "...", "confidence": 0.0} — confidence from 0 to 1, '
    "how confident you are."
)

# promotion.py::DESCRIBE_SYSTEM_PROMPT — описание узла по примерам заметок.
DESCRIBE_SYSTEM_PROMPT = (
    "You generate descriptions of sections in hierarchical memory. From the "
    "example notes, write a description of the section: what kind of notes "
    "live there. Strictly 1–2 short sentences, no lists or explanations — "
    "only the description."
)

# promotion.py::DESCRIBE_USER_TEMPLATE ({domain}, {slug}, {summaries}).
DESCRIBE_USER_TEMPLATE = (
    "New subsection: {domain}/{slug}\n\n"
    "Example notes of the section (brief summaries):\n{summaries}"
)

# promotion.py::JUDGE_SYSTEM_PROMPT — судья структуры
# (протокол СОЗДАТЬ/СЛИТЬ <path>/ОТКЛОНИТЬ).
STRUCTURE_JUDGE_SYSTEM_PROMPT = (
    "You are the structure judge of hierarchical memory. Evaluate the "
    "candidate for a new sub-section. Rules: (1) if the meaning of the "
    "candidate matches an existing topical node (the same theme in other "
    "words) — it is a merge, not a new node; (2) the slug and the "
    "description must be meaningful: a meaningless, junk or empty-in-meaning "
    "candidate — reject. Answer with exactly one marker, no explanations: "
    "CREATE — the candidate is new and meaningful; MERGE <path> — the "
    "candidate duplicates an existing node, for path specify ONLY the "
    "topical path from the \"Existing nodes\" list (never the candidate's "
    "path; default is the system swap, merging into it never happens); "
    "REJECT — the candidate is meaningless."
)

# promotion.py::JUDGE_USER_TEMPLATE ({domain}, {slug}, {description},
# {nodes}, {nearest}).
STRUCTURE_JUDGE_USER_TEMPLATE = (
    "Candidate: {domain}/{slug} — {description}\n\n"
    "Existing nodes:\n{nodes}\n\n"
    "Nearest by vector similarity: {nearest}"
)

# Имя → встроенный текст (единый словарь: и свойства реестра, и seed-файлы).
BUILTIN_PROMPTS: dict[str, str] = {
    "summary_system": SUMMARY_SYSTEM_PROMPT,
    "summary_merge_system": SUMMARY_MERGE_SYSTEM_PROMPT,
    "judge_system": JUDGE_SYSTEM_PROMPT,
    "merge_user": SUMMARY_MERGE_USER_TEMPLATE,
    "judge_user": JUDGE_USER_TEMPLATE,
    "classifier_system": CLASSIFIER_SYSTEM_PROMPT,
    "describe_system": DESCRIBE_SYSTEM_PROMPT,
    "describe_user": DESCRIBE_USER_TEMPLATE,
    "structure_judge_system": STRUCTURE_JUDGE_SYSTEM_PROMPT,
    "structure_judge_user": STRUCTURE_JUDGE_USER_TEMPLATE,
}

# Редактируемые: только эти имена выносятся в файлы (зашитые 7 — никогда).
EDITABLE_PROMPTS: tuple[str, ...] = (
    "summary_system",
    "summary_merge_system",
    "judge_system",
)

_PROMPT_FILE_SUFFIX = ".txt"


class PromptRegistry:
    """Встроенные константы всех 10 промптов + файловые переопределения.

    Конструирование = полная подготовка к работе: seed-if-missing (при
    заданном `prompts_dir`), чтение переопределений и контрактная проверка
    `judge_system`. Валидация падает фатальным `ConfigError` — старт сервиса
    обязан прерваться (см. docstring модуля).

    Доступ — по именам промптов (свойства, как в задании пула 3):
    редактируемые `summary_system`, `summary_merge_system`, `judge_system` и
    зашитые `merge_user`, `judge_user`, `classifier_system`, `describe_system`,
    `describe_user`, `structure_judge_system`, `structure_judge_user`;
    универсальный `get(name)` — для обхода по списку `BUILTIN_PROMPTS`.
    """

    def __init__(self, prompts_dir: str | os.PathLike[str] | None = None) -> None:
        # Копия встроенных: файловые переопределения ложатся поверх.
        self._texts: dict[str, str] = dict(BUILTIN_PROMPTS)
        # Имена, чей финальный текст пришёл из непустого файла (диагностика
        # источника в ошибках валидации).
        self._file_overrides: set[str] = set()
        self.prompts_dir: Path | None = (
            Path(prompts_dir) if prompts_dir is not None else None
        )
        if self.prompts_dir is not None:
            self._load_files()
        self._validate_judge_markers()

    # --- API (имена — контракт пула 3/4) -------------------------------------

    @property
    def summary_system(self) -> str:
        """Системный промпт пересказа заметки (редактируемый)."""
        return self._texts["summary_system"]

    @property
    def summary_merge_system(self) -> str:
        """Системный промпт слияния дубликатов (редактируемый)."""
        return self._texts["summary_merge_system"]

    @property
    def judge_system(self) -> str:
        """Системный промпт судьи дедупа ДУБЛЬ/НЕ ДУБЛЬ (редактируемый)."""
        return self._texts["judge_system"]

    @property
    def merge_user(self) -> str:
        """User-шаблон слияния: {text_a}, {text_b} (зашитый)."""
        return self._texts["merge_user"]

    @property
    def judge_user(self) -> str:
        """User-шаблон судьи дедупа: {text_new}, {text_candidate} (зашитый)."""
        return self._texts["judge_user"]

    @property
    def classifier_system(self) -> str:
        """Системный промпт классификатора, JSON-контракт (зашитый)."""
        return self._texts["classifier_system"]

    @property
    def describe_system(self) -> str:
        """Системный промпт описания узла (зашитый)."""
        return self._texts["describe_system"]

    @property
    def describe_user(self) -> str:
        """User-шаблон описания узла: {domain}/{slug}, {summaries} (зашитый)."""
        return self._texts["describe_user"]

    @property
    def structure_judge_system(self) -> str:
        """Системный промпт судьи структуры СОЗДАТЬ/СЛИТЬ/ОТКЛОНИТЬ (зашитый)."""
        return self._texts["structure_judge_system"]

    @property
    def structure_judge_user(self) -> str:
        """User-шаблон судьи структуры (зашитый)."""
        return self._texts["structure_judge_user"]

    def get(self, name: str) -> str:
        """Текст промпта по имени (см. BUILTIN_PROMPTS); KeyError — незнакомое."""
        try:
            return self._texts[name]
        except KeyError:
            raise KeyError(
                f"PromptRegistry: неизвестный промпт «{name}» — доступны: "
                + ", ".join(BUILTIN_PROMPTS)
            ) from None

    # --- файловая механика ---------------------------------------------------

    def _file_path(self, name: str) -> Path:
        assert self.prompts_dir is not None
        return self.prompts_dir / f"{name}{_PROMPT_FILE_SUFFIX}"

    def _load_files(self) -> None:
        """Seed-if-missing трёх редактируемых + чтение непустых файлов.

        Существующие файлы НЕ перезаписываются никогда: файл, созданный на
        прошлом старте или оператором вручную, остаётся как есть. Пустой
        (в т.ч. пробельный) файл → встроенный дефолт; непустой → побеждает.
        """
        directory = self.prompts_dir
        assert directory is not None
        directory.mkdir(parents=True, exist_ok=True)
        for name in EDITABLE_PROMPTS:
            path = self._file_path(name)
            if not path.exists():
                # Первый старт: создаём файл со встроенным дефолтом как
                # стартовым текстом (текст уже в self._texts — перечитывать
                # не нужно).
                path.write_text(self._texts[name], encoding="utf-8")
                continue
            content = path.read_text(encoding="utf-8").strip()
            if content:
                self._texts[name] = content
                self._file_overrides.add(name)

    # --- валидация -----------------------------------------------------------

    def _validate_judge_markers(self) -> None:
        """Финальный judge_system обязан содержать маркеры «DUPLICATE» и «NOT DUPLICATE».

        Вердикты дедупа парсятся по этим маркерам (judge._verdict: сначала
        «NOT DUPLICATE», затем «DUPLICATE»). Промпт без маркеров не дал бы
        парсеру ни одного вердикта — дедуп молча встал бы на отказах.
        Фатально.

        «DUPLICATE» требуется самостоятельной отметкой (вхождение вне связки
        «NOT DUPLICATE»): промпт, разрешающий только «NOT DUPLICATE», тихо
        сломал бы сведение дубликатов в другую сторону — судья не смог бы
        вынести вердикт «дубль» (тот же класс отказа, что и потеря маркера
        вовсе).
        """
        judge = self._texts["judge_system"]
        missing = self._missing_judge_markers(judge)
        if missing:
            if "judge_system" in self._file_overrides:
                origin = f"file {self._file_path('judge_system')}"
            else:
                origin = "builtin judge_system"
            raise ConfigError(
                "judge_system prompt is missing verdict marker(s): "
                + " and ".join(missing)
                + f" (source: {origin}). Dedup verdicts are parsed by these "
                "markers (judge._verdict); without them dedup would silently "
                "stall on parser failures. Restore both markers — DUPLICATE "
                "and NOT DUPLICATE — in the text."
            )

    @staticmethod
    def _missing_judge_markers(text: str) -> list[str]:
        """Маркеры вердикта, отсутствующие в тексте промпта.

        «DUPLICATE» засчитывается только самостоятельным вхождением:
        вхождения внутри «NOT DUPLICATE» вычёркиваются. «NOT DUPLICATE» —
        как связка целиком.
        """
        missing: list[str] = []
        if "NOT DUPLICATE" not in text:
            missing.append("NOT DUPLICATE")
        if "DUPLICATE" not in text.replace("NOT DUPLICATE", ""):
            missing.append("DUPLICATE")
        return missing
