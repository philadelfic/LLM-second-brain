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
    "Сделай краткий пересказ заметки в 1–2 коротких и ёмких "
    "предложениях, суммарно не длиннее 30 слов. Передай главную мысль "
    "так, чтобы по этому сокращению было предельно понятно, о чём текст. "
    "Без вступлений, кавычек и пояснений. Отвечай на языке заметки."
)

# summary.py::MERGE_SYSTEM_PROMPT — слияние дубликатов (системный).
SUMMARY_MERGE_SYSTEM_PROMPT = (
    "У тебя две версии одной заметки. Сведи их в единый текст: объедини "
    "всю информацию обеих — факты, имена, числа, даты, статусы, конфиги "
    "и пути; каждый факт скажи один раз, повторяющееся опусти. Ничего "
    "не выбрасывай и не добавляй от себя. Пиши связно, без заголовков, "
    "вступлений, кавычек и пояснений. Отвечай на языке заметок."
)

# judge.py::JUDGE_SYSTEM_PROMPT — определение одинаковости заметок
# (ДУБЛЬ/НЕ ДУБЛЬ).
JUDGE_SYSTEM_PROMPT = (
    "Ты проверяешь долговременную память на дубли. Определи, являются ли "
    "два текста дублями: одна и та же мысль, пересказанная другими словами "
    "(совпадение деталей важнее формы). Ответь строго одной отметкой: "
    "ДУБЛЬ или НЕ ДУБЛЬ. Без пояснений."
)

# Зашитые 7 (только константы, файлами не создаются).

# summary.py::MERGE_USER_TEMPLATE ({text_a}, {text_b}).
SUMMARY_MERGE_USER_TEMPLATE = "ТЕКСТ 1:\n{text_a}\n\nТЕКСТ 2:\n{text_b}"

# judge.py::JUDGE_USER_TEMPLATE ({text_new}, {text_candidate}).
JUDGE_USER_TEMPLATE = "ТЕКСТ 1:\n{text_new}\n\nТЕКСТ 2:\n{text_candidate}"

# classifier.py::CLASSIFY_SYSTEM_PROMPT — JSON-контракт разметки (§5.7).
CLASSIFIER_SYSTEM_PROMPT = (
    "Ты классификатор заметок для иерархической памяти. Определи, к какому "
    "разделу относится заметка. Известные узлы перечислены в запросе. "
    "Правила: если заметка относится к существующему домену — верни его путь "
    "как domain_hint; если к конкретному подразделу — верни слаг листа как "
    "subdomain_hint (латиница-цифры-дефис), иначе null; если заметка общая и "
    "не привязана к домену — верни null для обоих. Ответь строго одним "
    'JSON-объектом без пояснений: {"domain_hint": "...", '
    '"subdomain_hint": "...", "confidence": 0.0} — confidence от 0 до 1, '
    "насколько уверен в выборе."
)

# promotion.py::DESCRIBE_SYSTEM_PROMPT — описание узла по примерам заметок.
DESCRIBE_SYSTEM_PROMPT = (
    "Ты генератор описаний разделов иерархической памяти. По примерам "
    "заметок напиши описание раздела: какие заметки в нём живут. Строго "
    "1–2 коротких предложения, без списков и пояснений — только описание."
)

# promotion.py::DESCRIBE_USER_TEMPLATE ({domain}, {slug}, {summaries}).
DESCRIBE_USER_TEMPLATE = (
    "Новый подраздел: {domain}/{slug}\n\n"
    "Примеры заметок раздела (краткие содержания):\n{summaries}"
)

# promotion.py::JUDGE_SYSTEM_PROMPT — судья структуры
# (протокол СОЗДАТЬ/СЛИТЬ <path>/ОТКЛОНИТЬ).
STRUCTURE_JUDGE_SYSTEM_PROMPT = (
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

# promotion.py::JUDGE_USER_TEMPLATE ({domain}, {slug}, {description},
# {nodes}, {nearest}).
STRUCTURE_JUDGE_USER_TEMPLATE = (
    "Кандидат: {domain}/{slug} — {description}\n\n"
    "Существующие узлы:\n{nodes}\n\n"
    "Ближайший по векторному сходству: {nearest}"
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
        """Финальный judge_system обязан содержать маркеры «ДУБЛЬ» и «НЕ ДУБЛЬ».

        Вердикты дедупа парсятся по этим маркерам (judge._verdict: сначала
        «НЕ ДУБЛЬ», затем «ДУБЛЬ»). Промпт без маркеров не дал бы парсеру
        ни одного вердикта — дедуп молча встал бы на отказах. Фатально.

        «ДУБЛЬ» требуется самостоятельной отметкой (вхождение вне связки
        «НЕ ДУБЛЬ»): промпт, разрешающий только «НЕ ДУБЛЬ», тихо сломал бы
        сведение дубликатов в другую сторону — судья не смог бы вынести
        вердикт «дубль» (тот же класс отказа, что и потеря маркера вовсе).
        """
        judge = self._texts["judge_system"]
        missing = self._missing_judge_markers(judge)
        if missing:
            if "judge_system" in self._file_overrides:
                origin = f"файл {self._file_path('judge_system')}"
            else:
                origin = "встроенный judge_system"
            raise ConfigError(
                "промпт judge_system не содержит маркер вердикта "
                + " и ".join(missing)
                + f" (источник: {origin}). Вердикты дедупа парсятся именно "
                "по этим маркерам (judge._verdict); без них дедуп молча "
                "встанет на отказах парсера. Верни в текст обе отметки — "
                "«ДУБЛЬ» и «НЕ ДУБЛЬ»."
            )

    @staticmethod
    def _missing_judge_markers(text: str) -> list[str]:
        """Маркеры вердикта, отсутствующие в тексте промпта.

        «ДУБЛЬ» засчитывается только самостоятельным вхождением: вхождения
        внутри «НЕ ДУБЛЬ» вычёркиваются. «НЕ ДУБЛЬ» — как связка целиком.
        """
        missing: list[str] = []
        if "НЕ ДУБЛЬ" not in text:
            missing.append("«НЕ ДУБЛЬ»")
        if "ДУБЛЬ" not in text.replace("НЕ ДУБЛЬ", ""):
            missing.append("«ДУБЛЬ»")
        return missing
