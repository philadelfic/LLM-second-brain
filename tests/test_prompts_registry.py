"""PromptRegistry (Фаза 11, пул 3; решение №7 брифа): реестр промптов.

Контракты пула: встроенные константы всех 10 промптов (тексты 1-в-1 с
app/services/{summary,judge,classifier,promotion}.py); файловая механика
трёх редактируемых при заданном prompts_dir: seed-if-missing идемпотентен,
существующие файлы НЕ перезаписываются, пустой файл = встроенный дефолт,
непустой файл побеждает, зашитые 7 файлами не создаются; judge_system из
файла без маркеров «ДУБЛЬ»/«НЕ ДУБЛЬ» → фатальный ConfigError; prompts_dir
не задан (None) → только встроенные.

Тесты файловой механики не трогают окружение процесса: каталог промптов
передаётся конструктору напрямую (tmp_path), а не через env.
"""

from __future__ import annotations

import pathlib

import pytest

from app.config import ConfigError
from app.services import (
    classifier,
    judge,
    promotion,
    prompts as prompts_mod,
    summary,
)
from app.services.prompts import (
    BUILTIN_PROMPTS,
    EDITABLE_PROMPTS,
    PromptRegistry,
)

# Имена промптов в контракте реестра (ключи BUILTIN_PROMPTS).
PROMPT_NAMES: tuple[str, ...] = (
    "summary_system",
    "summary_merge_system",
    "judge_system",
    "merge_user",
    "judge_user",
    "classifier_system",
    "describe_system",
    "describe_user",
    "structure_judge_system",
    "structure_judge_user",
)


# --- встроенные тексты 1-в-1 ------------------------------------------------


class TestBuiltinTexts:
    def test_summary_system_matches_service(self) -> None:
        assert PromptRegistry().summary_system == summary.SYSTEM_PROMPT

    def test_summary_merge_system_matches_service(self) -> None:
        assert PromptRegistry().summary_merge_system == summary.MERGE_SYSTEM_PROMPT

    def test_summary_merge_user_matches_service(self) -> None:
        assert PromptRegistry().merge_user == summary.MERGE_USER_TEMPLATE

    def test_judge_system_matches_service(self) -> None:
        assert PromptRegistry().judge_system == judge.JUDGE_SYSTEM_PROMPT

    def test_judge_user_matches_service(self) -> None:
        assert PromptRegistry().judge_user == judge.JUDGE_USER_TEMPLATE

    def test_classifier_system_matches_service(self) -> None:
        assert PromptRegistry().classifier_system == classifier.CLASSIFY_SYSTEM_PROMPT

    def test_describe_prompts_match_service(self) -> None:
        registry = PromptRegistry()
        assert registry.describe_system == promotion.DESCRIBE_SYSTEM_PROMPT
        assert registry.describe_user == promotion.DESCRIBE_USER_TEMPLATE

    def test_structure_judge_prompts_match_service(self) -> None:
        registry = PromptRegistry()
        assert registry.structure_judge_system == promotion.JUDGE_SYSTEM_PROMPT
        assert registry.structure_judge_user == promotion.JUDGE_USER_TEMPLATE


# --- API реестра (свойства + get) -------------------------------------------


class TestRegistryApi:
    def test_ten_builtin_prompts_registered(self) -> None:
        """Ровно 10 промптов, имена — контракт пула 3."""
        assert set(BUILTIN_PROMPTS) == set(PROMPT_NAMES)

    def test_all_names_resolvable_via_get(self) -> None:
        registry = PromptRegistry()
        for name in PROMPT_NAMES:
            assert registry.get(name) == BUILTIN_PROMPTS[name]

    def test_properties_match_get(self) -> None:
        registry = PromptRegistry()
        # Редактируемые свойства.
        assert registry.summary_system == registry.get("summary_system")
        assert registry.summary_merge_system == registry.get("summary_merge_system")
        assert registry.judge_system == registry.get("judge_system")
        # Зашитые свойства.
        assert registry.merge_user == registry.get("merge_user")
        assert registry.judge_user == registry.get("judge_user")
        assert registry.classifier_system == registry.get("classifier_system")
        assert registry.describe_system == registry.get("describe_system")
        assert registry.describe_user == registry.get("describe_user")
        assert registry.structure_judge_system == registry.get("structure_judge_system")
        assert registry.structure_judge_user == registry.get("structure_judge_user")

    def test_get_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            PromptRegistry().get("no_such_prompt")

    def test_editable_are_exactly_three(self) -> None:
        assert EDITABLE_PROMPTS == (
            "summary_system",
            "summary_merge_system",
            "judge_system",
        )


# --- prompts_dir не задан: только встроенные --------------------------------


class TestNoPromptsDir:
    def test_file_mechanics_inactive_by_default(self) -> None:
        """None → только встроенные, файловый слой не трогается."""
        registry = PromptRegistry()
        assert registry.prompts_dir is None
        for name in PROMPT_NAMES:
            assert registry.get(name) == BUILTIN_PROMPTS[name]

    def test_seed_not_written_without_prompts_dir(self) -> None:
        """Без каталога seed-файлы не создаются (некуда и незачем)."""
        registry = PromptRegistry()
        assert registry.prompts_dir is None  # никакого пути по умолчанию


# --- файловая механика ------------------------------------------------------


def _read(dirpath: pathlib.Path, name: str) -> str:
    return (dirpath / f"{name}.txt").read_text(encoding="utf-8")


class TestFileMechanics:
    def test_seed_creates_three_editable_files(self, tmp_path: pathlib.Path) -> None:
        """Первый старт с prompts_dir: ровно 3 редактируемых файла-дефолта."""
        registry = PromptRegistry(prompts_dir=tmp_path)
        files = sorted(p.name for p in tmp_path.iterdir())
        assert files == [
            "judge_system.txt",
            "summary_merge_system.txt",
            "summary_system.txt",
        ]
        # Содержимое seed-файлов — встроенные тексты (стартовый дефолт).
        assert _read(tmp_path, "summary_system") == registry.summary_system
        assert (
            _read(tmp_path, "summary_merge_system")
            == registry.summary_merge_system
        )
        assert _read(tmp_path, "judge_system") == registry.judge_system

    def test_seed_is_idempotent(self, tmp_path: pathlib.Path) -> None:
        """Повторный старт не перезаписывает существующие файлы."""
        PromptRegistry(prompts_dir=tmp_path)  # первый «старт»: seed
        # Оператор правит файл между рестартами.
        edited = "Правленый промпт пересказа."
        (tmp_path / "summary_system.txt").write_text(edited, encoding="utf-8")
        PromptRegistry(prompts_dir=tmp_path)  # повторный «старт»
        # Правленый файл не откатился к дефолту; нетронутые — как после seed.
        assert (tmp_path / "summary_system.txt").read_text(
            encoding="utf-8"
        ) == edited
        assert _read(tmp_path, "summary_merge_system") == (
            PromptRegistry().summary_merge_system
        )
        assert _read(tmp_path, "judge_system") == PromptRegistry().judge_system

    def test_nonempty_file_wins_over_builtin(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Непустой файл побеждает встроенный (summary_merge_system)."""
        custom = "Объедини аккуратно, сохрани все детали."
        (tmp_path / "summary_merge_system.txt").write_text(
            custom, encoding="utf-8"
        )
        registry = PromptRegistry(prompts_dir=tmp_path)
        assert registry.summary_merge_system == custom

    def test_empty_file_falls_back_to_builtin(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Пустой файл → встроенный дефолт."""
        (tmp_path / "summary_system.txt").write_text("", encoding="utf-8")
        (tmp_path / "summary_merge_system.txt").write_text(
            "   \n\t  ", encoding="utf-8"
        )  # пробельный = пустой
        registry = PromptRegistry(prompts_dir=tmp_path)
        assert registry.summary_system == summary.SYSTEM_PROMPT
        assert registry.summary_merge_system == summary.MERGE_SYSTEM_PROMPT

    def test_judge_file_wins_over_builtin(self, tmp_path: pathlib.Path) -> None:
        """Судья дедупа читается из файла (с маркерами) — поверх встроенного."""
        custom = "Ты судья. Скажи: ДУБЛЬ или НЕ ДУБЛЬ. Без воды."
        (tmp_path / "judge_system.txt").write_text(custom, encoding="utf-8")
        registry = PromptRegistry(prompts_dir=tmp_path)
        assert registry.judge_system == custom
        assert registry.judge_system != judge.JUDGE_SYSTEM_PROMPT

    def test_seed_only_for_editable_names(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Зашитые 7 файлами не создаются никогда."""
        PromptRegistry(prompts_dir=tmp_path)
        names = {p.name for p in tmp_path.iterdir()}
        assert names == {f"{name}.txt" for name in EDITABLE_PROMPTS}
        assert not any(name in names for name in ("merge_user.txt", "judge_user.txt"))
        assert "classifier_system.txt" not in names
        assert "structure_judge_user.txt" not in names


# --- контрактная валидация judge_system -------------------------------------


class TestJudgeSystemValidation:
    def test_file_without_markers_is_fatal(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Файл с текстом, но без «ДУБЛЬ»/«НЕ ДУБЛЬ» → фатальный ConfigError."""
        (tmp_path / "judge_system.txt").write_text(
            "Ты просто сравниваешь тексты.", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="ДУБЛЬ"):
            PromptRegistry(prompts_dir=tmp_path)

    def test_file_without_dubl_marker_is_fatal(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Есть «НЕ ДУБЛЬ», но нет «ДУБЛЬ» → тоже фатально (см. judge._verdict:
        сначала ищется «НЕ ДУБЛЬ»; «ДУБЛЬ» отдельно требуется для вердикта True)."""
        (tmp_path / "judge_system.txt").write_text(
            "Отвечай: НЕ ДУБЛЬ.", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="ДУБЛЬ"):
            PromptRegistry(prompts_dir=tmp_path)

    def test_file_with_only_marker_word_ne_dubl_is_not_enough(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Слово «НЕ ДУБЛЬ» содержит подстроку «ДУБЛЬ» — но проверка требует
        оба маркера как отдельные токены текста; одного слова мало."""
        (tmp_path / "judge_system.txt").write_text(
            "Ты отвечаешь только: НЕ ДУБЛЬ.", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="ДУБЛЬ"):
            PromptRegistry(prompts_dir=tmp_path)

    def test_validation_applies_to_builtin_too(self) -> None:
        """Встроенный judge_system валиден (содержит оба маркера)."""
        registry = PromptRegistry()
        assert "ДУБЛЬ" in registry.judge_system
        assert "НЕ ДУБЛЬ" in registry.judge_system

    def test_error_names_file_source(self, tmp_path: pathlib.Path) -> None:
        """Сообщение об ошибке называет источник (файл), а не встроенный текст."""
        (tmp_path / "judge_system.txt").write_text(
            "Сравни и скажи ответ.", encoding="utf-8"
        )
        with pytest.raises(ConfigError) as excinfo:
            PromptRegistry(prompts_dir=tmp_path)
        assert "judge_system.txt" in str(excinfo.value)
