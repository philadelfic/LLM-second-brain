"""Тесты SkillsService (lsb-0007-01): форма и лимиты, extra, композит чтения,
архив версий, soft delete, сид шаблона и pending-вектора области.

ARCH lsb-0007 §3.1–3.4, §3.8: валидация формы обязательна (нарушение —
`SkillValidationError` с дословным hint), правка пишет версию в общую историю
(`record_versions`) одной транзакцией, удаление мягкое, глобальный
`instruction_template` — сид `init_db` и композит `get`. Запись навыка
кодировщик не зовёт: вектора догоняет петля `areas` воркера (субстрат §3.3).
"""

from __future__ import annotations

import json

import pytest
from fakes import clear_seeded_skills

from app.config import Settings, get_settings
from app.services import history
from app.services.embedding import EmbeddingError
from app.services.skills import (
    EXTRA_FIELDS,
    HINT_DESCRIPTION_LIMIT,
    HINT_EXAMPLE_LIMIT,
    HINT_EXTRA_LIMIT,
    HINT_MODE,
    HINT_NAME_LIMIT,
    HINT_NOT_FOUND,
    HINT_STEPS_LIMIT,
    HINT_TEMPLATE_LIMIT,
    HINT_TEXT_LIMIT,
    SkillValidationError,
    SkillsService,
)
from app.storage.db import (
    INSTRUCTION_TEMPLATE_KEY,
    INSTRUCTION_TEMPLATE_SEED,
    init_db,
    session,
)


class RecordingEmbedder:
    """Фейк-эмбеддер с журналом вызовов: запись навыка кодировщик НЕ зовёт.

    Любой вызов — ошибка теста: синхронная векторизация при save/правке
    запрещена (вектора догоняет петля `areas` воркера, субстрат §3.3).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingError("эмбеддер не должен вызываться при записи навыка")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:  # интерфейс-совместимость с EmbeddingService
        return None


def form(**overrides: object) -> dict[str, object]:
    """Валидная форма навыка; overrides правят отдельные поля."""
    payload: dict[str, object] = {
        "name": "Deploy the service",
        "description": "How to deploy this service",
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def settings() -> Settings:
    """Настройки тестового окружения (лимиты формы — из них)."""
    return get_settings()


@pytest.fixture
def embedder() -> RecordingEmbedder:
    """Фейк-кодировщик с журналом вызовов (ожидается пустым)."""
    return RecordingEmbedder()


@pytest.fixture
def service(settings: Settings, embedder: RecordingEmbedder) -> SkillsService:
    """SkillsService над инициализированной БД (без сети, DI-эмбеддер).

    Сид skill-создателя (lsb-0007-04) снимаем: пул 01 проверяет форму,
    версии и удаление на пустом реестре.
    """
    init_db(settings)
    clear_seeded_skills(settings)
    return SkillsService(settings, embedder)


def _row(table: str, skill_id: int) -> dict | None:
    """Прямая вычитка строки (проверки «строка жива», статусов и extra)."""
    with session(get_settings()) as conn:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (skill_id,)
        ).fetchone()
    return dict(row) if row is not None else None


class TestFormLimits:
    """Лимиты формы (канон §3.8): граница — ок, граница+1 — отказ с hint."""

    @pytest.mark.parametrize(
        ("field", "setting", "hint"),
        [
            ("name", "skill_name_max_chars", HINT_NAME_LIMIT),
            ("description", "skill_description_max_chars", HINT_DESCRIPTION_LIMIT),
            ("steps", "skill_steps_max_chars", HINT_STEPS_LIMIT),
            ("text", "skill_text_max_chars", HINT_TEXT_LIMIT),
            ("example", "skill_example_max_chars", HINT_EXAMPLE_LIMIT),
        ],
    )
    def test_boundary_ok_and_limit_plus_one_rejected(
        self,
        service: SkillsService,
        settings: Settings,
        field: str,
        setting: str,
        hint: str,
    ) -> None:
        limit = getattr(settings, setting)
        created = service.save(**form(**{field: "x" * limit}))  # ровно лимит — ок
        assert created["created"] is True
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(**{field: "x" * (limit + 1)}))
        assert str(exc.value) == hint  # дословный hint канона §3.8

    @pytest.mark.parametrize("field", ["name", "description", "steps", "text"])
    def test_mandatory_field_empty_rejected(
        self, service: SkillsService, field: str
    ) -> None:
        with pytest.raises(SkillValidationError):
            service.save(**form(**{field: "   "}))

    def test_example_empty_is_absent(self, service: SkillsService) -> None:
        """Пустой example — как отсутствующий (секции в композите нет)."""
        assert service.save(**form(example="   "))["created"] is True
        assert "example" not in service.get(1)

    def test_rejected_form_creates_nothing(self, service: SkillsService) -> None:
        with pytest.raises(SkillValidationError):
            service.save(**form(name="x" * 66))
        listed = service.list()
        assert listed["items"] == []
        assert listed["total"] == 0


class TestRaisedLimits:
    """Новые лимиты формы навыка (решение О. 12.10.2026; канон §6/§3.8).

    Лимиты подняты: `description` 250→400, `steps` 500→1000, `text` 4000→7500,
    `example` 1000→2000, поле `extra` 500→1000 (сумма 2000→4000),
    `instruction_template` 1000→2000; `name` (65) не трогали.

    Числа заданы ЛИТЕРАЛАМИ, а не через настройки: тест ловит и «сервис читает
    настройку», и «настройка равна согласованному числу». Прежняя граница
    (250/500/4000/1000) теперь проходит молча — это и проверяется.
    """

    def test_description_300_saved_and_401_rejected(
        self, service: SkillsService
    ) -> None:
        """Согласовано 400: 300 символов сохраняются, 401 — отказ с цифрой 400."""
        assert service.save(**form(description="d" * 300))["created"] is True
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(description="d" * 401))
        assert str(exc.value) == HINT_DESCRIPTION_LIMIT
        assert "400 characters" in str(exc.value)

    def test_steps_700_saved_and_1001_rejected(self, service: SkillsService) -> None:
        """Согласовано 1000: 700 шагов сохраняются, 1001 — отказ с цифрой 1000."""
        assert service.save(**form(steps="s" * 700))["created"] is True
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(steps="s" * 1001))
        assert str(exc.value) == HINT_STEPS_LIMIT
        assert "1000 characters" in str(exc.value)

    def test_text_7000_saved_and_7501_rejected(self, service: SkillsService) -> None:
        """Согласовано 7500: тело 7000 символов сохраняется, 7501 — отказ.

        Прежний лимит 4000: до решения такое тело было отказом (пример
        требования — «тексты скиллов в 7000 символов сохраняются»).
        """
        assert service.save(**form(text="t" * 7000))["created"] is True
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(text="t" * 7501))
        assert str(exc.value) == HINT_TEXT_LIMIT
        assert "7500 characters" in str(exc.value)

    def test_example_1500_saved_and_2001_rejected(
        self, service: SkillsService
    ) -> None:
        """Согласовано 2000: пример 1500 символов сохраняется, 2001 — отказ."""
        assert service.save(**form(example="e" * 1500))["created"] is True
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(example="e" * 2001))
        assert str(exc.value) == HINT_EXAMPLE_LIMIT
        assert "2000 characters" in str(exc.value)

    def test_extra_field_1000_saved_and_1001_rejected(
        self, service: SkillsService
    ) -> None:
        """Согласовано 1000 на поле класса: 1000 — ок, 1001 — отказ (§3.1)."""
        assert service.save(**form(extra={"trigger": "x" * 1000}))[
            "created"
        ] is True
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(extra={"trigger": "x" * 1001}))
        assert str(exc.value) == HINT_EXTRA_LIMIT
        assert "1000 characters" in str(exc.value) and "4000" in str(exc.value)

    def test_extra_total_4000_saved_and_4001_rejected(
        self, service: SkillsService
    ) -> None:
        """Согласовано 4000 на сумму полей: 4×1000 — ок, 4001 — отказ (§3.1)."""
        extra = {
            "trigger": "x" * 1000,
            "preconditions": "x" * 1000,
            "fallbacks": "x" * 1000,
            "invariant": "x" * 1000,
        }
        assert sum(len(value) for value in extra.values()) == 4000
        assert service.save(**form(extra=extra))["created"] is True
        extra["exceptions"] = "x"  # сумма 4001 > 4000
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(extra=extra))
        assert str(exc.value) == HINT_EXTRA_LIMIT

    def test_instruction_template_1500_saved_and_2001_rejected(
        self, service: SkillsService
    ) -> None:
        """Согласовано 2000: шаблон 1500 символов сохраняется, 2001 — отказ."""
        assert service.set_instruction_template("x" * 1500) == {
            "instruction_template": "x" * 1500,
            "updated": True,
        }
        with pytest.raises(SkillValidationError) as exc:
            service.set_instruction_template("x" * 2001)
        assert str(exc.value) == HINT_TEMPLATE_LIMIT
        assert "2000 characters" in str(exc.value)


class TestExtra:
    """`extra` — один JSON-объект только из известных полей класса (§3.1)."""

    def test_valid_extra_saved_and_returned(self, service: SkillsService) -> None:
        extra = {
            "mode": "collaborative",
            "trigger": "on release",
            "references": "runbook/deploy",
        }
        assert service.save(**form(extra=extra))["created"] is True
        assert service.get(1)["extra"] == extra  # возвращен композитом
        row = _row("skills", 1)
        assert row is not None
        assert json.loads(row["extra"]) == extra  # и лежит одним JSON-объектом

    def test_autonomous_mode_is_valid(self, service: SkillsService) -> None:
        assert service.save(**form(extra={"mode": "autonomous"}))["created"] is True

    def test_unknown_key_rejected(self, service: SkillsService) -> None:
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(extra={"priority": "high"}))
        assert "unknown optional field" in str(exc.value)
        assert service.list()["total"] == 0

    def test_eleventh_field_rejected(self, service: SkillsService) -> None:
        """11-е поле — за пределами класса навыка (известных ключей 10)."""
        extra = {field: "x" for field in EXTRA_FIELDS}
        assert len(extra) == 10
        extra["one_more"] = "x"
        with pytest.raises(SkillValidationError):
            service.save(**form(extra=extra))

    def test_all_ten_fields_within_budget_ok(self, service: SkillsService) -> None:
        """Сумма ровно 4000 — граница бюджета extra (§3.1), сохранение ок."""
        extra = {
            "trigger": "x" * 1000,
            "preconditions": "x" * 1000,
            "fallbacks": "x" * 1000,
            "invariant": "x" * 1000,
        }
        assert sum(len(value) for value in extra.values()) == 4000
        assert service.save(**form(extra=extra))["created"] is True

    def test_field_over_limit_rejected(self, service: SkillsService) -> None:
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(extra={"trigger": "x" * 1001}))
        assert str(exc.value) == HINT_EXTRA_LIMIT

    def test_total_over_limit_rejected(self, service: SkillsService) -> None:
        extra = {
            "trigger": "x" * 1000,
            "preconditions": "x" * 1000,
            "fallbacks": "x" * 1000,
            "invariant": "x" * 1000,
            "exceptions": "x" * 1,  # сумма 4001 > 4000
        }
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(extra=extra))
        assert str(exc.value) == HINT_EXTRA_LIMIT

    def test_mode_outside_pair_rejected(self, service: SkillsService) -> None:
        with pytest.raises(SkillValidationError) as exc:
            service.save(**form(extra={"mode": "manual"}))
        assert str(exc.value) == HINT_MODE


class TestGetComposite:
    """Композит §3.1: секции формы + глобальный `instruction_template`."""

    def test_composite_contains_template_and_all_sections(
        self, service: SkillsService
    ) -> None:
        service.save(**form(example="Example: make deploy --prod"))
        composite = service.get(1)
        assert composite == {
            "id": 1,
            "name": "Deploy the service",
            "description": "How to deploy this service",
            "example": "Example: make deploy --prod",
            "steps": "1) build; 2) ship; 3) verify",
            "text": "Run make deploy, then check /health.",
            "instruction_template": INSTRUCTION_TEMPLATE_SEED,
        }

    def test_example_absent_when_not_set(self, service: SkillsService) -> None:
        service.save(**form())
        composite = service.get(1)
        assert "example" not in composite
        assert composite["instruction_template"] == INSTRUCTION_TEMPLATE_SEED

    def test_unknown_id_returns_hint(self, service: SkillsService) -> None:
        assert service.get(404) == {"id": 404, "hint": HINT_NOT_FOUND}


class TestList:
    """Листинг компактен: id/name/description, только активные (§3.3)."""

    def test_items_without_bodies(self, service: SkillsService) -> None:
        service.save(**form())
        service.save(**form(name="Rollback the service"))
        listed = service.list()
        assert listed["items"] == [
            {
                "id": 2,
                "name": "Rollback the service",
                "description": "How to deploy this service",
            },
            {
                "id": 1,
                "name": "Deploy the service",
                "description": "How to deploy this service",
            },
        ]
        assert listed["total"] == 2

    def test_pagination_contract(self, service: SkillsService) -> None:
        service.save(**form())
        assert service.list(limit=1)["total"] == 1
        with pytest.raises(SkillValidationError):
            service.list(limit=0)
        with pytest.raises(SkillValidationError):
            service.list(offset=-1)


class TestVersions:
    """Правка пишет версию в общую историю; legacy-архив не наполняется."""

    def test_update_writes_history_delta(self, service: SkillsService) -> None:
        service.save(**form())
        result = service.save(**form(id=1, name="Deploy v2"))
        assert result == {"id": 1, "updated": True, "version": 2}
        with session(get_settings()) as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT version, event FROM record_versions "
                    "WHERE entity = 'skill' AND entity_id = 1 ORDER BY version"
                )
            ]
        assert [row["version"] for row in rows] == [1, 2]
        assert rows[0]["event"] == "baseline" and rows[1]["event"] == "content"
        assert _row("skills", 1)["version"] == 2  # зеркало головной версии (§13.7)
        assert _row("skills", 1)["name"] == "Deploy v2"

    def test_version_grows_on_each_update(self, service: SkillsService) -> None:
        service.save(**form())
        service.save(**form(id=1, name="Deploy v2"))
        service.save(**form(id=1, name="Deploy v3"))
        with session(get_settings()) as conn:
            versions = [
                int(row["version"])
                for row in conn.execute(
                    "SELECT version FROM record_versions WHERE entity = 'skill' "
                    "AND entity_id = 1 ORDER BY version"
                )
            ]
        assert versions == [1, 2, 3]
        assert _row("skills", 1)["version"] == 3

    def test_archive_invisible_in_get_and_list(self, service: SkillsService) -> None:
        service.save(**form())
        service.save(**form(id=1, name="Deploy v2"))
        assert service.get(1)["name"] == "Deploy v2"  # выдача — только активная
        listed = service.list()
        assert listed["total"] == 1
        assert [item["name"] for item in listed["items"]] == ["Deploy v2"]

    def test_history_survives_soft_delete(self, service: SkillsService) -> None:
        service.save(**form())
        service.save(**form(id=1, name="Deploy v2"))
        service.delete(1)
        with session(get_settings()) as conn:
            events = [
                str(row["event"])
                for row in conn.execute(
                    "SELECT event FROM record_versions WHERE entity = 'skill' "
                    "AND entity_id = 1 ORDER BY version"
                )
            ]
        # История — операторская: мягкое удаление её не стирает (§13.7, FR-6).
        assert events == ["baseline", "content", "delete"]

    def test_update_unknown_id_returns_hint(self, service: SkillsService) -> None:
        assert service.save(**form(id=999)) == {
            "id": 999,
            "updated": False,
            "hint": HINT_NOT_FOUND,
        }

    def test_history_write_and_edit_are_one_transaction(
        self, service: SkillsService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Сбой на записи версии откатывает и правку (одна транзакция, §13.5)."""
        service.save(**form())

        def boom(*args: object, **kwargs: object) -> int:
            raise RuntimeError("history write failed")

        monkeypatch.setattr(history, "record_change", boom)
        with pytest.raises(RuntimeError):
            service.save(**form(id=1, name="Deploy v2"))
        row = _row("skills", 1)
        assert row is not None
        assert row["version"] == 1  # правка откатилась
        assert row["name"] == "Deploy the service"


class TestDelete:
    """Soft delete (§3.4): навык исчезает из выдач, строка/индексы живы."""

    def test_delete_hides_from_list_and_get(self, service: SkillsService) -> None:
        service.save(**form())
        assert service.delete(1) == {"id": 1, "deleted": True}
        listed = service.list()
        assert listed["items"] == []
        assert listed["total"] == 0
        assert service.get(1) == {"id": 1, "hint": HINT_NOT_FOUND}
        row = _row("skills", 1)
        assert row is not None  # строка жива (trash)
        assert row["deleted_at"] is not None

    def test_repeated_delete_returns_hint(self, service: SkillsService) -> None:
        service.save(**form())
        service.delete(1)
        assert service.delete(1) == {
            "id": 1,
            "deleted": False,
            "hint": HINT_NOT_FOUND,
        }

    def test_delete_unknown_id_returns_hint(self, service: SkillsService) -> None:
        assert service.delete(999)["deleted"] is False
        assert service.delete(999)["hint"] == HINT_NOT_FOUND

    def test_update_deleted_skill_returns_hint(self, service: SkillsService) -> None:
        service.save(**form())
        service.delete(1)
        assert service.save(**form(id=1, name="Deploy v2")) == {
            "id": 1,
            "updated": False,
            "hint": HINT_NOT_FOUND,
        }


class TestInstructionTemplate:
    """Глобальный шаблон: сид идемпотентен, правка оператора не затирается."""

    def test_seed_present_and_idempotent(self, settings: Settings) -> None:
        init_db(settings)
        init_db(settings)  # повторный старт — no-op
        with session(settings) as conn:
            rows = conn.execute(
                "SELECT key, value FROM skills_meta WHERE key = ?",
                (INSTRUCTION_TEMPLATE_KEY,),
            ).fetchall()
        assert len(rows) == 1  # сид не дублируется
        assert rows[0]["value"] == INSTRUCTION_TEMPLATE_SEED

    def test_operator_edit_survives_reinit(self, service: SkillsService) -> None:
        service.set_instruction_template("Do the steps in order, no skipping.")
        init_db(get_settings())
        with session(get_settings()) as conn:
            value = conn.execute(
                "SELECT value FROM skills_meta WHERE key = ?",
                (INSTRUCTION_TEMPLATE_KEY,),
            ).fetchone()["value"]
        assert value == "Do the steps in order, no skipping."
        assert service.instruction_template() == {
            "instruction_template": "Do the steps in order, no skipping."
        }

    def test_edit_visible_in_composite(self, service: SkillsService) -> None:
        service.save(**form())
        service.set_instruction_template("Execute strictly in order.")
        assert service.get(1)["instruction_template"] == "Execute strictly in order."

    @pytest.mark.parametrize("size", [1, 2000])
    def test_boundary_sizes_ok(self, service: SkillsService, size: int) -> None:
        result = service.set_instruction_template("x" * size)
        assert result == {"instruction_template": "x" * size, "updated": True}

    def test_over_limit_rejected(self, service: SkillsService) -> None:
        with pytest.raises(SkillValidationError) as exc:
            service.set_instruction_template("x" * 2001)
        assert str(exc.value) == HINT_TEMPLATE_LIMIT
        # Отказ ничего не изменил: в БД — сид.
        assert service.instruction_template() == {
            "instruction_template": INSTRUCTION_TEMPLATE_SEED
        }

    def test_empty_rejected(self, service: SkillsService) -> None:
        with pytest.raises(SkillValidationError):
            service.set_instruction_template("   ")


class TestVectorPending:
    """Запись мгновенная: pending + ноль вызовов кодировщика (субстрат §3.3)."""

    def test_create_and_update_stay_pending(
        self, service: SkillsService, embedder: RecordingEmbedder
    ) -> None:
        service.save(**form())
        assert embedder.calls == []  # синхронной векторизации нет
        assert _row("skills", 1)["vector_status"] == "pending"
        service.save(**form(id=1, name="Deploy v2"))
        assert embedder.calls == []
        assert _row("skills", 1)["vector_status"] == "pending"
        with session(get_settings()) as conn:
            vectors_count = conn.execute(
                "SELECT COUNT(*) FROM skills_vec"
            ).fetchone()[0]
        assert vectors_count == 0  # vec0 заполнит петля areas воркера

    def test_notifier_called_on_success_only(
        self, service: SkillsService
    ) -> None:
        """Сигнал петле areas: на создании/правке — да, на мягком отказе — нет."""
        calls: list[int] = []
        service.set_areas_notifier(lambda: calls.append(1))
        service.save(**form())
        service.save(**form(id=1, name="Deploy v2"))
        service.save(**form(id=999))  # навыка нет — сигнала нет
        service.delete(999)
        assert calls == [1, 1]
