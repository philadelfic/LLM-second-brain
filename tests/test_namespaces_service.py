"""Тесты NamespaceService (Фаза 10, Шаг 1): валидация контрактов §5.7,
CRUD реестра, счётчики узлов/поддеревьев."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.namespaces import (
    NamespaceError,
    NamespaceService,
    NamespaceValidationError,
    count_sentences,
    normalize_slug,
)
from app.storage.db import init_db, session


@pytest.fixture
def service() -> NamespaceService:
    init_db(get_settings())
    return NamespaceService(get_settings())


class TestSlug:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("work", "work"),
            ("Work", "work"),
            ("  work  ", "work"),
            ("Work / SBOS 2020", "work-sbos-2020"),  # один слаг, не путь
            ("sbos2020", "sbos2020"),
            ("подпроект-2024", "2024"),  # кириллица не транслитерируется
        ],
    )
    def test_normalize_slug(self, raw: str, expected: str) -> None:
        assert normalize_slug(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "СУБО", "!!!"])
    def test_normalize_slug_unsalvageable(self, raw: str) -> None:
        assert normalize_slug(raw) is None

    def test_validate_path_normalizes(self, service: NamespaceService) -> None:
        assert service.validate_path("work") == "work"
        assert service.validate_path("Work / SBOS 2020") == "work/sbos-2020"
        assert service.validate_path("default") == "default"

    def test_validate_path_rejects_depth3(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.validate_path("a/b/c")

    def test_validate_path_rejects_cyrillic(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.validate_path("СУБО")

    def test_validate_path_rejects_empty_and_long(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.validate_path("   ")
        with pytest.raises(NamespaceValidationError):
            service.validate_path("a" * 300)

    def test_count_sentences(self) -> None:
        assert count_sentences("Одно предложение.") == 1
        assert count_sentences("Первое. Второе.") == 2
        assert count_sentences("Раз. Два. Три.") == 3
        assert count_sentences("Тире — не граница: продолжаем мысль!") == 1


class TestDescriptionContract:
    """Контракт описаний (решение О. 2026-09-03): ≤2 кратких предложений."""

    def test_one_and_two_sentences_are_valid(self, service: NamespaceService) -> None:
        assert service.validate_description("Рабочие заметки.") == "Рабочие заметки."
        assert (
            service.validate_description("Рабочие заметки. Внутри — подпроекты.")
            == "Рабочие заметки. Внутри — подпроекты."
        )

    def test_three_sentences_is_invalid(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.validate_description("Раз. Два. Три.")

    def test_empty_is_invalid(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.validate_description("   ")


class TestCreate:
    def test_create_returns_node(self, service: NamespaceService) -> None:
        node = service.create("work", "Рабочие заметки. Подпроекты — в листьях.")
        assert node == {
            "path": "work",
            "description": "Рабочие заметки. Подпроекты — в листьях.",
            "status": "confirmed",
            "notes_count": 0,
            "subtree_count": 0,
            "created_at": node["created_at"],
            "updated_at": node["updated_at"],
        }
        assert service.exists("work") is True

    def test_duplicate_is_error(self, service: NamespaceService) -> None:
        service.create("work", "Рабочие заметки.")
        with pytest.raises(NamespaceError):
            service.create("work", "Другое описание.")

    def test_leaf_requires_registered_parent(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceError):
            service.create("work/sbos2020", "Лист без корня.")
        service.create("work", "Корень есть.")
        node = service.create("work/sbos2020", "Лист СУБО 2020.")
        assert node["path"] == "work/sbos2020"

    def test_invalid_description_is_rejected(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.create("work", "Раз. Два. Три.")

    def test_invalid_status_is_rejected(self, service: NamespaceService) -> None:
        with pytest.raises(NamespaceValidationError):
            service.create("work", "Рабочие заметки.", status="draft")


class TestListAndCounters:
    def test_counts_node_and_subtree(self, service: NamespaceService) -> None:
        init_db(get_settings())
        service.create("work", "Рабочие заметки.")
        service.create("work/sbos2020", "СУБО 2020: сервисы HR.")
        service.create("projects", "Личные проекты.")
        with session(get_settings()) as conn:
            for text, namespace in [
                ("заметка 1", "work"),
                ("заметка 2", "work"),
                ("заметка 3", "work/sbos2020"),
                ("заметка 4", "work/sbos2020"),
                ("заметка 5", "projects"),
                ("trash-заметка", "work"),
            ]:
                conn.execute(
                    "INSERT INTO notes (text, namespace) VALUES (?, ?)",
                    (text, namespace),
                )
            # trash: не считается ни notes_count, ни subtree_count
            conn.execute(
                "UPDATE notes SET deleted_at = '2026-01-01T00:00:00Z' "
                "WHERE text = 'trash-заметка'"
            )
        registry = {node["path"]: node for node in service.list_all()["namespaces"]}
        assert registry["work"]["notes_count"] == 2
        assert registry["work"]["subtree_count"] == 4  # 2 своих + 2 листа
        assert registry["work/sbos2020"]["notes_count"] == 2
        assert registry["work/sbos2020"]["subtree_count"] == 2  # лист: только себя
        assert registry["projects"]["subtree_count"] == 1
        assert registry["default"]["notes_count"] == 0

    def test_get_unknown_node_is_none(self, service: NamespaceService) -> None:
        assert service.get("work") is None


class TestMutations:
    def test_update_description(self, service: NamespaceService) -> None:
        service.create("work", "Старое описание.")
        node = service.update_description("work", "Новое описание. Одно предложение.")
        assert node is not None
        assert node["description"] == "Новое описание. Одно предложение."

    def test_update_description_unknown_node(self, service: NamespaceService) -> None:
        assert service.update_description("work", "Новое описание.") is None

    def test_set_status(self, service: NamespaceService) -> None:
        service.create("work", "Рабочие заметки.")
        node = service.set_status("work", "provisional")
        assert node is not None and node["status"] == "provisional"

    def test_set_status_unknown_node(self, service: NamespaceService) -> None:
        assert service.set_status("work", "confirmed") is None

    def test_set_status_invalid(self, service: NamespaceService) -> None:
        service.create("work", "Рабочие заметки.")
        with pytest.raises(NamespaceValidationError):
            service.set_status("work", "draft")