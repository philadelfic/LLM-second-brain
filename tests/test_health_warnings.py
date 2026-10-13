"""№12 (no12-02, 3.2.0): агрегаты `/health.warnings` (arch §7.1/§7.2).

`/health.warnings` = `{by_kind: {kind: N}, total: M}` — считает ОТКРЫТЫЕ
предупреждения по видам (только SQL, без модели). Виды памяти и терминов
видны в одном объекте: одна поверхность на систему (§13.4).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.warnings import (
    KIND_NOTE_MISPLACED,
    KIND_NOTE_NO_NODE,
    KIND_TERM_CONTEXTS_EXCESS,
)


def test_health_warnings_aggregate(client: TestClient) -> None:
    """/health.warnings = {by_kind, total}, открытые по видам; снятые — нет."""
    service = client.app.state.services.warnings  # type: ignore[attr-defined]
    service.open(KIND_NOTE_MISPLACED, "note", 1, "m1", {})
    service.open(KIND_NOTE_MISPLACED, "note", 2, "m2", {})
    service.open(KIND_NOTE_NO_NODE, "note", 3, "m3", {})
    dismissed = service.open(KIND_TERM_CONTEXTS_EXCESS, "term", 9, "m4", {})
    service.dismiss(dismissed)

    body = client.get("/health").json()
    assert body["warnings"] == {
        "by_kind": {
            KIND_NOTE_MISPLACED: 2,
            KIND_NOTE_NO_NODE: 1,
        },
        "total": 3,
    }
