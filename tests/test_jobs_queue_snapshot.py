"""Снимок очереди: ожидание и внимание различимы (№39, 3.2.0; arch §6.9/§7.2).

`queue_snapshot` — обратно совместимое расширение lsb-0014-03: старый вызов
`queue_snapshot(n, age)` остаётся валидным, а снимок несёт ещё `waiting`
(отложено слотом/готовностью) и `attention` (подмножество `waiting`, ждущее
человека). Пустая очередь — нули и `oldest_pending_sec=None`.
"""

from __future__ import annotations

from app.services.jobs import queue_snapshot


def test_snapshot_backward_compatible() -> None:
    """Старый вызов без счётчиков: waiting=0, attention=0 (совместимость)."""
    assert queue_snapshot(3, 42) == {
        "pending": 3,
        "oldest_pending_sec": 42,
        "waiting": 0,
        "attention": 0,
    }


def test_snapshot_empty_queue() -> None:
    """Пустая очередь: возраст старейшего — null, счётчики нулевые."""
    assert queue_snapshot(0, None) == {
        "pending": 0,
        "oldest_pending_sec": None,
        "waiting": 0,
        "attention": 0,
    }
    # pending=0 при переданном возрасте — тоже «пусто» (null, не отрицательное).
    assert queue_snapshot(None, 10) == {
        "pending": 0,
        "oldest_pending_sec": None,
        "waiting": 0,
        "attention": 0,
    }


def test_snapshot_passes_counters() -> None:
    """Счётчики ожидания и внимания проходят без изменений (№39)."""
    assert queue_snapshot(0, None, waiting=2, attention=1) == {
        "pending": 0,
        "oldest_pending_sec": None,
        "waiting": 2,
        "attention": 1,
    }
