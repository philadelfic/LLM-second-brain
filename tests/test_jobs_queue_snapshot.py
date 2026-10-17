"""Снимок очереди: ожидание и внимание различимы (№39, 3.2.0; arch §6.9/§7.2).

`queue_snapshot` — обратно совместимое расширение lsb-0014-03: старый вызов
`queue_snapshot(n, age)` остаётся валидным, а снимок несёт ещё `waiting`
(отложено слотом/готовностью) и `attention` (подмножество `waiting`, ждущее
человека). Пустая очередь — нули и `oldest_pending_sec=None`.

3.3.1 (LSBN-S-3): возраст — по pending+waiting (вся работа, ждущая
исполнения) — при разрыве слота работа переезжает pending → waiting, и
возраст больше не гаснет; null — только когда обе категории пусты.
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


def test_snapshot_age_survives_waiting_only_queue() -> None:
    """Суть фикса 3.3.1 (LSBN-S-3): pending=0, waiting>0 — возраст сохранён.

    При разрыве слота работа переезжает pending → waiting (возраст тот же),
    поэтому гасить его при пустом pending — ложное «очередь пуста»: возраст
    гаснет, только когда ждущая исполнения работа кончилась совсем.
    """
    assert queue_snapshot(0, 42, waiting=3) == {
        "pending": 0,
        "oldest_pending_sec": 42,
        "waiting": 3,
        "attention": 0,
    }


def test_snapshot_age_hidden_only_when_both_empty() -> None:
    """Старые вызовы без waiting ведут себя как раньше (регресс).

    `queue_snapshot(0, age)` → null (обе категории пусты); `queue_snapshot(5,
    age)` → возраст (непустой pending сохраняет его, как и до 3.3.1).
    """
    assert queue_snapshot(0, 42)["oldest_pending_sec"] is None
    assert queue_snapshot(None, 42)["oldest_pending_sec"] is None
    assert queue_snapshot(5, 42) == {
        "pending": 5,
        "oldest_pending_sec": 42,
        "waiting": 0,
        "attention": 0,
    }
