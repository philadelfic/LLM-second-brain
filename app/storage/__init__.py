"""Storage-слой (ARCHITECTURE §3.3): SQLite.

Здесь живут: схема notes + FTS5 (trigram) + vec0, работа с соединением
(WAL, busy_timeout), миграции/инициализация.
"""