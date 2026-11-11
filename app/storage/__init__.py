"""The storage layer (ARCHITECTURE §3.3): SQLite.

This package holds: the notes schema + FTS5 (trigram) + vec0, connection
handling (WAL, busy_timeout), migrations/initialization.
"""