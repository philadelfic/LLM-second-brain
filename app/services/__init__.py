"""Services-слой (ARCHITECTURE §3.2): бизнес-логика над хранилищем.

Здесь живут: NoteService (CRUD), SearchService (гибрид + RRF),
EmbeddingService и SummaryService (клиенты двух Ollama), DedupService,
BackgroundWorker (до-векторизация/досуммаризация), BackupService.
"""