"""Transport-слой (ARCHITECTURE §3.1): MCP-сервер + REST-ручки.

Здесь живут: FastMCP/MCPServer с 6 инструментами `memory_*`, тонкие REST
обёртки над теми же сервисами, миддлварь Bearer-аутентификации.
"""