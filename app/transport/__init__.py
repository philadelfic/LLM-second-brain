"""The transport layer (ARCHITECTURE §3.1): the MCP server + REST handles.

This holds: FastMCP/MCPServer with the 6 `memory_*` tools, thin REST wrappers
over the same services, and the Bearer-authentication middleware.
"""