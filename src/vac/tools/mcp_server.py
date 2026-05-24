"""Local MCP server exposing the small VAC probe tools."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from vac.tools import local_tools


mcp = FastMCP("VAC Local Tool Server", json_response=True)


@mcp.tool()
def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a safe arithmetic expression using +, -, *, /, and parentheses."""
    return local_tools.calculator(expression)


@mcp.tool()
def read_document(document_id: str) -> dict[str, Any]:
    """Read one named local fixture document by document_id."""
    return local_tools.read_document(document_id)


@mcp.tool()
def list_documents() -> dict[str, Any]:
    """List available local document ids."""
    return local_tools.list_documents()


if __name__ == "__main__":
    mcp.run(transport="stdio")
