"""
MCP server exposing the same three tools from the agent project
(calculator, knowledge_base_lookup, read_file) over the Model Context
Protocol, using FastMCP — the simplest way to write an MCP server.

Why this matters vs. the agent project:
  In agent/loop.py, only OUR script could call these tools — they
  were hard-wired into one Python process. An MCP server flips that:
  any MCP-compatible client (Claude Desktop, Claude Code, other agents,
  IDEs, etc.) can discover and call these same tools, without knowing
  anything about how they're implemented. The tool logic itself didn't
  change at all — only how it's exposed.

Run it:
    pip install -r requirements.txt
    python server.py

Connect it to Claude Desktop (or another MCP client) by pointing the
client's MCP config at this script — see README.md for the exact config.
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

from tools.implementations import calculator, knowledge_base_lookup, read_file
from tools.hybrid_search import HybridSearchIndex

# --- Logging setup ---
# IMPORTANT: this server talks to its client (e.g. Claude Desktop) over
# stdio — stdin/stdout is the actual protocol channel. Any print() or
# logging to stdout would corrupt that stream and break the connection.
# All logging here goes to a file instead, never to the console.
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(_LOG_DIR, "server.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("local-tools")

# The name shown to MCP clients when they connect to this server.
mcp = FastMCP("local-tools")

# Built once at server startup, not per-query — the document set is
# fixed while the server runs, so there's no reason to re-index on
# every search call.
logger.info("Building hybrid search index...")
_search_index = HybridSearchIndex()
logger.info(f"Indexed {len(_search_index.doc_names)} documents: {_search_index.doc_names}")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.
    Supports +, -, *, /, **, %, parentheses, and functions like
    sqrt(), pow(), abs(), log(), sin(), cos(), floor(), ceil(), round(),
    factorial(). Use this for any arithmetic instead of computing it
    yourself.

    Args:
        expression: A math expression, e.g. "(23 * 4) / 2" or "sqrt(144)"
    """
    logger.info(f"calculate called with expression={expression!r}")
    result = calculator({"expression": expression})
    if "error" in result:
        logger.warning(f"calculate error for {expression!r}: {result['error']}")
        return f"Error: {result['error']}"
    return str(result["result"])


@mcp.tool()
def lookup_knowledge_base(query: str) -> str:
    """
    Look up an entry in the local knowledge base (a small mock company
    database of pricing, refund policy, support hours, API rate limits,
    and data retention) by keyword.

    Args:
        query: Keyword or topic to search for, e.g. "refund policy"
    """
    logger.info(f"lookup_knowledge_base called with query={query!r}")
    result = knowledge_base_lookup({"query": query})
    if "error" in result:
        logger.warning(f"lookup_knowledge_base error for {query!r}: {result['error']}")
        return f"Error: {result['error']}"
    if not result.get("matches"):
        logger.info(f"lookup_knowledge_base found no matches for {query!r}")
        return "No matching entry found in knowledge base."
    lines = [f"{topic}: {answer}" for topic, answer in result["matches"].items()]
    return "\n".join(lines)


@mcp.tool()
def read_workspace_file(filename: str) -> str:
    """
    Read the contents of a text file from the local workspace directory.
    Only files inside the workspace folder can be read (no path
    traversal outside it).

    Args:
        filename: Name of the file to read, e.g. "sample_notes.txt"
    """
    logger.info(f"read_workspace_file called with filename={filename!r}")
    result = read_file({"filename": filename})
    if "error" in result:
        # Path traversal / invalid-filename attempts are worth a
        # distinct warning, since a pattern of these could indicate
        # something worth investigating rather than routine misuse.
        logger.warning(f"read_workspace_file error for {filename!r}: {result['error']}")
        return f"Error: {result['error']}"
    return result["result"]


@mcp.tool()
def search_documents(query: str, top_k: int = 3) -> str:
    """
    Search company policy documents using hybrid search — combining
    keyword matching (BM25) with semantic/topic-based matching (LSA)
    for better results than either alone. Use this for open-ended
    questions about company policy (onboarding, remote work, expenses,
    security, performance reviews) where the exact wording of the
    answer may differ from the question. Returns the full text of each
    matching document (they're short — a few hundred words each), not
    a truncated snippet, so answers can cite specific details without
    guessing at content that got cut off.

    Args:
        query: What to search for, e.g. "can I work from home"
        top_k: How many top matching documents to return (default 3)
    """
    logger.info(f"search_documents called with query={query!r}, top_k={top_k}")
    results = _search_index.search(query, top_k=top_k)
    if not results:
        logger.info(f"search_documents found no results for {query!r}")
        return "No matching documents found."

    matched_docs = [r["document"] for r in results]
    logger.info(f"search_documents matched: {matched_docs}")

    lines = []
    for r in results:
        lines.append(f"[{r['document']}] (relevance: {r['score']})\n{r['text']}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    # stdio transport: the client (e.g. Claude Desktop) launches this
    # script as a subprocess and talks to it over stdin/stdout. This is
    # the standard transport for local MCP servers.
    mcp.run(transport="stdio")