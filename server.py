"""MCP server that lets an AI assistant query a set of documents."""
import asyncio
import logging

from mcp.server import MCPServer

import config
from rag_utils import answer_question, get_model, search

# Logs go to a file and to stderr: stdout is reserved for the MCP protocol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # hides every HTTP request to the APIs
logger = logging.getLogger(__name__)

mcp = MCPServer("mcp-rag-server")

STATUSES = {
    "reliable": "RELIABLE: the answer is supported by the documents",
    "to_verify": "TO VERIFY: the answer may drift away from the documents",
    "refusal": "NO ANSWER: the information was not found in the documents",
}


def format_result(result):
    lines = [result["answer"], "", "---"]
    if result.get("intent"):
        intent = f"Detected intent: {result['intent']}"
        if result.get("doc_type"):
            intent += f" (search limited to: {result['doc_type']})"
        lines.append(intent)
    lines.append("Sources: " + (", ".join(result["sources"]) or "none"))
    status = STATUSES[result["status"]]
    if result["overall_score"] is not None:
        status += f" (score {result['overall_score']:.2f})"
    lines.append("Reliability: " + status)
    lines.append(f"Generation time: {result['latency_s']} s")
    return "\n".join(lines)


@mcp.tool()
async def search_documents(query: str) -> str:
    """Returns the document excerpts most relevant to a question, with their source.

    Args:
        query: The question or topic to search for
    """
    logger.info(f"Search: {query}")
    # Embeddings are blocking: they run outside the asyncio event loop
    passages = await asyncio.to_thread(search, query)
    return "\n\n---\n\n".join(
        f"[{p['source']}, excerpt {p['chunk']}, similarity {p['similarity']:.2f}]\n{p['text']}"
        for p in passages
    )


@mcp.tool()
async def ask_question(query: str) -> str:
    """Answers a question from the documents, citing the sources
    and giving a reliability status.

    Args:
        query: The question or topic to search for
    """
    logger.info(f"Question: {query}")
    # The LLM and the embeddings are blocking: they run outside the asyncio event loop
    result = await asyncio.to_thread(answer_question, query)
    logger.info(
        f"Intent: {result['intent']} | doc type: {result['doc_type']} | "
        f"status: {result['status']} | score: {result['overall_score']} | "
        f"scores: {result['scores']} | latency: {result['latency_s']} s"
    )
    return format_result(result)


if __name__ == "__main__":
    logger.info("Loading the embedding model...")
    get_model()  # loaded at startup so that the first question is not slow
    logger.info(f"Starting the MCP server (LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}, "
                f"router: {'on' if config.USE_ROUTER else 'off'})")
    mcp.run()  # stdio transport: Claude Desktop launches this script and talks over stdin/stdout
