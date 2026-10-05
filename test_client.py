"""Test client: calls both tools of the MCP server on a few questions."""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUESTIONS = [
    "What is an abstract?",
    "Compare Option A and Option B of the AI assignment.",
    "What is the capital of Australia?",
]


async def main():
    # Same Python as the one running this script (the venv's), absolute path to the server
    server = Path(__file__).resolve().parent / "server.py"
    params = StdioServerParameters(command=sys.executable, args=[str(server)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", ", ".join(t.name for t in tools.tools))

            for question in QUESTIONS:
                print("\n" + "=" * 60 + f"\nQUESTION: {question}")
                print("\n> search_documents")
                res = await session.call_tool("search_documents", {"query": question})
                print(res.content[0].text)
                print("\n> ask_question")
                res = await session.call_tool("ask_question", {"query": question})
                print(res.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
