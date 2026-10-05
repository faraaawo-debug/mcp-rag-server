"""Client de test : appelle les deux outils du serveur MCP sur quelques questions."""
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
    # Même Python que celui qui lance ce script (celui du venv), chemin absolu vers le serveur
    serveur = Path(__file__).resolve().parent / "server.py"
    params = StdioServerParameters(command=sys.executable, args=[str(serveur)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            outils = await session.list_tools()
            print("Outils disponibles :", ", ".join(t.name for t in outils.tools))

            for question in QUESTIONS:
                print("\n" + "=" * 60 + f"\nQUESTION : {question}")
                print("\n> rechercher_documents")
                res = await session.call_tool("rechercher_documents", {"query": question})
                print(res.content[0].text)
                print("\n> poser_question")
                res = await session.call_tool("poser_question", {"query": question})
                print(res.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
