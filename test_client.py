"""Client de test : appelle les deux outils du serveur MCP sur quelques questions."""
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUESTIONS = [
    "Which software tools are allowed for work purposes?",
    "How often must employees update their passwords?",
    "What is the consequence for a minor violation of this policy?",
    "What task did David complete before this meeting?",
    "What is the precise cause of the 3-day delay in Phase B?",
]


async def main():
    params = StdioServerParameters(command="python", args=["server.py"])
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
