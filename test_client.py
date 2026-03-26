import asyncio 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"]  )
    
    async with stdio_client(server_params) as (read, write): #ouvre une connection avec le serveur (lire et envoyer des requetes)
        async with ClientSession(read, write) as session: #session pour communiquer avec le serveur
            await session.initialize()

            #print(" OUTILS DISPONIBLES ") 
            #tools= await session.list_tools()
            #for tool in tools.tools:
                #print(f"• {tool.name} : {tool.description}")
     
            questions = [
             "Which software tools are allowed for work purposes?", "How often must employees update their passwords?",
             "What is the consequence for a minor violation of this policy?", 
            "What task did David complete before this meeting?",
             "What is the precise cause of the 3-day delay in Phase B?"
            ]

             # tester chaque question avec les 2 outils
            for q in questions:
                print("\n" + "=" * 60)
                print("QUESTION :")
                print(q)

                # afficher les chunks récupérés
                print("\n TEST rechercher_documents ")
                resultat_docs = await session.call_tool("rechercher_documents", {"query": q} )
                print(resultat_docs.content[0].text)

                # afficher la réponse générée par le LLM
                print("\n TEST poser_question ")
                resultat_reponse = await session.call_tool("poser_question",{"query": q} )
                print(resultat_reponse.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())

