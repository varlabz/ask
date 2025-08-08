import asyncio
import sys

from agent import AgentASK

async def main(query):
    research = await AgentASK.create_from_file(["research.yaml", "llm.yaml"]).run(query)
    outline = await AgentASK.create_from_file(["outline.yaml", "llm.yaml"]).run(
        f"topic: {query}\n"
        f"research: {research}\n"
    )
    post = await AgentASK.create_from_file(["post.yaml", "llm.yaml"]).run(
        f"topic: {query}\n"
        f"research: {research}\n"
        f"outline: {outline}\n"
    )
    print(post)

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the current state of AI research in 2023? What are the most promising areas of AI development?"
    asyncio.run(main(query))
