#!/usr/bin/env -S uvx --from git+https://github.com/varlabz/ask ask-run

# Programmatic research pipeline using AgentASK, based on the YAML instructions in examples/research
# - Uses searxng search, fetch, youtube, converter (HTML->Markdown), memory, and sequential thinking tools
# - Performs iterative research, compiles a comprehensive report, and emits a short summary and provenance

import asyncio
import os
import sys

from ask.core.agent_cache import CacheASK

sys.path.insert(0, os.path.dirname(__file__))
from assess import AssessInput, assess_agent
from fetch import FetchInput, fetch_agent
from finalize import FinalizeInput, finalize_agent
from query_gen import QueryGenInput, query_gen_agent
from report import ReportInput, report_agent
from search import SearchInput, search_agent

# 1. Given the user's query, generate a list of 3 distinct,
#    precise search queries that would help gather comprehensive information on the topic.
#    Consider different angles and aspects of the topic to ensure a thorough exploration.
# 2. Iterate on array of requests from step 1, use search tool to collect URLs of searched results.
#    Use search tool with categories web,videos,news,it,science,social media.
#    Collect unique URLs only.
# 3. For each URL from step 2, use fetch tool to get the content.
#    Use youtube tool for youtube URLs.
#    Use converter tool to convert HTML content to markdown format.
#    Accumulate the content from all fetched pages.
#    Use memory tool to store the markdown content with metadata including source URL and title.
# 4. Based on the original query, the search queries performed so far,
#    and the extracted contexts from webpages, determine if further research is needed.
#    If further research is needed, start from step 1 with new search queries.
#    If you believe no further research is needed, do the next steps.
# 5. Based on the gathered contexts and the original query,
#    write a comprehensive, well-structured, and detailed report that addresses the user's query thoroughly.
#    Include all relevant insights and conclusions without extraneous commentary.
# 6. Create a full report of what queries were made, what URLs were fetched, how many pages were retrieved, and rejected, and the final short summary.


async def main(user_query: str) -> None:
    cache = CacheASK()
    all_queries = []
    all_urls = set()
    total_retrieved = 0
    total_rejected = 0

    query_res = await query_gen_agent.cache(cache).run(QueryGenInput(query=user_query))
    queries = query_res.queries
    while True:
        all_queries.extend(queries)
        for q in queries:
            all_queries.append(q)
            search_res = await search_agent.cache(cache).run(SearchInput(query=q))
            new_urls = set(search_res.urls) - all_urls
            all_urls.update(new_urls)
            for url in new_urls:
                fetch_res = await fetch_agent.cache(cache).run(FetchInput(urls=url))
                total_retrieved += fetch_res.pages_retrieved
            total_rejected += fetch_res.pages_rejected

        # Assess if further research is needed
        assess_res = await assess_agent.cache(cache).run(
            AssessInput(query=user_query, queries_so_far=all_queries)
        )
        if not assess_res.continue_research:
            break
        queries = assess_res.new_queries

    # Generate comprehensive report
    report_res = await report_agent.cache(cache).run(
        ReportInput(query=user_query, urls=list(all_urls))
    )

    # Finalize and output
    finalize_res = await finalize_agent.cache(cache).run(
        FinalizeInput(
            queries=all_queries,
            urls=list(all_urls),
            pages_retrieved=total_retrieved,
            pages_rejected=total_rejected,
            report=report_res.report,
            summary=report_res.summary,
        )
    )
    cache.clean()

    print(finalize_res.model_dump_json(indent=2))
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <research query>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
