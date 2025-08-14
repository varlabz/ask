# Blog Post Multi-Agent Pipeline

This example demonstrates a multi-step, multi-agent workflow for generating, refining, and critiquing a long-form blog post using the `ask` framework + Model Context Protocol (MCP) tools.

## Pipeline Overview
Steps executed (in order) by `blog_post.py`:
1. Research (`research.yaml`): Gathers structured topical research using search + fetch + sequential thinking tools. Produces a research report and list of source URLs.
2. Outline (`outline.yaml`): Converts topic + research summary into an informal, narrative-friendly outline.
3. Post Draft (`post.yaml`): Expands the outline into a full blog post, enriching with research details; returns full article + (optionally) URLs used.
4. Critique & Score (`score.yaml`): Acts as an editor scoring the article across 5 dimensions (Clarity, Depth, Structure, Objectivity, Style) with /20 scores and final /100 total.

Each step is an agent instantiation with its own instructions (and potentially distinct tools / model config) orchestrated sequentially. Intermediate outputs are passed as structured prompt fragments (`# topic: ...`, `# research: ...`, etc.).

## Files
- `blog_post.py` – Orchestrator script. Creates agents from config files and feeds step outputs forward.
- `research.yaml` – Agent instructions + MCP tool definitions for search, fetch, and sequential reasoning.
- `outline.yaml` – Agent instructions to transform research into a narrative outline (sequential thinking tool only).
- `post.yaml` – Agent instructions for drafting the full article from topic + research + outline.
- `score.yaml` – Agent instructions for structured editorial critique and scoring.
- `llm.yaml` – Local/default LLM configuration (model, base URL, temperature). Merged with each agent’s config.

## How It Works
`blog_post.py` calls `AgentASK.create_from_file([...config_paths], name=step)` combining:
- A default LLM config (`~/.config/ask/llm.yaml`) – user-level override.
- Local step-specific YAMLs (in this folder) including `llm.yaml` to pin/override runtime model.
The agent then runs with the composed instructions and available MCP tool mounts defined under the `mcp:` key (when present).

Prompt chaining pattern (simplified):
- Research prompt: `<query>`
- Outline prompt: `# topic: <query>\n# research: <research_output>`
- Post prompt: `# topic: <query>\n# research: <research_output>\n# outline: <outline_output>`
- Critique prompt: `# topic: <query>\n# article: <post_output>`

## Prerequisites
- Python 3.13 environment. Either install this repo or set PYTHONPATH when running from source.
- Access to the model defined in `llm.yaml` (e.g., local LM Studio endpoint at `base_url`). Adjust as needed.
- uv (for `uvx` helpers used by MCP tools) and Node.js (for `npx` sequential thinking MCP server).
- (Optional) SearXNG instance for search (set `SEARX_HOST` in `research.yaml`; update from the placeholder to your instance).

## Running the Pipeline
From repository root, pick one:

- Installed package (recommended during dev):
  ```bash
  pip install -e .
  python examples/blog_post/blog_post.py "Your Topic Here"
  ```

- Run from source without install (ensure src is on PYTHONPATH):
  ```bash
  PYTHONPATH=src python examples/blog_post/blog_post.py "Your Topic Here"
  ```
Add `--logs` to persist step prompts/responses to `blog_post.log`:
```bash
python examples/blog_post/blog_post.py --logs "History of Large Language Models"
```
Output:
- STDOUT: Final blog post, then critique report (scores + summary)
- STDERR: Step banners (`### research`, etc.)
- Log file (if enabled): Full prompts and responses for auditing.

## Customization
- Swap Model: Edit `llm.yaml` (or your global `~/.config/ask/llm.yaml`).
- Add / Remove Tools: Modify `mcp:` section per YAML (e.g., add a vector store retrieval server).
- Change Style: Tweak instruction prose in `post.yaml` to alter narrative voice.
- Insert New Step: Create `something.yaml`, then insert another `await run([...])` call in `main()` with appropriate prompt assembly.
- Independent Critique Model: Provide an alternate `llm.yaml` (e.g., higher reasoning model) and reference only in the critic step configs.

## Extending With Additional Quality Gates
Potential extra steps you can add:
- Fact Check: Agent verifying claims against fresh search results.
- SEO Optimization: Agent that suggests meta description, title variants, keywords.
- Plagiarism / Originality Heuristic: Agent that paraphrases flagged sections.

## Error Handling Notes
- If a tool server (search/fetch) isn’t reachable, the research step may degrade or fail—ensure endpoints are running.
- Long outputs may exceed context windows; adjust temperature or model selection for stability.

## Minimal New Step Template
Example YAML (add as `seo.yaml`):
```yaml
agent:
  instructions: >
    You are an SEO analyst. Given the topic and draft article, propose:
    - 3 optimized titles
    - Meta description (<= 155 chars)
    - 10 target long-tail keywords
    - 5 internal linking anchor suggestions
```
Then in `blog_post.py` after `post`:
```python
seo = await run(["seo.yaml", LLM], f"# topic: {query}\n# article: {post}\n", step="seo")
```

## License / Attribution
This example is part of the `ask` project. Adjust endpoints and configs responsibly; ensure you have permission to query any external services you connect.

---
Happy writing!
