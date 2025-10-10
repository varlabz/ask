import json
import sys
from textwrap import dedent
from typing import Final

from attr import dataclass
from langfuse import Evaluation, Langfuse

from ask.core.agent import AgentASK
from ask.core.config import MCPServerConfig
from eval.agent import create_config, make_llm_config
from eval.data import (
    function_tools,
    serialize_config,
    task_executor_agent,
)
from eval.instrumentation import setup_instrumentation

langfuse: Final[Langfuse] = setup_instrumentation()


@dataclass
class Metadata:
    instructions: str


DATASET: Final[str] = "list_of_tools_max"
try:
    dataset = langfuse.get_dataset(name=DATASET)
except Exception:
    langfuse.create_dataset(
        name=DATASET,
        description="list of tools",
        metadata=Metadata(
            instructions=dedent(
                """
                    Must return a JSON array containing the names of the tools. No other text.
                    Example of the output:
                    [
                        {"name": "tool_name1"},
                        {"name": "tool_name2"},
                        ...
                    ]
                """,
            ),
        ),
    )
    input = [
        "list of tools",
        "print a list of available tools what you have.",
        "show a list of available tools what you have.",
        "display a list of available tools what you have.",
        "print/show a list of available tools what you have.",
    ]
    for i in input:
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            input=i,
        )

dataset = langfuse.get_dataset(name=DATASET)
if dataset.metadata and isinstance(dataset.metadata, dict):
    dataset.metadata = Metadata(**dataset.metadata)


def _accuracy_evaluator(*, input, output, expected_output, size=1):
    """Evaluator that checks if the expected answer is in the output"""
    try:
        tool_names = [tool["name"] for tool in function_tools]
        arr = json.loads(output)  # validate JSON
        cnt = sum(1 for i in arr if i["name"] in tool_names)
        return Evaluation(name="accuracy", value=1 if cnt == size else 0)
    except Exception as e:
        print(f"Output is not valid JSON {e}", file=sys.stderr)
        return Evaluation(name="accuracy", value=0.0)


tools = [
    "--echo",
    "--add",
    "--long-running",
    "--print-env",
    "--sample-llm",
    "--tiny-image",
    "--annotated",
    "--resource-ref",
    # # "--elicitation",      # doesn't work well, skip for now
    "--resource-links",
    "--structured",
    # # "--list-roots",       # doesn't work well, skip for now
]


def run_experiment(model: str, base_url: str, session_id: str):
    for i, _ in enumerate(tools):
        mcp = {
            "everything": MCPServerConfig(
                command=[
                    "npx",
                    "-y",
                    "https://github.com/varlabz/everything-mcp",
                    "stdio",
                ]
                + tools[: i + 1]
            )
        }
        config = create_config(
            llm=make_llm_config(model=model, base_url=base_url),
            mcp=mcp,
            instructions=dataset.metadata.instructions if dataset.metadata else "",
        )
        agent = AgentASK.create_from_config(config=config)
        result = dataset.run_experiment(
            name=f"tools listing with {i + 1} tools",
            run_name=f"{config.llm.model} with {i + 1} tools",
            description=f"testing tools listing functionality with {i + 1} tools",
            task=lambda *, item, **kwargs: task_executor_agent(
                agent=agent,
                item=item,
                callback=lambda: langfuse.update_current_trace(
                    session_id=session_id,
                    user_id="eval",
                    tags=[config.llm.model],
                ),
            ),
            evaluators=[
                lambda *, input, output, expected_output, **kwargs: _accuracy_evaluator(
                    input=input,
                    output=output,
                    expected_output=expected_output,
                    size=i + 1,
                )
            ],
            max_concurrency=1,  # Limit concurrency to 1 to avoid race timing issues
            metadata={
                "config": serialize_config(config),
            },
        )
        print(result.format(), file=sys.stderr)
