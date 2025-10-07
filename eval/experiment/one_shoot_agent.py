import argparse
import json
import sys
from textwrap import dedent
from typing import Final

from langfuse import Evaluation, Langfuse
from langfuse.experiment import LocalExperimentItem

from ask.core.config import Config

from ..agent import create_config, local
from ..instrumentation import setup_instrumentation
from .data import function_tools, serialize_config, task_executor

parser = argparse.ArgumentParser(
    description="Run experiments with a specified model as provider:model"
)
parser.add_argument(
    "-m", "--model", required=True, help="The model to use for the experiment."
)
parser.add_argument(
    "-u",
    "--base-url",
    default="http://bacook.local:11434/v1",
    help="The base URL for the model API.",
)
parser.add_argument(
    "-s",
    "--session-id",
    default="eval_session",
    help="The session ID for the experiment.",
)
args = parser.parse_args()

print(
    f">>> model: {args.model}, base_url: {args.base_url}, session_id: {args.session_id}",
    file=sys.stderr,
)

session_id: Final[str] = args.session_id
config: Final[Config] = create_config(
    llm=local(model=args.model, base_url=args.base_url),
)


async def task_tools(cfg: Config, item, **kwargs):
    return await task_executor(cfg, item, langfuse, session_id, **kwargs)


def accuracy_evaluator(*, input, output, expected_output, **kwargs):
    """Evaluator that checks if the expected answer is in the output"""
    try:
        tool_names = [tool["name"] for tool in function_tools]
        arr = json.loads(output)  # validate JSON
        cnt = sum(1 for i in arr if i["name"] in tool_names)
        return Evaluation(name="accuracy", value=cnt / len(tool_names))
    except Exception as e:
        print(f"Output is not valid JSON {e}", file=sys.stderr)
        return Evaluation(name="accuracy", value=0.0)


langfuse: Final[Langfuse] = setup_instrumentation()

def experiment_list_tools():
    inputs_tools = [
        LocalExperimentItem(
            input="Print/show a list of available tools what you have.",
            metadata={
                "input_type": str,
                "output_type": str,
                "instructions": dedent("""
                    Must return a JSON array containing the names of the tools. No other text.
                    Example of the output:
                    [
                        {"name": "tool_name1"},
                        {"name": "tool_name2"},
                        ...
                    ]
                """),
            },
        ),
        LocalExperimentItem(
            input="Print/show a list of available tools what you have with description.",
            metadata={
                "input_type": str,
                "output_type": str,
                "instructions": dedent("""
                    Must return a JSON array containing the names and descriptions of the tools. No other text.
                    Example of the output:
                    [
                        {"name": "tool_name1", "description": "Description of tool_name1"},
                        {"name": "tool_name2", "description": "Description of tool_name2"},
                        ...
                    ]
                """),
            },
        ),
    ]

    result_tools = langfuse.run_experiment(
        name="tools listing",
        description="testing tools listing functionality",
        data=inputs_tools,
        task=lambda *, item, **kwargs: task_tools(config, item=item, **kwargs),
        evaluators=[accuracy_evaluator],
        max_concurrency=1,  # Limit concurrency to 1 to avoid race timing issues
        metadata={
            "config": serialize_config(config),
        },
    )
    print(result_tools.format(), file=sys.stderr)

experiment_list_tools()
