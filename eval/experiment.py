import argparse
import json
import sys
from textwrap import dedent
from typing import Final

from langfuse import Evaluation, Langfuse
from langfuse.experiment import LocalExperimentItem
from pydantic import (
    BaseModel,
    Field,
)

from ask.core.agent import AgentASK
from ask.core.config import Config

from .agent import create_config, local
from .experiment_data import function_tools
from .instrumentation import setup_instrumentation

parser = argparse.ArgumentParser(
    description="Run experiments with a specified model as provider:model"
)
parser.add_argument(
    "-m", "--model", required=True, 
    help="The model to use for the experiment."
)
parser.add_argument(
    "-u", "--base-url",
    default="http://bacook.local:11434/v1",
    help="The base URL for the model API.",
)
args = parser.parse_args()

print(f">>> model: {args.model}, base_url: {args.base_url}", file=sys.stderr)

config = create_config(
    llm=local(model=args.model, base_url=args.base_url),
)

langfuse: Final[Langfuse] = setup_instrumentation()


def serialize_config(cfg):
    """Serialize config object, converting type objects to their string representation."""

    def default(obj):
        if isinstance(obj, type):
            return (
                f"{obj.__module__}.{obj.__name__}"
                if hasattr(obj, "__module__")
                else obj.__name__
            )
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(cfg.model_dump(), indent=2, default=default)


INSTRUCTIONS: Final[str] = dedent("""
    You are an advanced AI assistant with access to various tools.
    Provide accurate, helpful, and concise responses.
    Must follow instructions precisely.

    {instructions}
    """)


async def _task_executor(cfg: Config, item, **kwargs):
    question = item["input"]
    input_type: type = item["metadata"].get(
        "input_type",
    )
    output_type: type = item["metadata"].get(
        "output_type",
    )
    instructions: str = item["metadata"].get("instructions", "")
    task_cfg = cfg.model_copy(deep=True)
    task_cfg.agent.input_type = input_type
    task_cfg.agent.output_type = output_type
    task_cfg.agent.instructions = INSTRUCTIONS.format(instructions=instructions)
    agent = AgentASK[input_type, output_type].create_from_config(task_cfg)
    result = await agent.run(question)
    # print(f"Question: {question}", file=sys.stderr)
    # print(f"Answer: {result}", file=sys.stderr)
    return str(result)


async def task_tools(cfg: Config, item, **kwargs):
    return await _task_executor(cfg, item, **kwargs)


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


class OutputArray(BaseModel):
    name: str = Field(..., description="Name of the tool")


inputs_tools = [
    LocalExperimentItem(
        input="list of the all tools.",
        expected_output="JSON array of tool names.",
        metadata={
            "input_type": str,
            "output_type": OutputArray,
            "instructions": """
                Your task is to return a JSON array containing the names of the tools.
                Example of the output:
                [
                    {"name": "tool_name1"},
                    {"name": "tool_name2"},
                    ...
                ]
            """,
        },
    ),
    LocalExperimentItem(
        input="list of the all tools.",
        expected_output="JSON array of tool names.",
        metadata={
            "input_type": str,
            "output_type": str,
            "instructions": """
                Your task is to return a JSON array containing the names of the tools.
                Example of the output:
                [
                    {"name": "tool_name1"},
                    {"name": "tool_name2"},
                    ...
                ]
            """,
        },
    ),
]

result_tools = langfuse.run_experiment(
    name="test tools listing",
    description="testing tools listing functionality",
    data=inputs_tools,
    task=lambda *, item, **kwargs: task_tools(config, item=item, **kwargs),
    evaluators=[accuracy_evaluator],
    metadata={
        "config": serialize_config(config),
    },
)

# print(codecs.decode(result_tools.format(include_item_results=True), 'unicode_escape'), file=sys.stderr)
print(result_tools.format(), file=sys.stderr)
