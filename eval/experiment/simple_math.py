import sys
from textwrap import dedent
from typing import Final

from attr import dataclass
from langfuse import Evaluation, Langfuse

from ask.core.config import Config
from eval.agent import create_config, local
from eval.data import serialize_config, task_executor
from eval.instrumentation import setup_instrumentation

langfuse: Final[Langfuse] = setup_instrumentation()


@dataclass
class Metadata:
    instructions: str


DATASET: Final[str] = "simple math"
try:
    dataset = langfuse.get_dataset(name=DATASET)
except Exception:
    langfuse.create_dataset(
        name=DATASET,
        description="simple math. no tools",
        metadata=Metadata(
            instructions=dedent(
                """
                    Must return a one line answer. No other text.
                    Example of the output:
                    3
                """,
            ),
        ),
    )
    input = [
        {"question": "1+2", "answer": "3"},
        {"question": "5-3", "answer": "2"},
        {"question": "4×2", "answer": "8"},
        {"question": "4*2", "answer": "8"},
        {"question": "10÷2", "answer": "5"},
        {"question": "10/2", "answer": "5"},
        {"question": "3+4*2", "answer": "11"},
        {"question": "15-5×2", "answer": "5"},
        {"question": "18/3+2", "answer": "8"},
        {"question": "2*3+4*5", "answer": "26"},
        {"question": "20÷4-2×3", "answer": "-1"},
        {"question": "5+3×2-4", "answer": "7"},
        {"question": "12÷3+6÷2", "answer": "7"},
        {"question": "4×5-3×2", "answer": "14"},
        {"question": "2+3×4-5÷5", "answer": "13"},
        {"question": "10-2×3+4÷2", "answer": "6"},
        {"question": "3×4+5×2-6", "answer": "16"},
        {"question": "20÷4×3-5", "answer": "10"},
        {"question": "2×3+4×5-6÷2", "answer": "23"},
        {"question": "5+10÷2×3-4", "answer": "16"},
        {"question": "3×4-2×5+10÷2", "answer": "7"},
        {"question": "2*(3+4)-5/5", "answer": "13"},
    ]
    for i in input:
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            input=i["question"],
            expected_output=i["answer"],
        )

dataset = langfuse.get_dataset(name=DATASET)
if dataset.metadata and isinstance(dataset.metadata, dict):
    dataset.metadata = Metadata(**dataset.metadata)


def _accuracy_evaluator(
    *,
    input,
    output,
    expected_output,
):
    try:
        return Evaluation(
            name="accuracy", value=1 if output.strip() == expected_output else 0
        )
    except Exception as e:
        print(f"Output is not valid JSON {e}", file=sys.stderr)
        return Evaluation(name="accuracy", value=0.0)


def run_experiment(model: str, base_url: str, session_id: str):
    config: Final[Config] = create_config(
        llm=local(model=model, base_url=base_url),
        mcp={},
        instructions=dataset.metadata.instructions if dataset.metadata else "",
    )
    result = dataset.run_experiment(
        name="simple math",
        description="simple math. 1 line answer",
        task=lambda *, item, **kwargs: task_executor(
            config,
            item=item,
            callback=lambda: langfuse.update_current_trace(
                session_id=session_id, user_id="eval", tags=[config.llm.model]
            ),
        ),
        evaluators=[
            lambda *, input, output, expected_output, **kwargs: _accuracy_evaluator(
                input=input,
                output=output,
                expected_output=expected_output,
            )
        ],
        max_concurrency=1,  # Limit concurrency to 1 to avoid race timing issues
        metadata={
            "config": serialize_config(config),
        },
    )
    print(result.format(), file=sys.stderr)
