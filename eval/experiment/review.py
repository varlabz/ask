import json
import sys
from textwrap import dedent
from typing import Final

from attr import dataclass
from langfuse import Evaluation, Langfuse

from ask.core.agent import AgentASK
from eval.agent import create_config, make_llm_config
from eval.data import serialize_config, task_executor_agent
from eval.instrumentation import setup_instrumentation

langfuse: Final[Langfuse] = setup_instrumentation()


@dataclass
class Metadata:
    instructions: str


DATASET: Final[str] = "review"
try:
    dataset = langfuse.get_dataset(name=DATASET)
except Exception:
    langfuse.create_dataset(
        name=DATASET,
        description="analyze product reviews",
        metadata=Metadata(
            instructions=dedent("""
                    Analyze a product review, and then based on your analysis give me the
                    corresponding rating (integer). The rating should be an integer between 1 and
                    5. 1 is the worst rating, and 5 is the best rating. A strongly dissatisfied
                    review that only mentions issues should have a rating of 1 (worst). A strongly
                    satisfied review that only mentions positives and upsides should have a rating of 5 (best).
                    Be opinionated. Use the full range of possible ratings (1 to 5).

                    Output format is JSON with two fields: "analysis" and "rating".
                    {
                        "analysis": "some text ...",
                        "rating": 0
                    }

                    Here are some examples of reviews and their corresponding analyses and
                    ratings:

                    Review: 'Stylish and functional. Not sure how it'll handle rugged outdoor
                    use, but it's perfect for urban exploring.'
                    {
                        "analysis": "The reviewer appreciates the product's style and basic
                            functionality. They express some uncertainty about its ruggedness but
                            overall find it suitable for their intended use, resulting in a positive,
                            but not top-tier rating.",
                        "rating": 4
                    }

                    Review: 'It's a solid backpack at a decent price. Does the job, but nothing
                    particularly amazing about it.'
                    {
                        "analysis": "This reflects an average opinion. The backpack is functional and
                            fulfills its essential purpose. However, the reviewer finds it unremarkable
                            and lacking any standout features deserving of higher praise.",
                        "rating": 3
                    }

                    Review: 'The waist belt broke on my first trip! Customer service was unresponsive too. Would not recommend.'
                    {
                        "analysis": "A serious product defect and poor customer service experience naturally warrants the lowest possible rating. The reviewer is extremely unsatisfied with both the product and the company.",
                        "rating": 1
                    }
                """),
        ),
    )
    input = [
        "Review: 'Absolutely love the fit! Distributes weight well and surprisingly comfortable even on all-day treks. Would recommend.'",
    ]
    for i in input:
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            input=i,
            expected_output=5,
        )

dataset = langfuse.get_dataset(name=DATASET)
if dataset.metadata and isinstance(dataset.metadata, dict):
    dataset.metadata = Metadata(**dataset.metadata)


def _accuracy_evaluator(*, input, output, expected_output):
    """Evaluator that checks if the expected answer is in the output"""
    try:
        obj = json.loads(output)  # validate JSON
        return Evaluation(
            name="accuracy",
            value=1
            if (len(obj["analysis"]) > 5 and obj["rating"] == expected_output)
            else 0,
        )
    except Exception as e:
        print(f"Output is not valid JSON {e}", file=sys.stderr)
        return Evaluation(name="accuracy", value=0.0)


def run_experiment(model: str, base_url: str, session_id: str):
    config = create_config(
        llm=make_llm_config(model=model, base_url=base_url),
        instructions=dataset.metadata.instructions if dataset.metadata else "",
    )
    agent = AgentASK.create_from_config(config=config)
    result = dataset.run_experiment(
        name="review",
        run_name=config.llm.model,
        description="testing review functionality",
        task=lambda *, item, **kwargs: task_executor_agent(
            agent=agent,
            item=item,
            callback=lambda: langfuse.update_current_trace(
                session_id=session_id, user_id="eval", tags=[config.llm.model]
            ),
        ),
        evaluators=[
            lambda *, input, output, expected_output, **kwargs: _accuracy_evaluator(
                input=input, output=output, expected_output=expected_output
            )
        ],
        max_concurrency=1,  # Limit concurrency to 1 to avoid race timing issues
        metadata={
            "config": serialize_config(config),
        },
    )
    print(result.format(), file=sys.stderr)
