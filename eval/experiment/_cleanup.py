import importlib
from typing import Final

from langfuse import Langfuse
from langfuse.api.resources.datasets.types.paginated_dataset_runs import (
    PaginatedDatasetRuns,
)

from eval.instrumentation import setup_instrumentation

langfuse: Final[Langfuse] = setup_instrumentation()

# List of experiment modules to clean up
EXPERIMENT_MODULES = [
    "eval.experiment.simple",
    "eval.experiment.list_of_tools_max",
    "eval.experiment.review",
    "eval.experiment.list_of_tools",
    "eval.experiment.simple_math",
]


def get_datasets():
    datasets = []
    for module_name in EXPERIMENT_MODULES:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "DATASET"):
                datasets.append(module.DATASET)
        except Exception as e:
            print(f"Error importing {module_name}: {e}")
    return datasets


DATASETS = get_datasets()


def cleanup_experiments():
    for dataset_name in DATASETS:
        try:
            runs_response: PaginatedDatasetRuns = langfuse.api.datasets.get_runs(
                dataset_name=dataset_name, limit=100
            )
            runs = runs_response.data
            if not runs:
                print(f"No runs found for dataset {dataset_name}")
                continue

            for run in runs:
                run_name = run.name
                try:
                    langfuse.api.datasets.delete_run(
                        dataset_name=dataset_name, run_name=run_name
                    )
                except Exception as e:
                    print(f"Error deleting run {run_name}: {e}")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")


if __name__ == "__main__":
    cleanup_experiments()
