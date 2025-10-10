import argparse
import importlib
import sys
from typing import Final

parser = argparse.ArgumentParser(
    description="Run experiments with a specified model as provider:model"
)
parser.add_argument(
    "-m", "--model", required=True, help="The model to use for the experiment."
)
parser.add_argument(
    "-u",
    "--base-url",
    help="The base URL for the model API.",
)
parser.add_argument(
    "-p",
    "--packages",
    nargs="+",
    required=True,
    help="List of experiment packages to run.",
)
parser.add_argument(
    "-s",
    "--session-id",
    default="test_session",
    help="The session ID for the experiment.",
)
args = parser.parse_args()

SESSION_ID: Final[str] = args.session_id
MODEL: Final[str] = args.model
BASE_URL: Final[str] = args.base_url
PACKAGES: Final[list[str]] = args.packages

print(
    f">>> model: {MODEL}, base_url: {BASE_URL}, session_id: {SESSION_ID}, packages: {PACKAGES}",
    file=sys.stderr,
)

for package_name in PACKAGES:
    print(f">>> experiment: {package_name}", file=sys.stderr)
    package = importlib.import_module(package_name)
    package.run_experiment(model=MODEL, base_url=BASE_URL, session_id=SESSION_ID)
