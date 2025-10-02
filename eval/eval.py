"""
Comprehensive LLM Model Evaluation System

This module provides a framework for testing and evaluating LLM models with:
- Multi-model testing and comparison
- Multiple iterations per test case
- Tool usage tracking
- Historical data persistence
- Analytics and reporting
"""

__all__ = [
    "TestCase",
    "TestIteration",
    "TestResult",
    "EvalRunner",
    "EvalAnalyzer",
    "create_test_config",
]

import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from pydantic_ai.messages import ToolCallPart

# Add eval directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from local instrumentation module
from instrumentation import (
    LLMCallData,
    extract_llm_call_data,
    setup_instrumentation_with_callback,
)

from ask.core.agent import AgentASK
from ask.core.config import Config, LLMConfig, load_config


@dataclass
class TestCase:
    """A single test case with input and expected behavior."""

    name: str
    prompts: list[str]
    description: str = ""
    expected_tools: list[str] | None = None  # Tools that should be called
    expected_output_contains: list[str] | None = None  # Keywords in output
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestIteration:
    """Results from a single test iteration."""

    iteration_id: int
    model_name: str
    prompt: str
    timestamp: str
    duration_seconds: float
    output: Any
    llm_calls: list[LLMCallData]
    tools_used: list[str]
    total_tokens: int
    input_tokens: int
    output_tokens: int
    num_llm_requests: int
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Aggregated results for a test case across multiple iterations."""

    test_case_name: str
    model_name: str
    config: dict[str, Any]
    iterations: list[TestIteration]
    timestamp: str

    # Aggregated metrics
    avg_duration: float = 0.0
    avg_tokens: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    success_rate: float = 0.0
    tools_consistency: dict[str, int] = field(default_factory=dict)

    def compute_aggregates(self):
        """Compute aggregate metrics from iterations."""
        if not self.iterations:
            return

        total = len(self.iterations)
        self.avg_duration = sum(i.duration_seconds for i in self.iterations) / total
        self.avg_tokens = sum(i.total_tokens for i in self.iterations) / total
        self.avg_input_tokens = sum(i.input_tokens for i in self.iterations) / total
        self.avg_output_tokens = sum(i.output_tokens for i in self.iterations) / total
        self.success_rate = sum(1 for i in self.iterations if i.success) / total * 100

        # Count tool usage across iterations
        tool_counts = defaultdict(int)
        for iteration in self.iterations:
            for tool in iteration.tools_used:
                tool_counts[tool] += 1
        self.tools_consistency = dict(tool_counts)


class EvalRunner:
    """Runner for executing evaluation test cases."""

    def __init__(self, storage_dir: str = "eval_results"):
        """
        Initialize the evaluation runner.

        Args:
            storage_dir: Directory to store evaluation results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.llm_call_buffer: list[LLMCallData] = []
        self._setup_telemetry()

    def _setup_telemetry(self):
        """Set up OpenTelemetry instrumentation to capture LLM calls."""

        def span_callback(span: ReadableSpan):
            """Callback to capture LLM call data from spans."""
            llm_data = extract_llm_call_data(span)
            if llm_data:
                self.llm_call_buffer.append(llm_data)

        setup_instrumentation_with_callback(span_callback, service_name="eval-runner")

    async def run_iteration(
        self,
        agent: AgentASK,
        test_case: TestCase,
        prompt: str,
        iteration_id: int,
        model_name: str,
    ) -> TestIteration:
        """
        Run a single test iteration.

        Args:
            agent: The agent to test
            test_case: The test case to run
            iteration_id: Iteration number
            model_name: Name of the model being tested

        Returns:
            TestIteration with results
        """
        self.llm_call_buffer.clear()
        start_time = time.time()
        error = None
        success = True
        output = None

        try:
            output = await agent.run(prompt)
        except Exception as e:
            error = str(e)
            success = False
            print(f"    ⚠️  Error: {error}", file=sys.stderr)

        duration = time.time() - start_time

        # Extract tool usage
        tools_used = []
        for llm_call in self.llm_call_buffer:
            if llm_call.parsed_output:
                for msg in llm_call.parsed_output:
                    for part in msg.parts:
                        if isinstance(part, ToolCallPart):
                            tools_used.append(part.tool_name)

        # Calculate token usage
        total_tokens = sum(
            (call.usage.get("total_tokens", 0) if call.usage else 0)
            for call in self.llm_call_buffer
        )
        input_tokens = sum(
            (call.usage.get("input_tokens", 0) if call.usage else 0)
            for call in self.llm_call_buffer
        )
        output_tokens = sum(
            (call.usage.get("output_tokens", 0) if call.usage else 0)
            for call in self.llm_call_buffer
        )

        # Validate expectations
        if test_case.expected_tools and success:
            expected_set = set(test_case.expected_tools)
            actual_set = set(tools_used)
            if not expected_set.issubset(actual_set):
                success = False
                error = f"Missing expected tools: {expected_set - actual_set}"

        if test_case.expected_output_contains and success and output:
            output_str = str(output).lower()
            missing_keywords = [
                kw
                for kw in test_case.expected_output_contains
                if kw.lower() not in output_str
            ]
            if missing_keywords:
                success = False
                error = f"Output missing keywords: {missing_keywords}"

        return TestIteration(
            iteration_id=iteration_id,
            model_name=model_name,
            prompt=prompt,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            output=output,
            llm_calls=self.llm_call_buffer.copy(),
            tools_used=tools_used,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_llm_requests=len(self.llm_call_buffer),
            success=success,
            error=error,
        )

    async def run_test(
        self,
        test_case: TestCase,
        config: Config,
        iterations: int = 3,
    ) -> TestResult:
        """
        Run a test case multiple times with a given configuration.

        Args:
            test_case: Test case to run
            config: Agent configuration
            iterations: Number of iterations to run

        Returns:
            TestResult with aggregated results
        """
        agent = AgentASK.create_from_config(config)
        model_name = config.llm.model

        iteration_results = []
        iteration_id = 0
        for i in range(iterations):
            for prompt in test_case.prompts:
                iteration_id += 1
                print(
                    f"  Running iteration {iteration_id} for prompt '{str(prompt)[:50]}...'...",
                    flush=True,
                )
                result = await self.run_iteration(
                    agent, test_case, prompt, iteration_id, model_name
                )
                iteration_results.append(result)
                # Small delay between iterations
                await asyncio.sleep(0.5)

        # Create test result
        test_result = TestResult(
            test_case_name=test_case.name,
            model_name=model_name,
            config=config.model_dump(),
            iterations=iteration_results,
            timestamp=datetime.now().isoformat(),
        )
        test_result.compute_aggregates()

        # Save result
        self._save_result(test_result)

        return test_result

    async def run_multi_model_test(
        self,
        test_case: TestCase,
        configs: list[Config],
        iterations: int = 3,
    ) -> list[TestResult]:
        """
        Run a test case across multiple model configurations.

        Args:
            test_case: Test case to run
            configs: List of configurations to test
            iterations: Number of iterations per configuration

        Returns:
            List of TestResult for each configuration
        """
        results = []
        for i, config in enumerate(configs):
            print(
                f"\nTesting model {i + 1}/{len(configs)}: {config.llm.model}",
                flush=True,
            )
            result = await self.run_test(test_case, config, iterations)
            results.append(result)

        return results

    def _save_result(self, result: TestResult):
        """Save a test result to disk."""
        # Create filename with timestamp
        filename = (
            f"{result.test_case_name}_{result.model_name}_{result.timestamp}.json"
        )
        filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        filepath = self.storage_dir / filename

        # Convert to dict and handle non-serializable objects
        data = self._serialize_result(result)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Saved results to {filepath}", flush=True)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, type):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return str(obj)
        else:
            return obj

    def _serialize_result(self, result: TestResult) -> dict:
        """Convert TestResult to JSON-serializable dict."""
        data = asdict(result)

        # Convert config to JSON-serializable format
        if "config" in data and isinstance(data["config"], dict):
            data["config"] = self._make_json_serializable(data["config"])

        # Clean up non-serializable objects in iterations
        for iteration in data["iterations"]:
            # Convert output to string
            iteration["output"] = str(iteration["output"])

            # Serialize LLM calls (simplified to avoid serialization issues)
            llm_calls_serialized = []
            for call in iteration["llm_calls"]:
                # After asdict(), call is already a dict - extract only serializable fields
                call_dict = {
                    "span_name": call.get("span_name")
                    if isinstance(call, dict)
                    else str(getattr(call, "span_name", "")),
                    "model": call.get("model")
                    if isinstance(call, dict)
                    else str(getattr(call, "model", "")),
                    "usage": call.get("usage")
                    if isinstance(call, dict)
                    else getattr(call, "usage", {}),
                    "duration_ms": call.get("duration_ms")
                    if isinstance(call, dict)
                    else getattr(call, "duration_ms", 0),
                    "status": call.get("status")
                    if isinstance(call, dict)
                    else str(getattr(call, "status", "")),
                    "timestamp": call.get("timestamp")
                    if isinstance(call, dict)
                    else str(getattr(call, "timestamp", "")),
                }
                # Skip parsed messages as they contain complex objects
                llm_calls_serialized.append(call_dict)

            iteration["llm_calls"] = llm_calls_serialized

        return data

    def load_results(
        self, test_case_name: str | None = None, model_name: str | None = None
    ) -> list[TestResult]:
        """
        Load saved test results from disk.

        Args:
            test_case_name: Filter by test case name (optional)
            model_name: Filter by model name (optional)

        Returns:
            List of TestResult objects
        """
        results = []
        for filepath in self.storage_dir.glob("*.json"):
            with open(filepath) as f:
                data = json.load(f)

            # Apply filters
            if test_case_name and data["test_case_name"] != test_case_name:
                continue
            if model_name and data["model_name"] != model_name:
                continue

            # Reconstruct TestResult (simplified, without full LLMCallData objects)
            result = TestResult(
                test_case_name=data["test_case_name"],
                model_name=data["model_name"],
                config=data["config"],
                iterations=[],  # We'll reconstruct simplified iterations
                timestamp=data["timestamp"],
                avg_duration=data["avg_duration"],
                avg_tokens=data["avg_tokens"],
                avg_input_tokens=data["avg_input_tokens"],
                avg_output_tokens=data["avg_output_tokens"],
                success_rate=data["success_rate"],
                tools_consistency=data["tools_consistency"],
            )
            results.append(result)

        return results


class EvalAnalyzer:
    """Analyzer for generating reports and comparisons from test results."""

    def __init__(self, storage_dir: str = "eval_results"):
        self.storage_dir = Path(storage_dir)

    def compare_models(
        self, results: list[TestResult], metric: str = "avg_tokens"
    ) -> dict:
        """
        Compare models based on a specific metric.

        Args:
            results: List of test results to compare
            metric: Metric to compare (e.g., 'avg_tokens', 'avg_duration', 'success_rate')

        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        for result in results:
            comparison[result.model_name] = {
                "metric_value": getattr(result, metric, None),
                "success_rate": result.success_rate,
                "iterations": len(result.iterations),
            }

        # Sort by metric
        sorted_models = sorted(
            comparison.items(), key=lambda x: x[1]["metric_value"] or float("inf")
        )

        return {
            "metric": metric,
            "rankings": sorted_models,
            "comparison": comparison,
        }

    def generate_report(self, results: list[TestResult]) -> str:
        """
        Generate a formatted report from test results.

        Args:
            results: List of test results

        Returns:
            Formatted report string
        """
        report = ["=" * 80, "LLM MODEL EVALUATION REPORT", "=" * 80, ""]

        for result in results:
            report.append(f"\nTest Case: {result.test_case_name}")
            report.append(f"Model: {result.model_name}")
            report.append(f"Iterations: {len(result.iterations)}")
            report.append(f"Timestamp: {result.timestamp}")
            report.append("-" * 40)
            report.append(f"Success Rate: {result.success_rate:.1f}%")
            report.append(f"Avg Duration: {result.avg_duration:.2f}s")
            report.append(f"Avg Total Tokens: {result.avg_tokens:.0f}")
            report.append(f"Avg Input Tokens: {result.avg_input_tokens:.0f}")
            report.append(f"Avg Output Tokens: {result.avg_output_tokens:.0f}")
            report.append(f"Tools Used: {', '.join(result.tools_consistency.keys())}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def export_to_csv(self, results: list[TestResult], output_path: str):
        """
        Export results to CSV for further analysis.

        Args:
            results: Test results to export
            output_path: Path to output CSV file
        """
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Test Case",
                    "Model",
                    "Timestamp",
                    "Iterations",
                    "Success Rate",
                    "Avg Duration (s)",
                    "Avg Tokens",
                    "Avg Input Tokens",
                    "Avg Output Tokens",
                    "Tools Used",
                ]
            )

            for result in results:
                writer.writerow(
                    [
                        result.test_case_name,
                        result.model_name,
                        result.timestamp,
                        len(result.iterations),
                        f"{result.success_rate:.1f}%",
                        f"{result.avg_duration:.2f}",
                        f"{result.avg_tokens:.0f}",
                        f"{result.avg_input_tokens:.0f}",
                        f"{result.avg_output_tokens:.0f}",
                        ", ".join(result.tools_consistency.keys()),
                    ]
                )

        print(f"Exported results to {output_path}")


# Helper function to create test configurations
def create_test_config(
    llm: LLMConfig,
) -> Config:
    def _here(filename: str) -> str:
        return str(Path(__file__).parent / filename)

    config = load_config([_here("test.yaml")])  # Load base config from file
    config.agent.name = f"test-agent-{llm.model.replace(':', '-')}"
    config.llm = llm
    return config
