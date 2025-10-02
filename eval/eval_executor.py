import asyncio
from datetime import date

from dotenv import load_dotenv
from langfuse import get_client

from ask.core.config import LLMConfig

from .eval import EvalAnalyzer, EvalRunner, TestCase, create_test_config

load_dotenv()


async def run_test_with_langfuse(
    runner, test_case, config, iterations, langfuse_client, langfuse_dataset
):
    """
    Run test and save result to Langfuse dataset.
    """
    # Run your existing eval code
    result = await runner.run_test(test_case, config, iterations)

    # Save to Langfuse dataset
    for iteration in result.iterations:
        langfuse_client.create_dataset_item(
            dataset_name=langfuse_dataset.name,
            input={"prompt": iteration.prompt, "test_case": test_case.name},
            expected_output={
                "expected_contains": test_case.expected_output_contains or []
            },
            metadata={
                "model": iteration.model_name,
                "actual_output": str(iteration.output),
                "success": iteration.success,
                "error": iteration.error,
                "duration": iteration.duration_seconds,
                "tokens": iteration.total_tokens,
                "iteration_id": iteration.iteration_id,
                "timestamp": iteration.timestamp,
            },
        )

    return result


test_cases = [
    TestCase(
        name="simple_greeting",
        prompts=["Say hello and introduce yourself as a helpful AI assistant."],
        description="Tests basic response generation",
        expected_output_contains=["hello", "assistant"],
        metadata={"category": "basic", "difficulty": "easy"},
    ),
    TestCase(
        name="math_calculation",
        prompts=["What is 15 multiplied by 23? Show your calculation."],
        description="Tests mathematical reasoning",
        expected_output_contains=["345"],
        metadata={"category": "reasoning", "difficulty": "easy"},
    ),
    TestCase(
        name="list_generation",
        prompts=["List 3 primary colors.", "List 3 primary colors."],
        description="Tests knowledge recall",
        expected_output_contains=["red", "blue", "yellow"],
        metadata={"category": "knowledge", "difficulty": "easy"},
    ),
]


models = [
    "qwen3:1.7b-q4_K_M",
    "qwen3:4b-q4_K_M",
    "gpt-oss:20b",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
]


def ollama(name: str, base_url: str = "http://bacook.local:11434/v1") -> LLMConfig:
    return LLMConfig(model=f"ollama:{name}", base_url=base_url)


configs = [create_test_config(llm=ollama(m)) for m in models]


async def main():
    langfuse_client = get_client()
    runner = EvalRunner(storage_dir="tmp/eval_results")
    all_results = []
    for config in configs:
        model_name = config.llm.model
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        for test_case in test_cases:
            print(f"\nTest Case: {test_case.name}")
            print(f"{test_case.description}")
            langfuse_dataset = langfuse_client.create_dataset(
                name=test_case.name,
                description=test_case.description,
                metadata={"created_at": date.today(), "type": "eval_results"},
            )
            result = await run_test_with_langfuse(
                runner,
                test_case,
                config,
                iterations=3,
                langfuse_client=langfuse_client,
                langfuse_dataset=langfuse_dataset,
            )

            all_results.append(result)
            # if result.tools_consistency:
            #     print(f"  🔧 Tools Used: {', '.join(result.tools_consistency.keys())}")

    analyzer = EvalAnalyzer(storage_dir="tmp/eval_results")
    print("\n" + analyzer.generate_report(all_results))

    # # CSV export
    # csv_path = "tmp/eval_results/summary_with_langfuse.csv"
    # analyzer.export_to_csv(all_results, csv_path)
    # print(f"\n💾 CSV exported to: {csv_path}")

    # Model comparison
    if len(configs) > 1:
        comparison = analyzer.compare_models(all_results, metric="duration")
        print("\nModel Comparison (Avg Duration):")
        print("Rank | Model                                         | Duration (s)")
        print("-----|-----------------------------------------------|----------------")
        for rank, (model, data) in enumerate(comparison["rankings"], 1):
            print(f"{rank:4} | {model:45} | {data['metric_value']}")

    langfuse_client.flush()


if __name__ == "__main__":
    asyncio.run(main())
