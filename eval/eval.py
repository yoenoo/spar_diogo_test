import argparse

from .evaluators.lm_eval import evaluate_lm_eval
from .utils import (
    clear_gpu_memory,
    load_lm_eval_config,
    load_model_and_tokenizer,
    load_model_config,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models on various benchmarks"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Model key from configs/models.yaml"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["tinyMMLU"],
        help="Space-separated list of tasks to evaluate on.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--num_fewshot", type=int, default=0, help="Number of few-shot examples"
    )

    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--log_samples", action="store_true", help="Log samples to output directory"
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face API token for accessing gated models",
        default=None,
    )

    return parser.parse_args()


@clear_gpu_memory
def evaluate_model(model, tokenizer, tasks, batch_size=4, num_fewshot=0, output_dir="./output", log_samples=True, config=None):
    """Evaluate model on specified tasks.

    This function serves as a router to different evaluation methods based on task type.
    Each evaluation method should handle its own task validation and configuration.
    """
    results = {}

    # LM Evaluation Harness tasks
    lm_eval_tasks = tasks  # TODO: Filter tasks by type when we add more evaluators
    if lm_eval_tasks:

        lm_eval_results = evaluate_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=lm_eval_tasks,
            task_config=load_lm_eval_config(),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            output_dir=output_dir,
            log_samples=log_samples,
            config=config,
        )
        results.update(lm_eval_results)

    # Add other evaluation types here
    # Example:
    # custom_tasks = [task for task in tasks if task in CUSTOM_TASK_TYPES]
    # if custom_tasks:
    #     custom_results = evaluate_custom(...)
    #     results.update(custom_results)

    return results


def main():
    args = parse_args()

    # Load model configuration
    config = load_model_config(args.model)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, args.hf_token)

    # Run evaluations
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks=args.tasks,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        output_dir=args.output_dir,
        log_samples=args.log_samples,
        config=config,
    )

    print(f"\nResults for model {args.model} on tasks {args.tasks}:")
    print(results)


if __name__ == "__main__":
    main()
