from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datetime import datetime
from pathlib import Path
import json


def evaluate_lm_eval(model, tokenizer, tasks, task_config, batch_size=4, num_fewshot=0, log_samples=False, output_dir=None, config=None):
    """Run evaluations using lm-eval-harness."""
    lm_model = HFLM(model, tokenizer=tokenizer, batch_size=batch_size)
    eval_results = {}

    if config is not None:
        model_name = config["name"].replace("/", "__")
    else:
        raise ValueError("Config is required (for now)")

    today = datetime.now()

    for task in tasks:
        task_results = simple_evaluate(
            model=lm_model, tasks=[task], num_fewshot=num_fewshot, log_samples=log_samples,
        )

        # Follow the metric path to get the result
        metric_path = task_config["tasks"][task]["key_metric_path"]
        result = task_results
        for key in metric_path:
            result = result[key]

        eval_results[task] = result

        if log_samples and output_dir is not None:
            save_dir = Path(output_dir) / model_name 
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"samples_{task}_{today}.jsonl", "w") as f:
                for entry in task_results["samples"][task]:
                    record = json.dumps(entry)
                    f.write(record + '\n')

    # TODO
    # with open(Path(output_dir) / model_name / f"results_{today}.json", "w") as f:
    #     json.dump(eval_results, f)

    return eval_results
