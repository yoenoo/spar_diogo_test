from functools import wraps

import torch
import yaml


def clear_gpu_memory(func):
    """Decorator to clear GPU memory before and after function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        result = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return result

    return wrapper


def load_model_config(model_key):
    """Load model configuration from yaml file."""
    with open("eval/configs/models.yaml", "r") as f:
        configs = yaml.safe_load(f)

    if model_key not in configs:
        raise ValueError(f"Model {model_key} not found in configs")

    return configs[model_key]


def load_model_and_tokenizer(config, hf_token=None):
    """Load model and tokenizer from config."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        device_map=config["device_map"],
        torch_dtype=torch.bfloat16
        if config["precision"] == "bfloat16"
        else torch.float16,
        token=hf_token,
    )

    # Use different tokenizer if specified in config
    tokenizer_name = config.get("tokenizer", config["name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)

    # Ensure padding token is set for proper batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_lm_eval_config():
    """Load lm-eval task configurations from yaml file."""
    with open("eval/configs/lm_eval.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
