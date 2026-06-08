import argparse
import json
from pathlib import Path
from typing import Any

from runtime_defaults import DEFAULT_BASE_MODEL


DEFAULT_TRAIN_FILE = "data/clean_data/train_dpo_cot001_v1.json"
DEFAULT_POLICY_CHECKPOINT = "output/cot-001/checkpoint-2000"
DEFAULT_OUTPUT_DIR = "output/dpo-cot001"
EXPECTED_TRL_VERSION = "1.5.1"
REQUIRED_DPO_CONFIG_PARAMETERS = (
    "precompute_ref_log_probs",
    "precompute_ref_batch_size",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_length",
)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    if not data:
        raise ValueError(f"{path} 数据为空")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise KeyError(f"{path} 第 {index} 条样本缺少字段：{key}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--policy-checkpoint", default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--precompute-ref-log-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--precompute-ref-batch-size", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数")


def require_positive_float(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} 必须为正数")


def require_non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} 必须为非负整数")


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.policy_checkpoint).is_dir():
        raise NotADirectoryError(f"--policy-checkpoint 不存在：{args.policy_checkpoint}")
    if not Path(args.train_file).is_file():
        raise FileNotFoundError(f"--train-file 不存在：{args.train_file}")
    require_positive_float(args.epochs, "--epochs")
    require_positive_float(args.learning_rate, "--learning-rate")
    require_positive_float(args.beta, "--beta")
    require_positive_int(args.max_length, "--max-length")
    require_positive_int(args.save_steps, "--save-steps")
    require_positive_int(args.per_device_train_batch_size, "--per-device-train-batch-size")
    require_positive_int(args.gradient_accumulation_steps, "--gradient-accumulation-steps")
    require_positive_int(args.precompute_ref_batch_size, "--precompute-ref-batch-size")
    require_non_negative_int(args.dataloader_num_workers, "--dataloader-num-workers")


def validate_dpo_runtime_capabilities() -> None:
    import inspect
    import trl
    from trl import DPOConfig

    if trl.__version__ != EXPECTED_TRL_VERSION:
        raise RuntimeError(f"TRL 版本必须为 {EXPECTED_TRL_VERSION}，当前为 {trl.__version__}")

    signature = inspect.signature(DPOConfig.__init__)
    missing = [
        parameter
        for parameter in REQUIRED_DPO_CONFIG_PARAMETERS
        if parameter not in signature.parameters
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"DPOConfig 缺少必需参数：{joined}")


def load_tokenizer(base_model: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token_id，不能安全训练")
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(base_model: str, checkpoint: str, gradient_checkpointing: bool) -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.config.use_cache = False
    if gradient_checkpointing:
        # PEFT + gradient checkpointing 需要输入梯度，否则反向传播会缺失可训练路径。
        model.enable_input_require_grads()
    return model


def load_reference_model(base_model: str, checkpoint: str) -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_length": args.max_length,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "logging_steps": 10,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
        "precompute_ref_batch_size": args.precompute_ref_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "seed": args.seed,
        "data_seed": args.seed,
        "disable_dropout": True,
        "auto_find_batch_size": False,
        "bf16": True,
        "report_to": "none",
    }


def build_training_args(args: argparse.Namespace) -> Any:
    validate_dpo_runtime_capabilities()

    from trl import DPOConfig

    return DPOConfig(**build_training_kwargs(args))


def main() -> None:
    args = parse_args()
    validate_args(args)

    from datasets import Dataset
    from trl import DPOTrainer

    training_args = build_training_args(args)
    tokenizer = load_tokenizer(args.base_model)
    train_dataset = Dataset.from_list(load_json_list(Path(args.train_file)))
    policy_model = load_policy_model(args.base_model, args.policy_checkpoint, args.gradient_checkpointing)
    reference_model = load_reference_model(args.base_model, args.policy_checkpoint)

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
