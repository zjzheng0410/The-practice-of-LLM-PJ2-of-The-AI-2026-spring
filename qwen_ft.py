import argparse
import json
from pathlib import Path
from typing import Any

import swanlab
import torch
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.huggingface import SwanLabCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


DEFAULT_BASE_MODEL = "./Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "./output/Qwen"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
    return data


def process_func(example: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    for key in ("instruction", "question", "answer"):
        if key not in example:
            raise KeyError(f"训练样本缺少字段：{key}")

    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['answer']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    # 截断后必须仍保留至少一个答案 token，否则该样本会变成无监督信号。
    if not any(label != -100 for label in labels):
        sample_id = example.get("id", "unknown")
        raise ValueError(f"样本 {sample_id} 在 max_length={max_length} 下没有答案 token")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dataset(path: Path, tokenizer: Any, max_length: int) -> list[dict[str, list[int]]]:
    rows = load_json_list(path)
    return [process_func(row, tokenizer, max_length=max_length) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="基础模型路径")
    parser.add_argument("--train-file", default="train.json", help="训练集 JSON 路径")
    parser.add_argument("--valid-file", default=None, help="验证集 JSON 路径；不传则保持 baseline 训练行为")
    parser.add_argument("--output-dir", default=None, help="checkpoint 输出目录")
    parser.add_argument("--experiment-id", default=None, help="实验编号；未指定 output-dir 时用于生成输出目录")
    parser.add_argument("--epochs", type=float, default=5, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max-length", type=int, default=384, help="最大序列长度")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--save-steps", type=int, default=1000, help="checkpoint 保存步数")
    return parser.parse_args()


def _resolve_output_dir(output_dir: str | None, experiment_id: str | None) -> str:
    if output_dir:
        return output_dir
    if experiment_id:
        return str(Path("output") / experiment_id)
    return DEFAULT_OUTPUT_DIR


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length 必须为正整数")
    if args.lora_r <= 0:
        raise ValueError("--lora-r 必须为正整数")
    if args.save_steps <= 0:
        raise ValueError("--save-steps 必须为正整数")

    output_dir = _resolve_output_dir(args.output_dir, args.experiment_id)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token_id，不能安全训练")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()  # 开启梯度检查点时必须保留输入梯度。

    train_dataset = build_dataset(Path(args.train_file), tokenizer, max_length=args.max_length)
    eval_dataset = None
    if args.valid_file:
        eval_dataset = build_dataset(Path(args.valid_file), tokenizer, max_length=args.max_length)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)

    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "logging_steps": 10,
        "num_train_epochs": args.epochs,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "save_on_each_node": True,
        "gradient_checkpointing": True,
        "report_to": "none",
    }
    if eval_dataset is not None:
        training_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": args.save_steps,
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )

    training_args = TrainingArguments(**training_kwargs)

    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-0.5B-fintune",
        experiment_name=args.experiment_id or "Qwen/Qwen2.5-0.5B-Instruct",
        config={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "output_dir": output_dir,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
        },
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            processing_class=tokenizer,
            callbacks=[swanlab_callback],
        )
        trainer.train()
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
