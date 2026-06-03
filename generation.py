import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime_defaults import DEFAULT_BASE_MODEL


def build_messages(instruction: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]


def _model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("模型没有可用参数，无法确定输入设备") from exc


def load_model_and_tokenizer(base_model: str, checkpoint: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token_id，不能安全生成")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, model_id=checkpoint)
    model.eval()
    return model, tokenizer


def predict_response(
    messages: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> str:
    responses = predict_responses(
        messages_batch=[messages],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    if len(responses) != 1:
        raise ValueError(f"单条生成返回数量异常：{len(responses)}")
    return responses[0]


def predict_responses(
    messages_batch: list[list[dict[str, str]]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> list[str]:
    if not messages_batch:
        raise ValueError("messages_batch 不能为空")

    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_padding_side

    device = _model_input_device(model)
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    input_token_count = model_inputs["input_ids"].shape[1]

    # 显式传入 attention_mask 和 pad_token_id，避免生成行为依赖 transformers 的隐式推断。
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # batch 输入已经 pad 到共同长度，统一从共同输入长度后截取新增 token。
    generated_ids = generated_ids[:, input_token_count:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
    return data
