import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_utils import extract_final_answer


DEFAULT_BASE_MODEL = "./Qwen/Qwen2.5-0.5B-Instruct/"
DEFAULT_CHECKPOINT = "./output/Qwen/checkpoint-3750/"
DEFAULT_TEST_FILE = "data/raw_data/test.json"


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
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    device = _model_input_device(model)
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    # 显式传入 attention_mask 和 pad_token_id，避免生成行为依赖 transformers 的隐式推断。
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
    return data


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output:
        output_name = args.output
    elif args.experiment_id:
        output_name = f"{args.experiment_id}.csv"
    else:
        output_name = "submit.csv"

    output_path = Path(output_name)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = Path("submit") / output_path

    if args.raw_output:
        raw_path = Path(args.raw_output)
    else:
        raw_stem = args.experiment_id if args.experiment_id else output_path.stem
        raw_path = Path("submit") / "raw" / f"{raw_stem}.jsonl"
    if not raw_path.is_absolute() and raw_path.parent == Path("."):
        raw_path = Path("submit") / "raw" / raw_path
    return output_path, raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="基础模型路径")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="LoRA checkpoint 路径")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="测试集 JSON 路径")
    parser.add_argument("--output", default=None, help="输出 CSV 文件名或路径")
    parser.add_argument("--raw-output", default=None, help="原始预测 JSONL 文件名或路径")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="最大生成 token 数")
    parser.add_argument("--experiment-id", default=None, help="实验编号，用于自动命名输出文件")
    parser.add_argument("--no-postprocess", action="store_true", help="关闭答案后处理")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须为正整数")

    output_path, raw_path = _resolve_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    test_data = load_json_rows(Path(args.test_file))
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.checkpoint)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file, raw_path.open(
        "w", encoding="utf-8"
    ) as raw_file:
        writer = csv.writer(csv_file)
        for index, row in enumerate(tqdm(test_data)):
            for key in ("id", "instruction", "question"):
                if key not in row:
                    raise KeyError(f"测试集第 {index} 条样本缺少字段：{key}")

            messages = build_messages(str(row["instruction"]), str(row["question"]))
            raw_response = predict_response(
                messages=messages,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            ).replace("\n", " ")

            if args.no_postprocess:
                final_answer = raw_response.strip()
                answer_type = "unknown"
                extract_status = "disabled"
            else:
                extracted = extract_final_answer(raw_response)
                final_answer = extracted.answer
                answer_type = extracted.answer_type
                extract_status = extracted.status

            writer.writerow([row["id"], final_answer])
            raw_file.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "raw_response": raw_response,
                        "answer": final_answer,
                        "answer_type": answer_type,
                        "extract_status": extract_status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
