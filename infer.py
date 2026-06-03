import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from answer_utils import extract_final_answer
from generation import (
    DEFAULT_BASE_MODEL,
    build_messages,
    load_json_rows,
    load_model_and_tokenizer,
    predict_responses,
)
from prompting.profiles import get_prompt_profile, resolve_max_new_tokens, select_instruction


DEFAULT_CHECKPOINT = "./output/Qwen/checkpoint-3750/"
DEFAULT_TEST_FILE = "data/raw_data/test.json"
DEFAULT_BATCH_SIZE = 8


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


def _iter_batches(
    rows: list[dict[str, Any]],
    batch_size: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    # 这里只做连续切分，字段校验、prompt 构造和写文件都留在主流程中。
    for start_index in range(0, len(rows), batch_size):
        yield start_index, rows[start_index : start_index + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="基础模型路径")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="LoRA checkpoint 路径")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="测试集 JSON 路径")
    parser.add_argument("--output", default=None, help="输出 CSV 文件名或路径")
    parser.add_argument("--raw-output", default=None, help="原始预测 JSONL 文件名或路径")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument("--prompt-profile", default="direct", help="提示词 profile：direct 或 cot")
    parser.add_argument("--experiment-id", default=None, help="实验编号，用于自动命名输出文件")
    parser.add_argument("--no-postprocess", action="store_true", help="关闭答案后处理")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="单进程推理 batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须为正整数")

    profile = get_prompt_profile(args.prompt_profile)
    max_new_tokens = resolve_max_new_tokens(profile, args.max_new_tokens)

    output_path, raw_path = _resolve_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    test_data = load_json_rows(Path(args.test_file))
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.checkpoint)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file, raw_path.open(
        "w", encoding="utf-8"
    ) as raw_file:
        writer = csv.writer(csv_file)
        with tqdm(total=len(test_data)) as progress:
            for batch_start, batch_rows in _iter_batches(test_data, args.batch_size):
                messages_batch = []
                for offset, row in enumerate(batch_rows):
                    index = batch_start + offset
                    for key in ("id", "instruction", "question"):
                        if key not in row:
                            raise KeyError(f"测试集第 {index} 条样本缺少字段：{key}")

                    instruction = select_instruction(profile, str(row["instruction"]))
                    messages_batch.append(build_messages(instruction, str(row["question"])))

                raw_responses = predict_responses(
                    messages_batch=messages_batch,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                )
                if len(raw_responses) != len(batch_rows):
                    raise ValueError(f"batch 生成返回数量异常：{len(raw_responses)} != {len(batch_rows)}")

                for row, raw_response in zip(batch_rows, raw_responses):
                    raw_response = raw_response.replace("\n", " ")
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
                                "prompt_profile": profile.name,
                                "answer_marker": profile.answer_marker,
                                "max_new_tokens": max_new_tokens,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                csv_file.flush()
                raw_file.flush()
                progress.update(len(batch_rows))


if __name__ == "__main__":
    main()
