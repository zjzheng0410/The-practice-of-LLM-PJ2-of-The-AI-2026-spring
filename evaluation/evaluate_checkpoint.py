import argparse
import json
from pathlib import Path
from typing import Any

from answer_utils import compare_answers, extract_final_answer
from evaluation.config import DEFAULT_BASE_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_NAME, DEFAULT_VALID_FILE
from evaluation.error_analysis import build_wrong_analysis
from evaluation.metrics import build_metrics
from generation import build_messages, load_json_rows, load_model_and_tokenizer, predict_response


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_checkpoint(
    base_model: str,
    checkpoint: str,
    valid_file: str,
    experiment_id: str,
    output_dir: str,
    max_new_tokens: int,
    split_name: str = DEFAULT_SPLIT_NAME,
    result_dir: Path | None = None,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须为正整数")

    valid_data = load_json_rows(Path(valid_file))
    if not valid_data:
        raise ValueError(f"{valid_file} 验证集为空")

    from tqdm import tqdm

    model, tokenizer = load_model_and_tokenizer(base_model, checkpoint)
    raw_records: list[dict[str, Any]] = []
    wrong_records: list[dict[str, Any]] = []
    for index, row in enumerate(tqdm(valid_data)):
        for key in ("id", "instruction", "question", "answer", "answer_type"):
            if key not in row:
                raise KeyError(f"验证集第 {index} 条样本缺少字段：{key}")

        messages = build_messages(str(row["instruction"]), str(row["question"]))
        raw_response = predict_response(
            messages=messages,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        ).replace("\n", " ")

        error = None
        try:
            extracted = extract_final_answer(raw_response)
            pred_answer = extracted.answer
            pred_answer_type = extracted.answer_type
            extract_status = extracted.status
            correct = compare_answers(pred_answer, row["answer"])
        except ValueError as exc:
            # 后处理失败属于评估结果，必须写入 wrong 记录，不能静默跳过。
            pred_answer = ""
            pred_answer_type = "unknown"
            extract_status = "error"
            correct = False
            error = str(exc)

        record = {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "answer_type": row["answer_type"],
            "raw_response": raw_response,
            "pred_answer": pred_answer,
            "pred_answer_type": pred_answer_type,
            "extract_status": extract_status,
            "correct": correct,
            "checkpoint": checkpoint,
            "valid_file": valid_file,
        }
        if error is not None:
            record["error"] = error
        raw_records.append(record)
        if not correct:
            wrong_records.append(record)

    target_dir = result_dir if result_dir is not None else Path(output_dir) / split_name / experiment_id
    metrics_path = target_dir / "metrics.json"
    raw_path = target_dir / "raw.jsonl"
    wrong_path = target_dir / "wrong.jsonl"
    wrong_analysis_path = target_dir / "wrong_analysis.json"
    wrong_analysis_jsonl_path = target_dir / "wrong_analysis.jsonl"

    wrong_analysis, wrong_analysis_rows = build_wrong_analysis(wrong_records)
    metrics = build_metrics(raw_records)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "checkpoint": checkpoint,
            "valid_file": valid_file,
            "metrics_output": str(metrics_path),
            "raw_output": str(raw_path),
            "wrong_output": str(wrong_path),
            "wrong_analysis_output": str(wrong_analysis_path),
            "wrong_analysis_jsonl_output": str(wrong_analysis_jsonl_path),
        }
    )

    _write_json(metrics_path, metrics)
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(wrong_path, wrong_records)
    _write_json(wrong_analysis_path, wrong_analysis)
    _write_jsonl(wrong_analysis_jsonl_path, wrong_analysis_rows)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="基础模型路径")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="验证集 JSON 路径")
    parser.add_argument("--experiment-id", required=True, help="实验编号")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="最大生成 token 数")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评估输出目录")
    parser.add_argument("--split-name", default=DEFAULT_SPLIT_NAME, help="验证集版本目录名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        valid_file=args.valid_file,
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        split_name=args.split_name,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
