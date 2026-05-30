import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from answer_utils import compare_answers, extract_final_answer
from infer import DEFAULT_BASE_MODEL, build_messages, load_json_rows, load_model_and_tokenizer, predict_response


LONG_QUESTION_LENGTH = 100
FRACTION_TYPES = {"fraction", "mixed_fraction"}
DEFAULT_VALID_FILE = "data/clean_data/valid_sft_v1.json"


def _safe_accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return correct / total


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_metrics(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(raw_records)
    correct = sum(1 for row in raw_records if row["correct"])

    by_type_total = Counter(row["answer_type"] for row in raw_records)
    by_type_correct = Counter(row["answer_type"] for row in raw_records if row["correct"])
    by_type = {
        answer_type: {
            "total": by_type_total[answer_type],
            "correct": by_type_correct[answer_type],
            "accuracy": _safe_accuracy(by_type_correct[answer_type], by_type_total[answer_type]),
        }
        for answer_type in sorted(by_type_total)
    }

    special_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_records:
        if len(str(row["question"])) > LONG_QUESTION_LENGTH:
            special_groups["long_question"].append(row)
        if row["answer_type"] in FRACTION_TYPES:
            special_groups["fraction"].append(row)
        if row["answer_type"] == "percent":
            special_groups["percent"].append(row)

    special_metrics = {}
    for name, rows in special_groups.items():
        group_correct = sum(1 for row in rows if row["correct"])
        special_metrics[name] = {
            "total": len(rows),
            "correct": group_correct,
            "accuracy": _safe_accuracy(group_correct, len(rows)),
        }

    postprocess_failure_count = sum(
        1 for row in raw_records if row["extract_status"] not in {"matched", "disabled"}
    )
    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_accuracy(correct, total),
        "by_answer_type": by_type,
        "postprocess_failure_count": postprocess_failure_count,
        "postprocess_failure_rate": _safe_accuracy(postprocess_failure_count, total),
        "special_metrics": special_metrics,
        "long_question_threshold": LONG_QUESTION_LENGTH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="基础模型路径")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="验证集 JSON 路径")
    parser.add_argument("--experiment-id", required=True, help="实验编号")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="最大生成 token 数")
    parser.add_argument("--output-dir", default="eval", help="评估输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须为正整数")

    valid_data = load_json_rows(Path(args.valid_file))
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.checkpoint)

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
            max_new_tokens=args.max_new_tokens,
        ).replace("\n", " ")

        error = None
        try:
            extracted = extract_final_answer(raw_response)
            pred_answer = extracted.answer
            pred_answer_type = extracted.answer_type
            extract_status = extracted.status
            correct = compare_answers(pred_answer, row["answer"])
        except ValueError as exc:
            # 后处理无法抽取属于评估对象，必须写入记录，不能静默忽略。
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
        }
        if error is not None:
            record["error"] = error
        raw_records.append(record)
        if not correct:
            wrong_records.append(record)

    output_dir = Path(args.output_dir) / args.experiment_id
    metrics_path = output_dir / "metrics.json"
    raw_path = output_dir / "raw.jsonl"
    wrong_path = output_dir / "wrong.jsonl"

    metrics = _build_metrics(raw_records)
    metrics.update(
        {
            "experiment_id": args.experiment_id,
            "valid_file": args.valid_file,
            "checkpoint": args.checkpoint,
            "raw_output": str(raw_path),
            "wrong_output": str(wrong_path),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(wrong_path, wrong_records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
