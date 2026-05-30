import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from answer_utils import classify_answer, normalize_answer


DEFAULT_SEED = 20260530
DEFAULT_VALID_SIZE = 1000
DEFAULT_VERSION = "v1"


def clean_question(value: Any) -> tuple[str, bool, int]:
    if isinstance(value, str):
        return _clean_question_part(value), False, 0
    if isinstance(value, list):
        if not all(isinstance(part, str) for part in value):
            raise TypeError("question 列表中存在非字符串元素")
        parts = [_clean_question_part(part, allow_empty=True) for part in value]
        parts = [part for part in parts if part]
        if not parts:
            raise ValueError("question 列表清洗后为空")
        return "，".join(parts), True, len(value) - len(parts)
    raise TypeError(f"question 类型不支持：{type(value).__name__}")


def _clean_question_part(text: str, allow_empty: bool = False) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = text.strip('"').strip("'").strip("“").strip("”").strip("‘").strip("’")
    text = " ".join(text.split())
    if not text:
        if allow_empty:
            return ""
        raise ValueError("question 清洗后为空")
    return text


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
    return data


def _build_output_paths(output_dir: Path, version: str) -> dict[str, Path]:
    return {
        "train_clean": output_dir / f"train_clean_{version}.json",
        "train_sft": output_dir / f"train_sft_{version}.json",
        "valid_sft": output_dir / f"valid_sft_{version}.json",
        "clean_report": output_dir / f"clean_report_{version}.json",
    }


def _ensure_can_write(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出文件已存在，请使用 --overwrite：" + ", ".join(existing))


def _duplicate_id_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows if "id" in row]
    counts = Counter(ids)
    duplicate_ids = [id_value for id_value, count in counts.items() if count > 1]
    return {
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_extra_rows": sum(counts[id_value] - 1 for id_value in duplicate_ids),
        "duplicate_ids_sample": duplicate_ids[:50],
    }


def clean_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned_candidates: list[dict[str, Any]] = []
    dropped_empty_answer: list[dict[str, Any]] = []
    fixed_question_list_count = 0
    skipped_empty_question_part_count = 0

    for index, row in enumerate(raw_rows):
        for key in ("id", "question", "instruction", "answer"):
            if key not in row:
                raise KeyError(f"第 {index} 条样本缺少字段：{key}")

        raw_answer = row["answer"]
        if raw_answer is None or str(raw_answer).strip() == "":
            dropped_empty_answer.append({"index": index, "id": row["id"], "reason": "empty_answer"})
            continue

        question, fixed_question_list, skipped_empty_parts = clean_question(row["question"])
        fixed_question_list_count += int(fixed_question_list)
        skipped_empty_question_part_count += skipped_empty_parts
        answer = normalize_answer(raw_answer)
        cleaned_candidates.append(
            {
                "id": str(row["id"]),
                "question": question,
                "answer": answer,
                "instruction": str(row["instruction"]).strip(),
                "answer_type": classify_answer(answer),
            }
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cleaned_candidates:
        groups[(row["question"], row["instruction"])].append(row)

    conflict_groups = []
    conflict_keys: set[tuple[str, str]] = set()
    for key, rows in groups.items():
        answers = sorted({row["answer"] for row in rows})
        if len(answers) > 1:
            conflict_keys.add(key)
            conflict_groups.append(
                {
                    "question": key[0],
                    "instruction": key[1],
                    "answers": answers,
                    "ids": [row["id"] for row in rows],
                    "row_count": len(rows),
                }
            )

    # 同题不同答案没有可靠依据判定哪条正确，整组丢弃，避免把冲突监督信号写入 SFT。
    cleaned_rows = [
        row for row in cleaned_candidates if (row["question"], row["instruction"]) not in conflict_keys
    ]

    report = {
        "original_count": len(raw_rows),
        "candidate_count": len(cleaned_candidates),
        "cleaned_count": len(cleaned_rows),
        "dropped_empty_answer_count": len(dropped_empty_answer),
        "dropped_empty_answer_sample": dropped_empty_answer[:50],
        "fixed_question_list_count": fixed_question_list_count,
        "skipped_empty_question_part_count": skipped_empty_question_part_count,
        "conflict_group_count": len(conflict_groups),
        "conflict_drop_count": len(cleaned_candidates) - len(cleaned_rows),
        "conflict_groups_sample": conflict_groups[:100],
        "answer_type_distribution": dict(Counter(row["answer_type"] for row in cleaned_rows)),
    }
    report.update(_duplicate_id_report(raw_rows))
    return cleaned_rows, report


def split_train_valid(
    rows: list[dict[str, Any]], valid_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if valid_size < 0:
        raise ValueError("valid_size 不能为负数")
    if valid_size > len(rows):
        raise ValueError(f"valid_size={valid_size} 大于清洗后样本数={len(rows)}")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    valid_rows = shuffled[:valid_size]
    train_rows = shuffled[valid_size:]
    return train_rows, valid_rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_prepare_data(
    input_file: Path,
    output_dir: Path,
    version: str = DEFAULT_VERSION,
    valid_size: int = DEFAULT_VALID_SIZE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_paths = _build_output_paths(output_dir, version)
    _ensure_can_write(output_paths, overwrite)

    raw_rows = load_json_list(input_file)
    cleaned_rows, report = clean_rows(raw_rows)
    train_rows, valid_rows = split_train_valid(cleaned_rows, valid_size=valid_size, seed=seed)

    report.update(
        {
            "version": version,
            "seed": seed,
            "valid_size": valid_size,
            "train_count": len(train_rows),
            "valid_count": len(valid_rows),
            "output_files": {name: str(path) for name, path in output_paths.items()},
        }
    )

    write_json(output_paths["train_clean"], cleaned_rows)
    write_json(output_paths["train_sft"], train_rows)
    write_json(output_paths["valid_sft"], valid_rows)
    write_json(output_paths["clean_report"], report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="train.json", help="原始训练数据路径")
    parser.add_argument("--output-dir", default="data", help="清洗数据输出目录")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="输出数据版本号")
    parser.add_argument("--valid-size", type=int, default=DEFAULT_VALID_SIZE, help="固定验证集大小")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="固定切分随机种子")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_prepare_data(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        version=args.version,
        valid_size=args.valid_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
