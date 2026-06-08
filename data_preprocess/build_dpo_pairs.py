import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from answer_utils import compare_answers, extract_final_answer
from prompting.profiles import get_prompt_profile, select_instruction


DEFAULT_TRAIN_FILE = "data/clean_data/train_sft_cot_v1.json"
DEFAULT_GENERATED_RAW_FILE = "eval_result/train_dpo_source/cot-001-train-dpo-source/raw.jsonl"
DEFAULT_OUTPUT_FILE = "data/clean_data/train_dpo_cot001_v1.json"
DEFAULT_REPORT_FILE = "data/clean_data/dpo_report_cot001_v1.json"


@dataclass(frozen=True)
class DpoBuildConfig:
    train_file: Path
    generated_raw_file: Path
    output_file: Path
    report_file: Path
    prompt_profile: str


@dataclass(frozen=True)
class DpoBuildReport:
    train_file: str
    generated_raw_file: str
    output_file: str
    prompt_profile: str
    train_count: int
    generated_count: int
    chosen_valid_count: int
    rejected_valid_count: int
    pair_count: int


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"{path} 第 {line_number} 行为空")
            row = json.loads(text)
            if not isinstance(row, dict):
                raise TypeError(f"{path} 第 {line_number} 行不是 dict")
            rows.append(row)
    return rows


def require_text(row: dict[str, Any], key: str, row_id: str) -> str:
    if key not in row:
        raise KeyError(f"样本 {row_id} 缺少字段：{key}")
    value = row[key]
    if value is None:
        raise ValueError(f"样本 {row_id} 字段为空：{key}")
    if isinstance(value, (list, dict, tuple, set)):
        raise TypeError(f"样本 {row_id} 字段类型不支持：{key}={type(value).__name__}")
    text = str(value)
    if not text.strip():
        raise ValueError(f"样本 {row_id} 字段为空字符串：{key}")
    return text


def require_bool(row: dict[str, Any], key: str, row_id: str) -> bool:
    if key not in row:
        raise KeyError(f"样本 {row_id} 缺少字段：{key}")
    value = row[key]
    if not isinstance(value, bool):
        raise TypeError(f"样本 {row_id} 字段类型不支持：{key}={type(value).__name__}")
    return value


def index_by_id(rows: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = require_text(row, "id", source_name)
        if row_id in indexed:
            raise ValueError(f"{source_name} 中存在重复 id：{row_id}")
        indexed[row_id] = row
    return indexed


def validate_chosen_response(row: dict[str, Any]) -> str:
    row_id = require_text(row, "id", "train")
    gold_answer = require_text(row, "answer", row_id)
    chosen = require_text(row, "cot_response", row_id)
    if "最终答案：" not in chosen:
        raise ValueError(f"样本 {row_id} chosen 缺少最终答案标记")
    extracted = extract_final_answer(chosen)
    if not compare_answers(extracted.answer, gold_answer):
        raise ValueError(f"样本 {row_id} chosen 最终答案与 gold 不一致")
    return chosen


def validate_rejected_response(row: dict[str, Any]) -> str | None:
    row_id = require_text(row, "id", "generated")
    correct = require_bool(row, "correct", row_id)
    extract_status = require_text(row, "extract_status", row_id)
    if correct is not False:
        return None
    if extract_status != "matched":
        return None

    raw_response = require_text(row, "raw_response", row_id)
    if "最终答案：" not in raw_response:
        return None

    # generated 标记为错误时，再用当前抽取规则复核一次，避免把正确样本写成 rejected。
    gold_answer = require_text(row, "answer", row_id)
    extracted = extract_final_answer(raw_response)
    if compare_answers(extracted.answer, gold_answer):
        raise ValueError(f"样本 {row_id} generated 标记为错误但抽取答案与 gold 一致")
    return raw_response


def build_prompt(row: dict[str, Any], prompt_profile: str) -> list[dict[str, str]]:
    row_id = require_text(row, "id", "train")
    profile = get_prompt_profile(prompt_profile)
    instruction = select_instruction(profile, require_text(row, "instruction", row_id))
    question = require_text(row, "question", row_id)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]


def build_dpo_pairs(config: DpoBuildConfig) -> tuple[list[dict[str, Any]], DpoBuildReport]:
    train_rows = load_json_list(config.train_file)
    generated_rows = load_jsonl(config.generated_raw_file)
    generated_by_id = index_by_id(generated_rows, "generated_raw")

    pairs: list[dict[str, Any]] = []
    chosen_valid_count = 0
    rejected_valid_count = 0
    for train_row in train_rows:
        row_id = require_text(train_row, "id", "train")
        chosen = validate_chosen_response(train_row)
        chosen_valid_count += 1

        generated_row = generated_by_id.get(row_id)
        if generated_row is None:
            raise KeyError(f"generated_raw 缺少训练样本 id：{row_id}")
        rejected = validate_rejected_response(generated_row)
        if rejected is None:
            continue

        rejected_valid_count += 1
        pairs.append(
            {
                "prompt": build_prompt(train_row, config.prompt_profile),
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
        )

    if not pairs:
        raise ValueError("DPO pair 数量为 0")

    report = DpoBuildReport(
        train_file=str(config.train_file),
        generated_raw_file=str(config.generated_raw_file),
        output_file=str(config.output_file),
        prompt_profile=config.prompt_profile,
        train_count=len(train_rows),
        generated_count=len(generated_rows),
        chosen_valid_count=chosen_valid_count,
        rejected_valid_count=rejected_valid_count,
        pair_count=len(pairs),
    )
    return pairs, report


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--generated-raw-file", default=DEFAULT_GENERATED_RAW_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE)
    parser.add_argument("--prompt-profile", default="cot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DpoBuildConfig(
        train_file=Path(args.train_file),
        generated_raw_file=Path(args.generated_raw_file),
        output_file=Path(args.output_file),
        report_file=Path(args.report_file),
        prompt_profile=args.prompt_profile,
    )
    pairs, report = build_dpo_pairs(config)
    write_json(config.output_file, pairs)
    write_json(config.report_file, asdict(report))
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
