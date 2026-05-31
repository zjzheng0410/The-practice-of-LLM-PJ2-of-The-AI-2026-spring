import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_preprocess.feature_tags import (
    FEATURE_TAG_ORDER,
    GEOMETRY_KEYWORDS,
    MULTI_STEP_KEYWORDS,
    RATE_RATIO_KEYWORDS,
    UNIT_SCALE_KEYWORDS,
    WEAK_ANSWER_TYPES,
    build_feature_tags,
)
from data_preprocess.prepare_data import load_json_list, write_json
from data_preprocess.report import build_augment_report
from data_preprocess.split import build_group_key


DEFAULT_INPUT_TRAIN = "data/clean_data/train_sft_v2.json"
DEFAULT_VALID_FILE = "data/clean_data/valid_sft_v2.json"
DEFAULT_OUTPUT_TRAIN = "data/clean_data/train_sft_clean008.json"
DEFAULT_REPORT_FILE = "data/clean_data/augment_report_clean008.json"
DEFAULT_POLICY_ALIAS = "clean008"
CLEAN008_POLICY_NAME = "clean008_weak_type_oversample_v1"
AUGMENT_ID_SUFFIX = "__clean008_aug1"
MAX_DUPLICATE_PER_SOURCE = 1
ORIGINAL_ID_FIELD = "original_id"
DUPLICATE_SOURCE_ID_SUFFIX = "__clean008_src"

TRAIN_REQUIRED_FIELDS = ("id", "question", "answer", "instruction", "answer_type")
AUGMENT_REASON_ORDER = ("weak_answer_type", "rate_ratio", "unit_scale_focus")
UNIT_SCALE_FOCUS_TAGS = ("weak_answer_type", "rate_ratio", "geometry")


@dataclass(frozen=True)
class AugmentBuildResult:
    output_rows: list[dict[str, Any]]
    added_rows: list[dict[str, Any]]
    feature_tag_counts: Counter[str]
    augment_reason_counts: Counter[str]
    duplicate_counts_by_source: Counter[str]


@dataclass(frozen=True)
class SourceIdNormalizeResult:
    rows: list[dict[str, Any]]
    duplicate_group_count: int
    renamed_row_count: int
    renamed_id_sample: list[dict[str, Any]]


def _resolve_policy(policy: str) -> str:
    if policy != DEFAULT_POLICY_ALIAS:
        raise ValueError(f"仅支持 --policy {DEFAULT_POLICY_ALIAS}")
    return CLEAN008_POLICY_NAME


def _ensure_can_write(output_paths: dict[str, Path], overwrite: bool) -> None:
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出文件已存在，请使用 --overwrite：" + ", ".join(existing))


def _validate_train_row(row: dict[str, Any], row_index: int) -> None:
    for key in TRAIN_REQUIRED_FIELDS:
        if key not in row:
            raise KeyError(f"训练集第 {row_index} 条样本缺少字段：{key}")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError(f"训练集第 {row_index} 条样本 id 必须是非空字符串")
    if not isinstance(row["instruction"], str) or not row["instruction"]:
        raise ValueError(f"训练集第 {row_index} 条样本 instruction 必须是非空字符串")


def _ensure_unique_ids(rows: list[dict[str, Any]], source_name: str) -> None:
    ids = [str(row["id"]) for row in rows]
    counts = Counter(ids)
    duplicate_ids = sorted(id_value for id_value, count in counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"{source_name} 存在重复 id：{duplicate_ids[:20]}")


def normalize_duplicate_source_ids(train_rows: list[dict[str, Any]]) -> SourceIdNormalizeResult:
    id_counts = Counter(str(row["id"]) for row in train_rows)
    duplicate_ids = {id_value for id_value, count in id_counts.items() if count > 1}
    seen_duplicate_ids: Counter[str] = Counter()
    renamed_by_original_id: dict[str, list[str]] = {}
    normalized_rows: list[dict[str, Any]] = []

    for row in train_rows:
        source_id = str(row["id"])
        normalized_row = dict(row)
        if source_id in duplicate_ids:
            seen_duplicate_ids[source_id] += 1
            new_id = f"{source_id}{DUPLICATE_SOURCE_ID_SUFFIX}{seen_duplicate_ids[source_id]}"
            # 仅重复原始 id 参与重编号，并保留原始 id 便于回溯。
            normalized_row[ORIGINAL_ID_FIELD] = source_id
            normalized_row["id"] = new_id
            renamed_by_original_id.setdefault(source_id, []).append(new_id)
        normalized_rows.append(normalized_row)

    _ensure_unique_ids(normalized_rows, "重编号后训练集")
    renamed_id_sample = [
        {"original_id": original_id, "renamed_ids": renamed_ids}
        for original_id, renamed_ids in sorted(renamed_by_original_id.items())[:20]
    ]
    return SourceIdNormalizeResult(
        rows=normalized_rows,
        duplicate_group_count=len(duplicate_ids),
        renamed_row_count=sum(len(ids) for ids in renamed_by_original_id.values()),
        renamed_id_sample=renamed_id_sample,
    )


def _build_group_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {build_group_key(row) for row in rows}


def find_train_valid_overlap(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
) -> set[str]:
    return _build_group_keys(train_rows) & _build_group_keys(valid_rows)


def build_augment_reasons(feature_tags: list[str]) -> list[str]:
    feature_tag_set = set(feature_tags)
    reason_flags = {
        "weak_answer_type": "weak_answer_type" in feature_tag_set,
        "rate_ratio": "rate_ratio" in feature_tag_set,
        "unit_scale_focus": "unit_scale" in feature_tag_set
        and any(tag in feature_tag_set for tag in UNIT_SCALE_FOCUS_TAGS),
    }
    return [reason for reason in AUGMENT_REASON_ORDER if reason_flags[reason]]


def _build_augmented_id(source_id: str) -> str:
    return source_id + AUGMENT_ID_SUFFIX


def build_augmented_train_rows(
    train_rows: list[dict[str, Any]],
    policy_name: str = CLEAN008_POLICY_NAME,
) -> AugmentBuildResult:
    output_rows = [dict(row) for row in train_rows]
    added_rows: list[dict[str, Any]] = []
    feature_tag_counts: Counter[str] = Counter()
    augment_reason_counts: Counter[str] = Counter()
    duplicate_counts_by_source: Counter[str] = Counter()

    for row_index, row in enumerate(train_rows):
        _validate_train_row(row, row_index)
        feature_tags = build_feature_tags(row, row_index=row_index)
        for tag in feature_tags:
            feature_tag_counts[tag] += 1

        augment_reasons = build_augment_reasons(feature_tags)
        if not augment_reasons:
            continue

        source_id = row["id"]
        if duplicate_counts_by_source[source_id] >= MAX_DUPLICATE_PER_SOURCE:
            raise ValueError(f"源样本重复复制超过限制：{source_id}")

        # 单个源样本最多复制一次，多原因只记录到 augment_reason。
        augmented_row = dict(row)
        augmented_row["id"] = _build_augmented_id(source_id)
        augmented_row["source_id"] = source_id
        augmented_row["is_augmented"] = True
        augmented_row["augment_policy"] = policy_name
        augmented_row["augment_reason"] = augment_reasons
        augmented_row["feature_tags"] = feature_tags
        added_rows.append(augmented_row)
        duplicate_counts_by_source[source_id] += 1
        for reason in augment_reasons:
            augment_reason_counts[reason] += 1

    output_rows.extend(added_rows)
    _ensure_unique_ids(output_rows, "增强后训练集")
    return AugmentBuildResult(
        output_rows=output_rows,
        added_rows=added_rows,
        feature_tag_counts=feature_tag_counts,
        augment_reason_counts=augment_reason_counts,
        duplicate_counts_by_source=duplicate_counts_by_source,
    )


def _build_strategy_parameters() -> dict[str, Any]:
    return {
        "policy_alias": DEFAULT_POLICY_ALIAS,
        "policy_name": CLEAN008_POLICY_NAME,
        "weak_answer_types": list(WEAK_ANSWER_TYPES),
        "rate_ratio_keywords": list(RATE_RATIO_KEYWORDS),
        "unit_scale_keywords": list(UNIT_SCALE_KEYWORDS),
        "geometry_keywords": list(GEOMETRY_KEYWORDS),
        "multi_step_keywords": list(MULTI_STEP_KEYWORDS),
        "feature_tag_order": list(FEATURE_TAG_ORDER),
        "augment_reason_order": list(AUGMENT_REASON_ORDER),
        "unit_scale_focus_tags": list(UNIT_SCALE_FOCUS_TAGS),
        "max_duplicate_per_source": MAX_DUPLICATE_PER_SOURCE,
        "augment_id_suffix": AUGMENT_ID_SUFFIX,
        "duplicate_source_id_suffix": DUPLICATE_SOURCE_ID_SUFFIX,
        "original_id_field": ORIGINAL_ID_FIELD,
        "multi_step_triggers_copy": False,
    }


def _validate_augment_report(report: dict[str, Any]) -> None:
    if report["train_valid_overlap_group_count"] != 0:
        raise ValueError("train/valid 存在同题泄漏")
    expected_count = report["original_train_count"] + report["added_count"]
    if report["augmented_train_count"] != expected_count:
        raise ValueError("增强后样本数与报告 added_count 不一致")
    if report["max_duplicate_per_source"] > MAX_DUPLICATE_PER_SOURCE:
        raise ValueError("单个源样本复制次数超过限制")


def run_augment_train(
    input_train: Path,
    valid_file: Path,
    output_train: Path,
    report_file: Path,
    policy: str = DEFAULT_POLICY_ALIAS,
    overwrite: bool = False,
) -> dict[str, Any]:
    policy_name = _resolve_policy(policy)
    output_paths = {"output_train": output_train, "augment_report": report_file}
    _ensure_can_write(output_paths, overwrite=overwrite)

    train_rows = load_json_list(input_train)
    valid_rows = load_json_list(valid_file)
    for row_index, row in enumerate(train_rows):
        _validate_train_row(row, row_index)
    source_id_result = normalize_duplicate_source_ids(train_rows)
    normalized_train_rows = source_id_result.rows

    overlap_group_keys = find_train_valid_overlap(normalized_train_rows, valid_rows)
    if overlap_group_keys:
        raise ValueError(f"train/valid 存在同题泄漏：{sorted(overlap_group_keys)[:20]}")

    build_result = build_augmented_train_rows(normalized_train_rows, policy_name=policy_name)
    report = build_augment_report(
        policy=policy_name,
        input_train_file=input_train,
        valid_file=valid_file,
        output_train_file=output_train,
        original_train_rows=train_rows,
        augmented_train_rows=build_result.output_rows,
        added_rows=build_result.added_rows,
        feature_tag_counts=build_result.feature_tag_counts,
        augment_reason_counts=build_result.augment_reason_counts,
        duplicate_counts_by_source=build_result.duplicate_counts_by_source,
        train_valid_overlap_group_keys=overlap_group_keys,
        output_paths=output_paths,
        strategy_parameters=_build_strategy_parameters(),
        source_id_duplicate_group_count=source_id_result.duplicate_group_count,
        source_id_renamed_row_count=source_id_result.renamed_row_count,
        source_id_renamed_sample=source_id_result.renamed_id_sample,
    )
    _validate_augment_report(report)

    write_json(output_train, build_result.output_rows)
    write_json(report_file, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", default=DEFAULT_INPUT_TRAIN, help="输入训练集 JSON 路径")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="固定验证集 JSON 路径")
    parser.add_argument("--output-train", default=DEFAULT_OUTPUT_TRAIN, help="增强训练集输出路径")
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE, help="增强报告输出路径")
    parser.add_argument("--policy", default=DEFAULT_POLICY_ALIAS, choices=[DEFAULT_POLICY_ALIAS], help="增强策略")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_augment_train(
        input_train=Path(args.input_train),
        valid_file=Path(args.valid_file),
        output_train=Path(args.output_train),
        report_file=Path(args.report_file),
        policy=args.policy,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
