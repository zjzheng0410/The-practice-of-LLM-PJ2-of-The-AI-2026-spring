import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_preprocess.augment_policies import (
    AugmentPolicy,
    CLEAN008_POLICY,
    UNIT_SCALE_FOCUS_TAGS,
    policy_aliases,
    resolve_policy,
)
from data_preprocess.feature_tags import build_feature_tags
from data_preprocess.prepare_data import load_json_list, write_json
from data_preprocess.report import build_augment_report
from data_preprocess.split import build_group_key


DEFAULT_INPUT_TRAIN = "data/clean_data/train_sft_v2.json"
DEFAULT_VALID_FILE = "data/clean_data/valid_sft_v2.json"
DEFAULT_OUTPUT_TRAIN = "data/clean_data/train_sft_clean008.json"
DEFAULT_REPORT_FILE = "data/clean_data/augment_report_clean008.json"
DEFAULT_POLICY_ALIAS = CLEAN008_POLICY.alias
CLEAN008_POLICY_NAME = CLEAN008_POLICY.name
AUGMENT_ID_SUFFIX = CLEAN008_POLICY.augment_id_suffix
MAX_DUPLICATE_PER_SOURCE = CLEAN008_POLICY.max_duplicate_per_source
ORIGINAL_ID_FIELD = "original_id"
DUPLICATE_SOURCE_ID_SUFFIX = CLEAN008_POLICY.duplicate_source_id_suffix

TRAIN_REQUIRED_FIELDS = ("id", "question", "answer", "instruction", "answer_type")
AUGMENT_REASON_ORDER = CLEAN008_POLICY.reason_order


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


def normalize_duplicate_source_ids(
    train_rows: list[dict[str, Any]],
    policy: AugmentPolicy = CLEAN008_POLICY,
) -> SourceIdNormalizeResult:
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
            new_id = f"{source_id}{policy.duplicate_source_id_suffix}{seen_duplicate_ids[source_id]}"
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


def build_augment_reasons(
    feature_tags: list[str],
    policy: AugmentPolicy = CLEAN008_POLICY,
) -> list[str]:
    return policy.build_reasons(feature_tags)


def _build_augmented_id(source_id: str, policy: AugmentPolicy) -> str:
    return source_id + policy.augment_id_suffix


def build_augmented_train_rows(
    train_rows: list[dict[str, Any]],
    policy: AugmentPolicy = CLEAN008_POLICY,
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

        augment_reasons = build_augment_reasons(feature_tags, policy=policy)
        if not augment_reasons:
            continue

        source_id = row["id"]
        if duplicate_counts_by_source[source_id] >= policy.max_duplicate_per_source:
            raise ValueError(f"源样本重复复制超过限制：{source_id}")

        # 单个源样本最多复制一次，多原因只记录到 augment_reason。
        augmented_row = dict(row)
        augmented_row["id"] = _build_augmented_id(source_id, policy)
        augmented_row["source_id"] = source_id
        augmented_row["is_augmented"] = True
        augmented_row["augment_policy"] = policy.name
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


def _build_strategy_parameters(policy: AugmentPolicy) -> dict[str, Any]:
    return policy.strategy_parameters(original_id_field=ORIGINAL_ID_FIELD)


def _validate_augment_report(report: dict[str, Any], policy: AugmentPolicy) -> None:
    if report["train_valid_overlap_group_count"] != 0:
        raise ValueError("train/valid 存在同题泄漏")
    expected_count = report["original_train_count"] + report["added_count"]
    if report["augmented_train_count"] != expected_count:
        raise ValueError("增强后样本数与报告 added_count 不一致")
    if report["max_duplicate_per_source"] > policy.max_duplicate_per_source:
        raise ValueError("单个源样本复制次数超过限制")


def run_augment_train(
    input_train: Path,
    valid_file: Path,
    output_train: Path,
    report_file: Path,
    policy: str = DEFAULT_POLICY_ALIAS,
    overwrite: bool = False,
) -> dict[str, Any]:
    augment_policy = resolve_policy(policy)
    output_paths = {"output_train": output_train, "augment_report": report_file}
    _ensure_can_write(output_paths, overwrite=overwrite)

    train_rows = load_json_list(input_train)
    valid_rows = load_json_list(valid_file)
    for row_index, row in enumerate(train_rows):
        _validate_train_row(row, row_index)
    source_id_result = normalize_duplicate_source_ids(train_rows, policy=augment_policy)
    normalized_train_rows = source_id_result.rows

    overlap_group_keys = find_train_valid_overlap(normalized_train_rows, valid_rows)
    if overlap_group_keys:
        raise ValueError(f"train/valid 存在同题泄漏：{sorted(overlap_group_keys)[:20]}")

    build_result = build_augmented_train_rows(normalized_train_rows, policy=augment_policy)
    report = build_augment_report(
        policy=augment_policy.name,
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
        strategy_parameters=_build_strategy_parameters(augment_policy),
        source_id_duplicate_group_count=source_id_result.duplicate_group_count,
        source_id_renamed_row_count=source_id_result.renamed_row_count,
        source_id_renamed_sample=source_id_result.renamed_id_sample,
    )
    _validate_augment_report(report, augment_policy)

    write_json(output_train, build_result.output_rows)
    write_json(report_file, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", default=DEFAULT_INPUT_TRAIN, help="输入训练集 JSON 路径")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="固定验证集 JSON 路径")
    parser.add_argument("--output-train", default=DEFAULT_OUTPUT_TRAIN, help="增强训练集输出路径")
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE, help="增强报告输出路径")
    parser.add_argument("--policy", default=DEFAULT_POLICY_ALIAS, choices=policy_aliases(), help="增强策略")
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
