from collections import Counter
from pathlib import Path
from typing import Any

from data_preprocess.split import GROUP_KEY_POLICY


def _sorted_counter_dict(counter: Counter[str] | dict[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def duplicate_id_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows if "id" in row]
    counts = Counter(ids)
    duplicate_ids = [id_value for id_value, count in counts.items() if count > 1]
    return {
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_extra_rows": sum(counts[id_value] - 1 for id_value in duplicate_ids),
        "duplicate_ids_sample": duplicate_ids[:50],
    }


def build_clean_report(
    raw_rows: list[dict[str, Any]],
    cleaned_candidates: list[dict[str, Any]],
    cleaned_rows: list[dict[str, Any]],
    dropped_empty_answer: list[dict[str, Any]],
    fixed_question_list_count: int,
    skipped_empty_question_part_count: int,
    conflict_groups: list[dict[str, Any]],
) -> dict[str, Any]:
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
    report.update(duplicate_id_report(raw_rows))
    return report


def build_split_report(
    version: str,
    seed: int,
    target_valid_size: int,
    actual_train_size: int,
    actual_valid_size: int,
    train_group_keys: set[str],
    valid_group_keys: set[str],
    overlap_group_keys: set[str],
    dropped_conflict_group_count: int,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "version": version,
        "seed": seed,
        "target_valid_size": target_valid_size,
        "actual_train_size": actual_train_size,
        "actual_valid_size": actual_valid_size,
        "train_group_count": len(train_group_keys),
        "valid_group_count": len(valid_group_keys),
        "dropped_conflict_group_count": dropped_conflict_group_count,
        "train_valid_overlap_group_count": len(overlap_group_keys),
        "train_valid_overlap_group_sample": sorted(overlap_group_keys)[:20],
        "group_key_policy": GROUP_KEY_POLICY,
        "output_files": {name: str(path) for name, path in output_paths.items()},
    }


def build_augment_report(
    policy: str,
    input_train_file: Path,
    valid_file: Path,
    output_train_file: Path,
    original_train_rows: list[dict[str, Any]],
    augmented_train_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    feature_tag_counts: Counter[str],
    augment_reason_counts: Counter[str],
    duplicate_counts_by_source: Counter[str],
    train_valid_overlap_group_keys: set[str],
    output_paths: dict[str, Path],
    strategy_parameters: dict[str, Any],
    source_id_duplicate_group_count: int,
    source_id_renamed_row_count: int,
    source_id_renamed_sample: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "policy": policy,
        "input_train_file": str(input_train_file),
        "valid_file": str(valid_file),
        "output_train_file": str(output_train_file),
        "original_train_count": len(original_train_rows),
        "augmented_train_count": len(augmented_train_rows),
        "added_count": len(added_rows),
        "answer_type_distribution_before": _sorted_counter_dict(
            Counter(str(row["answer_type"]) for row in original_train_rows)
        ),
        "answer_type_distribution_after": _sorted_counter_dict(
            Counter(str(row["answer_type"]) for row in augmented_train_rows)
        ),
        "feature_tag_counts": _sorted_counter_dict(feature_tag_counts),
        "augment_reason_counts": _sorted_counter_dict(augment_reason_counts),
        "max_duplicate_per_source": max(duplicate_counts_by_source.values(), default=0),
        "augmented_source_count": len(duplicate_counts_by_source),
        "train_valid_overlap_group_count": len(train_valid_overlap_group_keys),
        "train_valid_overlap_group_sample": sorted(train_valid_overlap_group_keys)[:20],
        "output_files": {name: str(path) for name, path in output_paths.items()},
        "strategy_parameters": strategy_parameters,
        "source_id_duplicate_group_count": source_id_duplicate_group_count,
        "source_id_renamed_row_count": source_id_renamed_row_count,
        "source_id_renamed_sample": source_id_renamed_sample,
    }
