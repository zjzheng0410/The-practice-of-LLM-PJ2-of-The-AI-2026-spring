from collections import Counter
from pathlib import Path
from typing import Any

from data_preprocess.split import GROUP_KEY_POLICY


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

