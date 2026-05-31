import random
from dataclasses import dataclass
from typing import Any


GROUP_KEY_POLICY = "cleaned_question"


@dataclass(frozen=True)
class SplitResult:
    train_rows: list[dict[str, Any]]
    valid_rows: list[dict[str, Any]]
    train_group_keys: set[str]
    valid_group_keys: set[str]
    overlap_group_keys: set[str]


def build_group_key(row: dict[str, Any]) -> str:
    if "question" not in row:
        raise KeyError("样本缺少 question 字段，无法构造分组 key")
    question = row["question"]
    if not isinstance(question, str) or not question:
        raise ValueError("清洗后 question 必须是非空字符串")
    return question


def group_rows_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = build_group_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def _choose_valid_group_keys(
    groups: dict[str, list[dict[str, Any]]],
    valid_size: int,
    seed: int,
) -> set[str]:
    if valid_size < 0:
        raise ValueError("valid_size 不能为负数")

    total_rows = sum(len(group_rows) for group_rows in groups.values())
    if valid_size > total_rows:
        raise ValueError(f"valid_size={valid_size} 大于清洗后样本数={total_rows}")
    if valid_size == 0:
        return set()
    if not groups:
        raise ValueError("没有可切分的数据分组")

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    max_group_size = max(len(group_rows) for _, group_rows in group_items)
    search_limit = min(total_rows, valid_size + max_group_size)

    # 分组大小不是恒定 1，使用子集和选择最接近目标数量的分组集合，保证同题不跨 split。
    reachable: dict[int, tuple[int, int]] = {0: (-1, -1)}
    for group_index, (_, group_rows) in enumerate(group_items):
        group_size = len(group_rows)
        for current_size in sorted(reachable.keys(), reverse=True):
            next_size = current_size + group_size
            if next_size > search_limit or next_size in reachable:
                continue
            reachable[next_size] = (current_size, group_index)

    best_size = min(
        reachable,
        key=lambda size: (abs(size - valid_size), size > valid_size, size),
    )
    selected_indices: set[int] = set()
    cursor = best_size
    while cursor:
        previous_size, group_index = reachable[cursor]
        if group_index < 0:
            raise ValueError("验证集分组回溯失败")
        selected_indices.add(group_index)
        cursor = previous_size

    return {group_items[index][0] for index in selected_indices}


def split_train_valid_by_group(
    rows: list[dict[str, Any]],
    valid_size: int,
    seed: int,
) -> SplitResult:
    groups = group_rows_by_question(rows)
    valid_group_keys = _choose_valid_group_keys(groups, valid_size=valid_size, seed=seed)
    train_group_keys = set(groups) - valid_group_keys

    valid_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in rows:
        if build_group_key(row) in valid_group_keys:
            valid_rows.append(row)
        else:
            train_rows.append(row)

    overlap_group_keys = train_group_keys & valid_group_keys
    if overlap_group_keys:
        sample = sorted(overlap_group_keys)[:10]
        raise ValueError(f"train/valid 存在同题泄漏：{sample}")

    return SplitResult(
        train_rows=train_rows,
        valid_rows=valid_rows,
        train_group_keys=train_group_keys,
        valid_group_keys=valid_group_keys,
        overlap_group_keys=overlap_group_keys,
    )

