from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from answer_utils import normalize_answer
from ensemble.prediction_io import SourceConfig, validate_source_configs


INVALID_EXTRACT_STATUSES = {"error", "empty"}
NULL_ANSWER = "Null"
SUPPORTED_VOTE_STRATEGY = "majority"
VOTE_STATUS_MAJORITY = "majority"
VOTE_STATUS_TIE_BREAK = "tie_break"
VOTE_STATUS_ALL_INVALID_FALLBACK = "all_invalid_fallback"


@dataclass(frozen=True)
class VoteCandidate:
    source: str
    row_id: str
    raw_answer: Any
    normalized_answer: str | None
    extract_status: str | None
    participates: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VoteDecision:
    answer: str
    selected_source: str
    vote_status: str
    tie_break_applied: bool
    vote_counts: dict[str, int]
    candidate_answers: dict[str, Any]
    candidate_normalized_answers: dict[str, str | None]
    candidate_extract_status: dict[str, str | None]
    skipped_sources: dict[str, str]
    source_order: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_vote_strategy(strategy: str) -> None:
    if strategy != SUPPORTED_VOTE_STRATEGY:
        raise ValueError(f"--strategy 仅支持 {SUPPORTED_VOTE_STRATEGY}：{strategy}")


def _normalize_candidate_answer(source: str, row_id: str, raw_answer: Any) -> str:
    try:
        return normalize_answer(raw_answer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} id={row_id} 答案规范化失败：{exc}") from exc


def _read_extract_status(source: str, row_id: str, record: dict[str, Any]) -> str | None:
    if "extract_status" not in record:
        return None
    status = record["extract_status"]
    if status is not None and not isinstance(status, str):
        raise TypeError(f"{source} id={row_id} extract_status 必须是字符串")
    return status


def _is_null_answer(raw_answer: Any) -> bool:
    if raw_answer is None:
        return True
    if isinstance(raw_answer, str):
        return raw_answer.strip().lower() == NULL_ANSWER.lower()
    return False


def build_vote_candidates(
    records_by_source: dict[str, dict[str, Any]],
    source_configs: list[SourceConfig],
    answer_field: str,
    row_id: str,
) -> list[VoteCandidate]:
    validate_source_configs(source_configs)
    if not isinstance(answer_field, str) or not answer_field:
        raise ValueError("answer_field 必须是非空字符串")

    candidates: list[VoteCandidate] = []
    for config in source_configs:
        if config.name not in records_by_source:
            raise KeyError(f"id={row_id} 缺少 source 记录：{config.name}")
        record = records_by_source[config.name]
        if answer_field not in record:
            raise KeyError(f"{config.name} id={row_id} 缺少 {answer_field}")

        extract_status = _read_extract_status(config.name, row_id, record)
        raw_answer = record[answer_field]
        null_answer = _is_null_answer(raw_answer)
        participates = extract_status not in INVALID_EXTRACT_STATUSES and not null_answer
        skip_reason = None
        if extract_status in INVALID_EXTRACT_STATUSES:
            skip_reason = f"extract_status={extract_status}"
        elif null_answer:
            skip_reason = f"answer={NULL_ANSWER}"

        normalized_answer = None
        if participates:
            # 有效候选的规范化失败必须直接暴露到 source/id，禁止静默改写。
            normalized_answer = _normalize_candidate_answer(
                source=config.name,
                row_id=row_id,
                raw_answer=raw_answer,
            )
        candidates.append(
            VoteCandidate(
                source=config.name,
                row_id=row_id,
                raw_answer=raw_answer,
                normalized_answer=normalized_answer,
                extract_status=extract_status,
                participates=participates,
                skip_reason=skip_reason,
            )
        )
    return candidates


def _candidate_maps(candidates: list[VoteCandidate]) -> dict[str, dict[str, Any]]:
    return {
        "candidate_answers": {
            candidate.source: candidate.raw_answer
            for candidate in candidates
        },
        "candidate_normalized_answers": {
            candidate.source: candidate.normalized_answer
            for candidate in candidates
        },
        "candidate_extract_status": {
            candidate.source: candidate.extract_status
            for candidate in candidates
        },
        "skipped_sources": {
            candidate.source: str(candidate.skip_reason)
            for candidate in candidates
            if candidate.skip_reason is not None
        },
    }


def _candidate_by_source(candidates: list[VoteCandidate]) -> dict[str, VoteCandidate]:
    indexed: dict[str, VoteCandidate] = {}
    for candidate in candidates:
        if candidate.source in indexed:
            raise ValueError(f"投票候选 source 重复：{candidate.source}")
        indexed[candidate.source] = candidate
    return indexed


def ranked_tie_break(
    tied_answers: set[str],
    candidates: list[VoteCandidate],
    source_order: list[str],
) -> tuple[str, str]:
    if not tied_answers:
        raise ValueError("平票答案集合不能为空")
    candidates_by_source = _candidate_by_source(candidates)
    for source in source_order:
        candidate = candidates_by_source.get(source)
        if candidate is None:
            raise KeyError(f"source_order 存在未知 source：{source}")
        if candidate.participates and candidate.normalized_answer in tied_answers:
            if candidate.normalized_answer is None:
                raise ValueError(f"{candidate.source} id={candidate.row_id} 缺少规范化答案")
            return candidate.source, candidate.normalized_answer
    raise ValueError("平票集合没有可选择候选")


def majority_vote(candidates: list[VoteCandidate], source_order: list[str]) -> VoteDecision:
    if not source_order:
        raise ValueError("source_order 不能为空")
    if not candidates:
        raise ValueError("candidates 不能为空")

    candidates_by_source = _candidate_by_source(candidates)
    missing_sources = [source for source in source_order if source not in candidates_by_source]
    if missing_sources:
        raise KeyError(f"缺少投票候选 source：{missing_sources}")

    maps = _candidate_maps(candidates)
    participating = [candidate for candidate in candidates if candidate.participates]
    if not participating:
        primary_source = source_order[0]
        primary_candidate = candidates_by_source[primary_source]
        primary_answer = primary_candidate.normalized_answer
        if primary_answer is None:
            primary_answer = _normalize_candidate_answer(
                source=primary_candidate.source,
                row_id=primary_candidate.row_id,
                raw_answer=primary_candidate.raw_answer,
            )
        maps["candidate_normalized_answers"] = {
            **maps["candidate_normalized_answers"],
            primary_source: primary_answer,
        }
        return VoteDecision(
            answer=primary_answer,
            selected_source=primary_source,
            vote_status=VOTE_STATUS_ALL_INVALID_FALLBACK,
            tie_break_applied=False,
            vote_counts={},
            source_order=list(source_order),
            **maps,
        )

    missing_normalized = [
        f"{candidate.source} id={candidate.row_id}"
        for candidate in participating
        if candidate.normalized_answer is None
    ]
    if missing_normalized:
        raise ValueError(f"参与投票候选缺少规范化答案：{missing_normalized}")

    vote_counts = dict(Counter(str(candidate.normalized_answer) for candidate in participating))
    max_count = max(vote_counts.values())
    tied_answers = {
        answer
        for answer, count in vote_counts.items()
        if count == max_count
    }
    selected_source, selected_answer = ranked_tie_break(
        tied_answers=tied_answers,
        candidates=candidates,
        source_order=source_order,
    )
    tie_break_applied = len(tied_answers) > 1
    vote_status = VOTE_STATUS_TIE_BREAK if tie_break_applied else VOTE_STATUS_MAJORITY
    return VoteDecision(
        answer=selected_answer,
        selected_source=selected_source,
        vote_status=vote_status,
        tie_break_applied=tie_break_applied,
        vote_counts=vote_counts,
        source_order=list(source_order),
        **maps,
    )


def vote_sample(
    records_by_source: dict[str, dict[str, Any]],
    source_configs: list[SourceConfig],
    answer_field: str,
    row_id: str,
) -> VoteDecision:
    candidates = build_vote_candidates(
        records_by_source=records_by_source,
        source_configs=source_configs,
        answer_field=answer_field,
        row_id=row_id,
    )
    return majority_vote(
        candidates=candidates,
        source_order=[config.name for config in source_configs],
    )
