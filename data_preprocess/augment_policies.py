from dataclasses import dataclass
from typing import Any, Callable

from data_preprocess.feature_tags import (
    FEATURE_TAG_ORDER,
    GEOMETRY_KEYWORDS,
    MULTI_STEP_KEYWORDS,
    RATE_RATIO_KEYWORDS,
    UNIT_SCALE_KEYWORDS,
    WEAK_ANSWER_TYPES,
)


CLEAN008_ALIAS = "clean008"
CLEAN008_POLICY_NAME = "clean008_weak_type_oversample_v1"
CLEAN008_AUGMENT_ID_SUFFIX = "__clean008_aug1"
CLEAN008_DUPLICATE_SOURCE_ID_SUFFIX = "__clean008_src"
CLEAN008_REASON_ORDER = ("weak_answer_type", "rate_ratio", "unit_scale_focus")

CLEAN010_ALIAS = "clean010"
CLEAN010_POLICY_NAME = "clean010_geometry_oversample_v1"
CLEAN010_AUGMENT_ID_SUFFIX = "__clean010_aug1"
CLEAN010_DUPLICATE_SOURCE_ID_SUFFIX = "__clean010_src"
CLEAN010_REASON_ORDER = ("geometry_focus",)

MAX_DUPLICATE_PER_SOURCE = 1
UNIT_SCALE_FOCUS_TAGS = ("weak_answer_type", "rate_ratio", "geometry")

ReasonBuilder = Callable[[list[str]], list[str]]
StrategyParameterBuilder = Callable[["AugmentPolicy", str], dict[str, Any]]


@dataclass(frozen=True)
class AugmentPolicy:
    alias: str
    name: str
    augment_id_suffix: str
    duplicate_source_id_suffix: str
    max_duplicate_per_source: int
    reason_order: tuple[str, ...]
    _reason_builder: ReasonBuilder
    _strategy_parameter_builder: StrategyParameterBuilder

    def build_reasons(self, feature_tags: list[str]) -> list[str]:
        return self._reason_builder(feature_tags)

    def strategy_parameters(self, original_id_field: str) -> dict[str, Any]:
        return self._strategy_parameter_builder(self, original_id_field)


def _build_clean008_reasons(feature_tags: list[str]) -> list[str]:
    feature_tag_set = set(feature_tags)
    reason_flags = {
        "weak_answer_type": "weak_answer_type" in feature_tag_set,
        "rate_ratio": "rate_ratio" in feature_tag_set,
        "unit_scale_focus": "unit_scale" in feature_tag_set
        and any(tag in feature_tag_set for tag in UNIT_SCALE_FOCUS_TAGS),
    }
    return [reason for reason in CLEAN008_REASON_ORDER if reason_flags[reason]]


def _build_clean010_reasons(feature_tags: list[str]) -> list[str]:
    # clean-010 只验证几何标签增强，其他标签不单独触发复制。
    if "geometry" in set(feature_tags):
        return ["geometry_focus"]
    return []


def _build_clean008_strategy_parameters(policy: AugmentPolicy, original_id_field: str) -> dict[str, Any]:
    return {
        "policy_alias": policy.alias,
        "policy_name": policy.name,
        "weak_answer_types": list(WEAK_ANSWER_TYPES),
        "rate_ratio_keywords": list(RATE_RATIO_KEYWORDS),
        "unit_scale_keywords": list(UNIT_SCALE_KEYWORDS),
        "geometry_keywords": list(GEOMETRY_KEYWORDS),
        "multi_step_keywords": list(MULTI_STEP_KEYWORDS),
        "feature_tag_order": list(FEATURE_TAG_ORDER),
        "augment_reason_order": list(policy.reason_order),
        "unit_scale_focus_tags": list(UNIT_SCALE_FOCUS_TAGS),
        "max_duplicate_per_source": policy.max_duplicate_per_source,
        "augment_id_suffix": policy.augment_id_suffix,
        "duplicate_source_id_suffix": policy.duplicate_source_id_suffix,
        "original_id_field": original_id_field,
        "multi_step_triggers_copy": False,
    }


def _build_clean010_strategy_parameters(policy: AugmentPolicy, original_id_field: str) -> dict[str, Any]:
    return {
        "policy_alias": policy.alias,
        "policy_name": policy.name,
        "trigger_tag": "geometry",
        "augment_reason": "geometry_focus",
        "feature_tag_order": list(FEATURE_TAG_ORDER),
        "geometry_keywords": list(GEOMETRY_KEYWORDS),
        "augment_reason_order": list(policy.reason_order),
        "max_duplicate_per_source": policy.max_duplicate_per_source,
        "augment_id_suffix": policy.augment_id_suffix,
        "duplicate_source_id_suffix": policy.duplicate_source_id_suffix,
        "original_id_field": original_id_field,
        "copy_unit_scale_only": False,
        "copy_weak_answer_type_only": False,
        "copy_rate_ratio_only": False,
        "copy_multi_step_only": False,
    }


CLEAN008_POLICY = AugmentPolicy(
    alias=CLEAN008_ALIAS,
    name=CLEAN008_POLICY_NAME,
    augment_id_suffix=CLEAN008_AUGMENT_ID_SUFFIX,
    duplicate_source_id_suffix=CLEAN008_DUPLICATE_SOURCE_ID_SUFFIX,
    max_duplicate_per_source=MAX_DUPLICATE_PER_SOURCE,
    reason_order=CLEAN008_REASON_ORDER,
    _reason_builder=_build_clean008_reasons,
    _strategy_parameter_builder=_build_clean008_strategy_parameters,
)
CLEAN010_POLICY = AugmentPolicy(
    alias=CLEAN010_ALIAS,
    name=CLEAN010_POLICY_NAME,
    augment_id_suffix=CLEAN010_AUGMENT_ID_SUFFIX,
    duplicate_source_id_suffix=CLEAN010_DUPLICATE_SOURCE_ID_SUFFIX,
    max_duplicate_per_source=MAX_DUPLICATE_PER_SOURCE,
    reason_order=CLEAN010_REASON_ORDER,
    _reason_builder=_build_clean010_reasons,
    _strategy_parameter_builder=_build_clean010_strategy_parameters,
)

# 增强主流程只依赖 registry 解析结果，不在流程层堆具体策略分支。
POLICY_REGISTRY = {
    CLEAN008_POLICY.alias: CLEAN008_POLICY,
    CLEAN010_POLICY.alias: CLEAN010_POLICY,
}


def policy_aliases() -> tuple[str, ...]:
    return tuple(POLICY_REGISTRY)


def resolve_policy(alias: str) -> AugmentPolicy:
    if alias not in POLICY_REGISTRY:
        raise ValueError(f"未知增强策略：{alias}")
    return POLICY_REGISTRY[alias]
