"""Performance requirement measurement for Hyper-τ evaluations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any, Optional

from tau2.data_model.message import AssistantMessage, Message
from tau2.hyper.data_model import (
    CreditPerformanceRequirement,
    LatencyPerformanceRequirement,
    PerformanceRequirement,
)


def _percentile(values: list[float], percentile: int) -> float | None:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def collect_agent_turn_latencies(messages: Iterable[Message]) -> list[float]:
    """Collect complete ``generate_next_message`` durations from a trajectory."""
    return [
        message.agent_turn_latency_seconds
        for message in messages
        if isinstance(message, AssistantMessage)
        and message.agent_turn_latency_seconds is not None
    ]


def _conversation_credits(
    usage: Optional[Mapping[str, Any]],
    models: Optional[set[str]],
) -> float:
    """Price one conversation, optionally scoped to a set of metered models."""
    if usage is None:
        return 0.0
    if models is not None:
        by_model = usage.get("by_model")
        if by_model is not None:
            return sum(
                float(stats.get("credits", 0.0))
                for model, stats in by_model.items()
                if model in models
            )
        # Usage recorded before per-model breakdowns: meter the total.
    if usage.get("total_credits") is None:
        raise ValueError("Agent credit usage is missing total_credits")
    return float(usage["total_credits"])


def collect_conversation_credits(
    usages: Iterable[Optional[Mapping[str, Any]]],
    models: Optional[set[str]] = None,
) -> list[float]:
    """Collect one credit sample per conversation, including failed runs.

    When ``models`` is given, only credits incurred on those models count —
    this is how tier-scoped budgets meter their own pool.
    """
    return [_conversation_credits(usage, models) for usage in usages]


def parse_performance_requirement(value: Mapping[str, Any]) -> PerformanceRequirement:
    """Parse a serialized requirement using its discriminating type."""
    if value.get("type") == "credits":
        return CreditPerformanceRequirement.model_validate(value)
    return LatencyPerformanceRequirement.model_validate(value)


def _format_budget_status(mean_credits: float, budget: float) -> str:
    """Render the shared over/near-budget suffix for one metered budget."""
    ratio = mean_credits / budget
    if ratio > 1:
        return f" — over budget by {(ratio - 1) * 100:.1f}%"
    if ratio > 0.8:
        return f" — close to budget ({ratio * 100:.1f}% used)"
    return ""


def format_credit_summary(summary: Mapping[str, Any]) -> Optional[str]:
    """Render only the concise credit information useful during iteration."""
    mean_credits = summary.get("mean_credits")
    if mean_credits is None:
        return None
    rendered = f"Agent model credits: mean={mean_credits:.4f}"
    budgets = summary.get("budgets")
    if budgets:
        for entry in budgets:
            rendered += (
                f"; {entry['id']}: mean={entry['mean_credits']:.4f}, "
                f"budget={entry['budget']:.4f}"
            )
            rendered += _format_budget_status(entry["mean_credits"], entry["budget"])
        return rendered
    budget = summary.get("budget")
    if budget is None:
        return rendered
    rendered += f", budget={budget:.4f}"
    return rendered + _format_budget_status(mean_credits, budget)


def evaluate_performance_requirements(
    requirements: list[PerformanceRequirement],
    messages: Iterable[Message],
    credit_usages: Iterable[Optional[Mapping[str, Any]]] = (),
    task_rewards: Optional[Sequence[float]] = None,
) -> dict:
    """Evaluate hard latency gates and soft set-wide mean credit budgets.

    A credit requirement carrying a ``models`` scope meters only credits
    incurred on those models (its tier's pool); requirements without a scope
    meter every credit. Overages from multiple credit requirements add up —
    tier scopes are disjoint, so each tier's overspend penalizes once.

    To prevent 0-token abort arbitrage (where an agent intentionally aborts a
    subset of conversations with 0.0 credits to dilute credit overages on
    other conversations), explicit 0-credit aborts on failed tasks are excluded
    from the effective overage denominator when other conversations exceed the
    budget.
    """
    latency_samples = collect_agent_turn_latencies(messages)
    credit_requirements = [
        requirement
        for requirement in requirements
        if isinstance(requirement, CreditPerformanceRequirement)
    ]
    usages = list(credit_usages) if credit_requirements else []
    credit_samples = collect_conversation_credits(usages) if credit_requirements else []
    rewards_list = list(task_rewards) if task_rewards is not None else None
    latency_summary = {
        "sample_count": len(latency_samples),
        "p50_seconds": _percentile(latency_samples, 50),
        "p90_seconds": _percentile(latency_samples, 90),
        "max_seconds": max(latency_samples) if latency_samples else None,
    }
    credit_summary = {
        "sample_count": len(credit_samples),
        "mean_credits": mean(credit_samples) if credit_samples else None,
        "budget": None,
    }

    details = []
    latency_gate_met = True
    credit_penalties = []
    budget_entries = []
    for requirement in requirements:
        if isinstance(requirement, LatencyPerformanceRequirement):
            observed = _percentile(latency_samples, requirement.percentile)
            met = observed is not None and observed <= requirement.max_seconds
            latency_gate_met = latency_gate_met and met
            details.append(
                {
                    "id": requirement.id,
                    "type": requirement.type,
                    "measurement": requirement.measurement,
                    "percentile": requirement.percentile,
                    "max_seconds": requirement.max_seconds,
                    "observed_seconds": observed,
                    "met": met,
                }
            )
            continue

        if requirement.models is None:
            model_scope = None
            requirement_samples = credit_samples
        else:
            model_scope = set(requirement.models)
            requirement_samples = collect_conversation_credits(usages, model_scope)
        observed_mean = mean(requirement_samples) if requirement_samples else None
        effective_mean = observed_mean
        if requirement_samples and any(c > requirement.budget for c in requirement_samples):
            # Exclude failed-task low/zero-credit aborts from diluting over-budget tasks.
            non_abort_samples = []
            abort_floor = 0.05 * requirement.budget
            for idx, usage_item in enumerate(usages):
                if usage_item is None:
                    continue
                sample_vals = collect_conversation_credits([usage_item], model_scope)
                if not sample_vals:
                    continue
                credit_val = sample_vals[0]
                passed_task = (
                    rewards_list is not None
                    and idx < len(rewards_list)
                    and rewards_list[idx] >= 1.0
                )
                is_failed_abort = (
                    isinstance(usage_item, Mapping)
                    and not passed_task
                    and (credit_val == 0.0 or (rewards_list is not None and credit_val < abort_floor))
                )
                if not is_failed_abort:
                    non_abort_samples.append(credit_val)
            if non_abort_samples:
                effective_mean = max(observed_mean or 0.0, mean(non_abort_samples))

        credit_summary["budget"] = requirement.budget
        if effective_mean is not None and observed_mean is not None and effective_mean > observed_mean:
            credit_summary["mean_credits"] = effective_mean
            observed_mean = effective_mean
        mean_overage = (
            max(0.0, observed_mean / requirement.budget - 1.0)
            if observed_mean is not None
            else 0.0
        )
        credit_penalties.append(mean_overage)
        detail = {
            "id": requirement.id,
            "type": requirement.type,
            "measurement": requirement.measurement,
            "budget": requirement.budget,
            "mean_credits": observed_mean,
            "mean_overage": mean_overage,
            "met": observed_mean is not None and observed_mean <= requirement.budget,
        }
        if requirement.tier is not None:
            detail["tier"] = requirement.tier
        budget_entries.append(
            {
                "id": requirement.id,
                "mean_credits": observed_mean,
                "budget": requirement.budget,
            }
        )
        details.append(detail)

    if len(credit_requirements) > 1:
        # Several budgets: the flat single-budget field is meaningless, so
        # report per budget. Entries key on the requirement id — the label a
        # kit's README defines — because sandbox-side requirements carry no
        # tier (the kit strips it) yet still need per-pool feedback.
        credit_summary["budget"] = None
        credit_summary["budgets"] = budget_entries

    requirements_met = all(detail["met"] for detail in details)
    return {
        "reward": 1.0 if latency_gate_met else 0.0,
        "penalty": sum(credit_penalties),
        "requirements_met": requirements_met,
        # Keep the original flat latency summary for artifact compatibility.
        "summary": latency_summary,
        "credit_summary": credit_summary,
        "requirements": details,
    }
