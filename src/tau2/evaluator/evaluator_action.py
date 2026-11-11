from functools import lru_cache
import re
from typing import Optional, get_args

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.environment.toolkit import ToolType
from tau2.evaluator.evaluator_base import EvaluatorBase

TRANSFER_TOOL = "transfer_to_human_agents"


@lru_cache(maxsize=1)
def _canonical_client_api_operation_names() -> frozenset[str]:
    """Return canonical reference action names backed by Client REST operations."""
    names: set[str] = {TRANSFER_TOOL}
    try:
        from tau2.hyper.client_api.catalogs import airline, banking, retail, telecom

        for mod in (airline, banking, retail, telecom):
            for op in mod.operations():
                names.add(op.operation_id)
                names.update(op.reference_tool_names)
    except Exception:
        pass
    return frozenset(names)


def _reason_from_summary(gold_action: Action) -> bool:
    """Whether a transfer's reason is read from its summary (τ^τ-Verified).

    The Client API transfer body accepts only `summary`, so a replayed transfer
    carries no `reason`, and gold that compares `reason` could never match.
    """
    if gold_action.name != TRANSFER_TOOL:
        return False
    if "reason" not in (gold_action.compare_args or []):
        return False
    from tau2.hyper.verified import verified_enabled

    return verified_enabled()


def _with_summary_reason(tool_call: ToolCall) -> ToolCall:
    """Fill a transfer's missing `reason` with the one reason code in its summary.

    No code, or several distinct codes, leave the call without a reason, so
    listing every code cannot match. `other` is not searched for, because it
    is also an ordinary word.
    """
    if tool_call.name != TRANSFER_TOOL or "reason" in tool_call.arguments:
        return tool_call
    from tau2.domains.banking_knowledge.tools import TransferReasonLiteral

    summary = str(tool_call.arguments.get("summary", ""))
    codes = [code for code in get_args(TransferReasonLiteral) if code != "other"]
    found = {
        code
        for code in codes
        if re.search(rf"(?<![a-z0-9_]){code}(?![a-z0-9_])", summary, re.IGNORECASE)
    }
    if len(found) != 1:
        return tool_call
    return tool_call.model_copy(
        update={"arguments": {**tool_call.arguments, "reason": found.pop()}}
    )


def _check_actions(
    predicted_tool_calls: list[ToolCall],
    golden_actions: list[Action],
    tool_types: Optional[dict[str, ToolType]] = None,
) -> list[ActionCheck]:
    """
    Check if all the gold actions are in the predicted tool calls.

    Args:
        predicted_tool_calls: List of tool calls made during the simulation.
        golden_actions: List of expected actions from evaluation criteria.
        tool_types: Optional mapping of tool names to their types (read/write/think/generic).

    Returns:
        List of ActionCheck results for each golden action.
    """
    action_checks = []
    for gold_action in golden_actions:
        candidates = predicted_tool_calls
        if _reason_from_summary(gold_action):
            transfer_calls = [
                call for call in predicted_tool_calls if call.name == TRANSFER_TOOL
            ]
            # Multiple transfer calls cannot be used to enumerate reason codes.
            candidates = (
                [_with_summary_reason(call) for call in predicted_tool_calls]
                if len(transfer_calls) <= 1
                else []
            )
        found = False
        for pred_tool_call in candidates:
            if gold_action.compare_with_tool_call(pred_tool_call):
                found = True
                break
        if found:
            gold_action_reward = 1.0
            gold_action_match = True
        else:
            gold_action_reward = 0.0
            gold_action_match = False

        # Get tool type if available
        tool_type = None
        if tool_types is not None:
            tool_type = tool_types.get(gold_action.name)

        action_checks.append(
            ActionCheck(
                action=gold_action,
                action_match=gold_action_match,
                action_reward=gold_action_reward,
                tool_type=tool_type,
            )
        )
    return action_checks


class ActionEvaluator(EvaluatorBase[Message]):
    """
    Evaluates whether or not the agent performed the required actions.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
        tool_types: Optional[dict[str, ToolType]] = None,
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent performed the required actions.

        Args:
            task: The task containing evaluation criteria.
            full_trajectory: List of messages from the simulation.
            tool_types: Optional mapping of tool names to their types (read/write/think/generic).
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(
            full_trajectory, golden_actions, tool_types
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.action_match for result in action_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
        )

    @classmethod
    def extract_tool_calls(cls, full_trajectory: list[Message]) -> list[ToolCall]:
        """
        Extract all tool calls from a message trajectory.

        Returns:
            List of ToolCall objects extracted from AssistantMessage, UserMessage,
            and host-attached ToolMessage.semantic_tool_calls.
        """
        tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
                or isinstance(message, UserMessage)
            ) and message.is_tool_call():
                tool_calls.extend(message.tool_calls)
            if isinstance(message, ToolMessage):
                tool_calls.extend(message.semantic_tool_calls or [])
        return tool_calls

    @classmethod
    def _extract_verified_semantic_calls(
        cls, full_trajectory: list[Message]
    ) -> list[ToolCall]:
        """Extract only UserMessage calls and host-verified semantic_tool_calls."""
        tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if isinstance(message, UserMessage) and message.is_tool_call():
                tool_calls.extend(message.tool_calls)
            if isinstance(message, ToolMessage):
                tool_calls.extend(message.semantic_tool_calls or [])
        return tool_calls

    @classmethod
    def evaluate_actions(
        cls,
        full_trajectory: list[Message],
        golden_actions: list[Action],
        tool_types: Optional[dict[str, ToolType]] = None,
    ) -> list[ActionCheck]:
        """
        Evaluate whether the agent performed the expected actions.

        In τ^τ-Verified REST trajectories (carrying ``semantic_tool_calls``),
        canonical Client API actions (such as ``transfer_to_human_agents``)
        must match host-verified ``semantic_tool_calls`` rather than untrusted
        candidate wrapper names on ``AssistantMessage.tool_calls``.
        """
        if len(golden_actions) == 0:
            return []

        from tau2.hyper.verified import verified_enabled

        has_semantic_trace = verified_enabled() and any(
            isinstance(m, ToolMessage) and m.semantic_tool_calls is not None
            for m in full_trajectory
        )
        if not has_semantic_trace:
            return _check_actions(
                cls.extract_tool_calls(full_trajectory), golden_actions, tool_types
            )

        all_calls = cls.extract_tool_calls(full_trajectory)
        verified_calls = cls._extract_verified_semantic_calls(full_trajectory)
        canonical_op_names = _canonical_client_api_operation_names()
        checks: list[ActionCheck] = []
        for gold_action in golden_actions:
            pool = (
                verified_calls
                if gold_action.name in canonical_op_names
                else all_calls
            )
            checks.extend(_check_actions(pool, [gold_action], tool_types))
        return checks


class FullDuplexActionEvaluator(EvaluatorBase[Tick]):
    """
    Evaluates whether or not the agent performed the required actions (full-duplex mode).
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Tick],
        tool_types: Optional[dict[str, ToolType]] = None,
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent performed the required actions.

        Args:
            task: The task containing evaluation criteria.
            full_trajectory: List of Tick objects from full-duplex simulation.
            tool_types: Optional mapping of tool names to their types (read/write/think/generic).
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(
            full_trajectory, golden_actions, tool_types
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.action_match for result in action_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
        )

    @classmethod
    def extract_tool_calls(cls, full_trajectory: list[Tick]) -> list[ToolCall]:
        """
        Extract all tool calls from a tick trajectory.

        Args:
            full_trajectory: List of Tick objects from full-duplex simulation.

        Returns:
            List of ToolCall objects (user tool calls first, then agent tool calls per tick).
        """
        tool_calls: list[ToolCall] = []
        for tick in full_trajectory:
            tool_calls.extend(tick.user_tool_calls)
            tool_calls.extend(tick.agent_tool_calls)
            for result in (*tick.user_tool_results, *tick.agent_tool_results):
                tool_calls.extend(result.semantic_tool_calls or [])
        return tool_calls

    @classmethod
    def _extract_verified_semantic_calls(
        cls, full_trajectory: list[Tick]
    ) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for tick in full_trajectory:
            tool_calls.extend(tick.user_tool_calls)
            for result in (*tick.user_tool_results, *tick.agent_tool_results):
                tool_calls.extend(result.semantic_tool_calls or [])
        return tool_calls

    @classmethod
    def evaluate_actions(
        cls,
        full_trajectory: list[Tick],
        golden_actions: list[Action],
        tool_types: Optional[dict[str, ToolType]] = None,
    ) -> list[ActionCheck]:
        """
        Evaluate whether the agent performed the expected actions.

        Args:
            full_trajectory: List of Tick objects from full-duplex simulation.
            golden_actions: List of expected actions from evaluation criteria.
            tool_types: Optional mapping of tool names to their types (read/write/think/generic).
        """
        if len(golden_actions) == 0:
            return []

        from tau2.hyper.verified import verified_enabled

        has_semantic_trace = verified_enabled() and any(
            result.semantic_tool_calls is not None
            for tick in full_trajectory
            for result in (*tick.user_tool_results, *tick.agent_tool_results)
        )
        if not has_semantic_trace:
            return _check_actions(
                cls.extract_tool_calls(full_trajectory), golden_actions, tool_types
            )

        all_calls = cls.extract_tool_calls(full_trajectory)
        verified_calls = cls._extract_verified_semantic_calls(full_trajectory)
        canonical_op_names = _canonical_client_api_operation_names()
        checks: list[ActionCheck] = []
        for gold_action in golden_actions:
            pool = (
                verified_calls
                if gold_action.name in canonical_op_names
                else all_calls
            )
            checks.extend(_check_actions(pool, [gold_action], tool_types))
        return checks
