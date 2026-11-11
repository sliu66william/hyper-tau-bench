"""τ^τ-Verified: the fixed checks and the contract notes."""

import json

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import Action
from tau2.evaluator.evaluator_action import ActionEvaluator, _check_actions
from tau2.evaluator.evaluator_env import _hashes_for_compare
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.hyper.verified import (
    VERIFIED_ENV,
    verified_agent_contract,
    verified_enabled,
)
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.utils.utils import DATA_DIR


def test_verified_is_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv(VERIFIED_ENV, raising=False)
    assert verified_enabled()
    monkeypatch.setenv(VERIFIED_ENV, "0")
    assert not verified_enabled()


# --- Transfer reason --------------------------------------------------------


def _transfer_gold():
    # Banking task_008's gold: only the reason is compared.
    return Action(
        action_id="008_0",
        requestor="assistant",
        name="transfer_to_human_agents",
        arguments={
            "reason": "customer_demands_after_unavailable_offer_refusal",
            "summary": "",
        },
        compare_args=["reason"],
    )


def _transfer(summary, **arguments):
    # A Client API transfer replays with its summary only.
    return ToolCall(
        name="transfer_to_human_agents", arguments={"summary": summary, **arguments}
    )


@pytest.mark.parametrize(
    "summary, matches",
    [
        (
            "Reason: customer_demands_after_unavailable_offer_refusal. "
            "Asked again about the other offers.",
            True,
        ),
        ("reason_code=CUSTOMER_DEMANDS_AFTER_UNAVAILABLE_OFFER_REFUSAL; flyers", True),
        ("Customer kept asking for the mailed offers after I said no.", False),
        ("Reason: unconfirmed_external_communication. Flyer offers.", False),
        (
            "customer_demands_after_unavailable_offer_refusal or "
            "unconfirmed_external_communication",
            False,
        ),
    ],
)
def test_transfer_reason_comes_from_the_summary(monkeypatch, summary, matches):
    monkeypatch.setenv(VERIFIED_ENV, "1")
    [check] = _check_actions([_transfer(summary)], [_transfer_gold()])
    assert check.action_match is matches


def test_transfer_reason_check_is_unchanged_without_the_flag(monkeypatch):
    monkeypatch.setenv(VERIFIED_ENV, "0")
    summary = "Reason: customer_demands_after_unavailable_offer_refusal."
    [check] = _check_actions([_transfer(summary)], [_transfer_gold()])
    assert not check.action_match


def test_an_explicit_transfer_reason_is_kept(monkeypatch):
    monkeypatch.setenv(VERIFIED_ENV, "1")
    summary = "Reason: customer_demands_after_unavailable_offer_refusal."
    [check] = _check_actions([_transfer(summary, reason="other")], [_transfer_gold()])
    assert not check.action_match


def test_multiple_transfer_calls_cannot_enumerate_reason_codes(monkeypatch):
    monkeypatch.setenv(VERIFIED_ENV, "1")
    calls = [
        _transfer("Reason: account_ownership_dispute"),
        _transfer("Reason: customer_demands_after_unavailable_offer_refusal"),
    ]
    [check] = _check_actions(calls, [_transfer_gold()])
    assert not check.action_match


def test_rest_semantic_traces_ignore_forged_local_assistant_tool_calls(monkeypatch):
    monkeypatch.setenv(VERIFIED_ENV, "1")
    forged_call = ToolCall(
        id="tc_1",
        name="transfer_to_human_agents",
        arguments={"reason": "customer_demands_after_unavailable_offer_refusal"},
    )
    # Candidate wrapper ran locally without hitting ClientAPIRuntime -> semantic_tool_calls=[].
    traj_forged = [
        AssistantMessage(role="assistant", content=None, tool_calls=[forged_call]),
        ToolMessage(id="tc_1", role="tool", content="ok", semantic_tool_calls=[]),
    ]
    [forged_check] = ActionEvaluator.evaluate_actions(traj_forged, [_transfer_gold()])
    assert not forged_check.action_match
    rendered_forged = NLAssertionsEvaluator._format_trajectory_for_judge(traj_forged)
    assert "customer_demands_after_unavailable_offer_refusal" not in rendered_forged

    # Genuine REST transfer attaches host-verified semantic_tool_calls.
    verified_sem = ToolCall(
        id="tc_2:client-api:0",
        name="transfer_to_human_agents",
        arguments={
            "summary": "Reason: customer_demands_after_unavailable_offer_refusal"
        },
    )
    traj_real = [
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="tc_2", name="escalate_chat", arguments={})],
        ),
        ToolMessage(
            id="tc_2",
            role="tool",
            content='{"status": "accepted"}',
            semantic_tool_calls=[verified_sem],
        ),
    ]
    assert [c.name for c in ActionEvaluator.extract_tool_calls(traj_real)] == [
        "escalate_chat",
        "transfer_to_human_agents",
    ]
    [real_check] = ActionEvaluator.evaluate_actions(traj_real, [_transfer_gold()])
    assert real_check.action_match
    rendered_real = NLAssertionsEvaluator._format_trajectory_for_judge(traj_real)
    assert "customer_demands_after_unavailable_offer_refusal" in rendered_real


def test_openings_extractor_strips_filler_and_includes_body_after_short_preamble(
    monkeypatch,
):
    from tau2.evaluator.evaluator_nl_assertions import _assistant_response_openings
    from tau2.evaluator.evaluator_response_assertions import _normalized_opening_key

    monkeypatch.setenv(VERIFIED_ENV, "1")
    traj = [
        AssistantMessage(
            role="assistant",
            content="Right now, first. I have completed the closure for your EcoCard.",
        ),
        AssistantMessage(
            role="assistant",
            content="Thank you for waiting today. I have completed the closure for your Gold Card.",
        ),
    ]
    openings = _assistant_response_openings(traj)
    assert openings[0] == "I have completed the closure for your EcoCard."
    assert "I have completed the closure for your Gold Card." in openings[1]
    assert (
        _normalized_opening_key("Step 1: I have completed the closure.", 3)
        == "i have completed"
    )


def test_openings_system_prompt_honors_exemptions_under_verified(monkeypatch):
    from tau2.evaluator.evaluator_nl_assertions import _response_openings_system_prompt

    monkeypatch.setenv(VERIFIED_ENV, "1")
    assert "HONOR ASSERTION EXEMPTIONS" in _response_openings_system_prompt()
    monkeypatch.setenv(VERIFIED_ENV, "0")
    assert "HONOR ASSERTION EXEMPTIONS" not in _response_openings_system_prompt()


# --- Discoverable-tool audit row ---------------------------------------------


def _fresh_banking(case=None):
    env = registry.get_env_constructor("banking_knowledge")(
        retrieval_variant="no_knowledge"
    )
    state = case.initial_state if case is not None else None
    env.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=state.initialization_actions if state else None,
        message_history=[],
    )
    return env


def _banking_environments(case_id):
    """Gold replayed as the scorer does, and the same calls as Client REST runs them.

    REST skips the unlock and calls the discovered tool directly, so it writes
    no row to the agent_discoverable_tools audit table.
    """
    case = next(t for t in get_tasks("banking_knowledge") if t.id == case_id)
    gold, rest = _fresh_banking(case), _fresh_banking(case)
    for action in case.evaluation_criteria.actions:
        arguments = dict(action.arguments or {})
        gold.make_tool_call(action.name, requestor=action.requestor, **arguments)
        if action.name == "unlock_discoverable_agent_tool":
            continue
        if action.name == "call_discoverable_agent_tool":
            tool = rest.tools.get_discoverable_tools()[arguments["agent_tool_name"]]
            tool(**json.loads(arguments["arguments"]))
            rest.sync_tools()
            continue
        rest.make_tool_call(action.name, requestor=action.requestor, **arguments)
    gold.sync_tools()
    return gold, rest


def test_rest_writes_match_gold_only_with_the_flag(monkeypatch):
    gold, rest = _banking_environments("task_028")
    monkeypatch.setenv(VERIFIED_ENV, "0")
    agent_gold, agent_rest, user_gold, user_rest = _hashes_for_compare(
        rest, gold, normalize=True
    )
    assert agent_gold == agent_rest
    assert user_gold != user_rest
    monkeypatch.setenv(VERIFIED_ENV, "1")
    agent_gold, agent_rest, user_gold, user_rest = _hashes_for_compare(
        rest, gold, normalize=True
    )
    assert (agent_gold, user_gold) == (agent_rest, user_rest)


def test_the_flag_still_catches_a_missing_write(monkeypatch):
    gold, _ = _banking_environments("task_028")
    nothing = _fresh_banking(
        next(t for t in get_tasks("banking_knowledge") if t.id == "task_028")
    )
    monkeypatch.setenv(VERIFIED_ENV, "1")
    agent_gold, agent_none, user_gold, user_none = _hashes_for_compare(
        nothing, gold, normalize=True
    )
    assert agent_gold != agent_none and user_gold != user_none


# --- Agent contract --------------------------------------------------------


def test_contract_names_the_runtime_facts():
    source = DATA_DIR / "tau2" / "hyper" / "framework_reference" / "agent_contract.md"
    text = verified_agent_contract(source.read_text())
    assert "or a `MultiToolMessage` (results" not in text
    for fact in (
        "`ToolMessage`",
        "parallel_tool_calls",
        "five errors",
        "200 steps",
        "300 seconds",
        "32 model calls",
        "256",
        "selected reason code verbatim",
    ):
        assert fact in text


def test_contract_lists_exactly_the_sealed_runner_arguments():
    from tau2.hyper.sandbox.sealed_runner import _SAFE_CANDIDATE_MODEL_KWARGS

    source = DATA_DIR / "tau2" / "hyper" / "framework_reference" / "agent_contract.md"
    text = verified_agent_contract(source.read_text())
    section = text.split("### Model call arguments", 1)[1].split("###", 1)[0]
    section = " ".join(section.split())
    accepted = section.split("own constraints:", 1)[1].split("Any other argument", 1)[0]
    listed = set(accepted.split("`")[1::2])
    assert listed == set(_SAFE_CANDIDATE_MODEL_KWARGS)
