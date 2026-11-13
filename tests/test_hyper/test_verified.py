"""τ^τ-Verified: the fixed checks and the contract notes."""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Action
from tau2.evaluator.evaluator_action import _check_actions
from tau2.evaluator.evaluator_env import _hashes_for_compare
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


def test_openings_system_prompt_honors_exemptions_under_verified(monkeypatch):
    from tau2.evaluator.evaluator_nl_assertions import _response_openings_system_prompt

    monkeypatch.setenv(VERIFIED_ENV, "1")
    assert "HONOR ASSERTION EXEMPTIONS" in _response_openings_system_prompt()
    monkeypatch.setenv(VERIFIED_ENV, "0")
    assert "HONOR ASSERTION EXEMPTIONS" not in _response_openings_system_prompt()


def test_banking_debit_card_rest_projection_preserves_diagnostic_fields(monkeypatch):
    from tau2.hyper.client_api.catalogs.banking import DebitCard, _debit_card

    monkeypatch.setenv(VERIFIED_ENV, "1")
    projected = _debit_card(
        "dc_101",
        {
            "card_id": "dc_101",
            "account_id": "chk_1",
            "last_4_digits": "4321",
            "expiration_date": "09/28",
            "status": "ACTIVE",
            "issue_reason": "initial",
            "issue_date": "2024-01-15",
            "pin_locked": True,
            "pin_lock_reason": "consecutive_incorrect_attempts",
            "fraud_alert_active": True,
            "velocity_blocked": False,
            "chip_verification_failures": 3,
            "daily_purchase_limit": "$2,500.00",
        },
    )
    validated = DebitCard.model_validate(projected)
    assert validated.pin_locked is True
    assert validated.fraud_alert_active is True
    assert validated.velocity_blocked is False
    assert validated.chip_verification_failures == 3
    assert validated.daily_purchase_limit == 2500.0
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


@pytest.mark.parametrize(
    ("summary", "expected_reason"),
    [
        ("Transferring: reason: other (outside standard card flow)", "other"),
        ('{"reason": "other", "notes": "edge case"}', "other"),
        ("Transfer reason code: other", "other"),
        ("Escalating with transfer_reason=other", "other"),
        ("Customer has some other question about rates", None),
        (
            "Ruled out fraud_or_security_concern and complex_billing_dispute; "
            "escalating under account_ownership_dispute.",
            "account_ownership_dispute",
        ),
    ],
)
def test_with_summary_reason_handles_explicit_other_json_and_exclusions(
    summary, expected_reason
):
    from tau2.data_model.message import ToolCall
    from tau2.evaluator.evaluator_action import _with_summary_reason

    call = ToolCall(
        id="t1",
        name="transfer_to_human_agents",
        arguments={"summary": summary},
    )
    updated = _with_summary_reason(call)
    assert updated.arguments.get("reason") == expected_reason

