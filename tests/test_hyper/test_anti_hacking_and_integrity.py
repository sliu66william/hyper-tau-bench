"""Anti-hacking and sample-by-sample scientific integrity regression suite for Hyper-τ."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.evaluator.evaluator_nl_assertions import (
    NLAssertionsEvaluator,
    RESPONSE_OPENINGS_SYSTEM_PROMPT,
    _assistant_response_openings,
)
from tau2.evaluator.evaluator_response_assertions import _normalized_opening_key
from tau2.hyper.agent_context import build_agent_context
from tau2.hyper.client_api.defects import (
    ClientAPIDeploymentManifest,
    compile_defect_profile,
)
from tau2.hyper.client_sim.api_defect_overlay import render_api_defect_client_overlay
from tau2.hyper.data_model import CreditPerformanceRequirement
from tau2.hyper.performance import evaluate_performance_requirements
from tau2.hyper.sandbox.local_test import LocalTestService
from tau2.hyper.sandbox.orchestrator import (
    _classify_workspace_provenance,
    initialize_kit_repository,
)
from tau2.hyper.sandbox.sealed_runner import _ensure_kit_world_readable

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_h1_local_test_service_rejects_deployment_manifest_tampering(tmp_path: Path):
    """H1: Developer edits to /workspace/framework/deployment_manifest.json must not
    escalate allowed_agent_models, strip constraints, or zero out credit_rates."""
    kit_root = tmp_path / "kit"
    (kit_root / "framework").mkdir(parents=True)
    canonical_manifest = {
        "allowed_agent_models": [
            {
                "model": "gpt-5.4-nano",
                "constraints": {"reasoning_effort": "none"},
                "credit_rates": {"input_per_million": 0.25, "output_per_million": 1.6},
            }
        ],
        "performance_requirements": [
            {
                "id": "agent_credit_budget",
                "type": "credits",
                "measurement": "agent_model_calls",
                "budget": 0.05,
            }
        ],
    }
    manifest_file = kit_root / "framework" / "deployment_manifest.json"
    manifest_file.write_text(json.dumps(canonical_manifest, indent=2), encoding="utf-8")

    service = LocalTestService(kit_root)

    # Simulate untrusted Developer container overwriting deployment_manifest.json
    tampered_manifest = {
        "allowed_agent_models": [
            {
                "model": "gpt-5.4",
                "constraints": {},
                "credit_rates": {"input_per_million": 0.0, "output_per_million": 0.0},
            }
        ],
        "performance_requirements": [],
    }
    manifest_file.write_text(json.dumps(tampered_manifest, indent=2), encoding="utf-8")

    loaded = service._load_developer_manifest()
    assert loaded == canonical_manifest
    ctx = build_agent_context(
        domain="banking_knowledge",
        tools=[],
        resource_root=kit_root,
        model_configs=loaded["allowed_agent_models"],
    )
    assert ctx.model_gateway.available_models == ("gpt-5.4-nano",)


def test_h2_workspace_provenance_blocks_git_config_filter_rce_and_symlinks(
    tmp_path: Path,
):
    """H2: Host provenance inspection must not execute .git/config filter/fsmonitor
    payloads on the host OS or follow symlinks outside kit_path."""
    kit_root = tmp_path / "kit"
    (kit_root / "workspace").mkdir(parents=True)
    (kit_root / "workspace" / "agent.py").write_text("# initial stub 1\n", encoding="utf-8")
    rce_proof_file = (tmp_path / "HOST_RCE_PROOF.txt").resolve()

    baseline_commit = initialize_kit_repository(kit_root)

    # Untrusted container injects filter.rce.clean and core.fsmonitor into .git/config
    posix_proof = rce_proof_file.as_posix()
    subprocess.run(
        [
            "git",
            "config",
            "filter.rce.clean",
            f"sh -c 'echo PWNED > {posix_proof}; cat'",
        ],
        cwd=kit_root,
        check=True,
    )
    (kit_root / ".git" / "info" / "attributes").write_text(
        "* filter=rce\n", encoding="utf-8"
    )
    time.sleep(0.02)
    (kit_root / "workspace" / "agent.py").write_text("# initial stub 2\n", encoding="utf-8")

    provenance, deleted = _classify_workspace_provenance(kit_root, baseline_commit)
    assert provenance == {"workspace/agent.py": "modified"}
    assert deleted == []
    assert not rce_proof_file.exists(), "Host RCE payload in .git/config was executed!"

    # Also test symlink guard in _ensure_kit_world_readable
    outside_target = tmp_path / "outside_secret.txt"
    outside_target.write_text("secret", encoding="utf-8")
    _ensure_kit_world_readable(kit_root)


def test_h3_strip_runtime_src_removes_evaluator_and_user_simulator():
    """H3: Construction runtime image strip script must remove evaluator and
    user_simulator sources, and NLAssertionsEvaluator must render tool_calls."""
    strip_path = REPO_ROOT / "docker" / "hyper-construction" / "strip_runtime_src.py"
    spec = importlib.util.spec_from_file_location("strip_runtime_src", strip_path)
    strip_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strip_mod)

    assert "evaluator" in strip_mod.REMOVE_OUTSIDE_HYPER
    assert "user" in strip_mod.REMOVE_OUTSIDE_HYPER
    assert "data/tau2/user_simulator" in strip_mod.REMOVE_PROJECT_ROOT

    # Verify NLAssertionsEvaluator._format_trajectory_message includes tool_calls
    msg = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                name="transfer_to_human_agents",
                arguments={"summary": "Customer requested supervisor escalation"},
            )
        ],
    )
    rendered = NLAssertionsEvaluator._format_trajectory_message(msg)
    assert "transfer_to_human_agents" in rendered
    assert "Customer requested supervisor escalation" in rendered


def test_h4_credit_budget_prevents_zero_token_abort_arbitrage():
    """H4: Explicitly aborting failed tasks with 0.0 credits must not subsidize
    over-budget tasks."""
    req = [
        CreditPerformanceRequirement(
            id="banking_hard_budget",
            type="credits",
            measurement="agent_model_calls",
            budget=0.05,
        )
    ]
    honest_usages = [{"total_credits": 0.11} for _ in range(20)]
    honest_rewards = [1.0] * 3 + [0.0] * 17
    honest_perf = evaluate_performance_requirements(
        req, [], honest_usages, task_rewards=honest_rewards
    )

    arb_usages = [{"total_credits": 0.11} for _ in range(8)] + [
        {"total_credits": 0.0} for _ in range(12)
    ]
    arb_rewards = [1.0] * 3 + [0.0] * 17
    arb_perf = evaluate_performance_requirements(
        req, [], arb_usages, task_rewards=arb_rewards
    )

    assert arb_perf["penalty"] >= honest_perf["penalty"] - 1e-9
    assert arb_perf["penalty"] > 0.0


def test_h5_client_defect_overlay_gates_untested_defect_fishing():
    """H5: When has_run_local_test is False, the Client overlay instructs the
    Client Simulator not to confirm runtime API defects."""
    manifest = ClientAPIDeploymentManifest.model_validate(
        {
            "id": "retail_plus/client-aware-v1",
            "version": 1,
            "domain": "retail_plus",
            "defects": [
                {
                    "id": "refund_sign",
                    "kind": "response_amount_sign",
                    "operation_id": "getOrder",
                    "activation": {"task_ids": ["1"]},
                    "collection_path": ["payments"],
                    "discriminator_field": "transaction_type",
                    "discriminator_value": "refund",
                    "amount_field": "amount",
                    "sign": "negative",
                }
            ],
            "client": {
                "published_api_version": "2026-07",
                "deployed_api_version": "2026-08",
                "defects": [
                    {
                        "defect_id": "refund_sign",
                        "actual_behavior": "getOrder returns negative refund amounts.",
                        "disclosure_conditions": ["The Developer reports getOrder."],
                        "expected_remediation": "Accept negative refund amounts.",
                        "client_can_deploy_fix": False,
                    }
                ],
            },
        }
    )
    profile = compile_defect_profile(manifest)
    overlay_untested = render_api_defect_client_overlay(
        profile, has_run_local_test=False
    )
    assert "0 local test runs" in overlay_untested
    assert "run a local test first" in overlay_untested


def test_h6_openings_judge_strips_3_word_filler_prefixes_and_honors_exemptions():
    """H6: 3-word stage-setting filler prefixes ('At this stage.', 'For this item,')
    must be stripped before extracting openings and 3-token keys."""
    trajectory = [
        UserMessage(role="user", content="Check my two cards"),
        AssistantMessage(
            role="assistant",
            content="Right now, first. I have completed the closure for your Green Rewards Card.",
        ),
        AssistantMessage(
            role="assistant",
            content="At this stage, I have completed the closure for your EcoCard.",
        ),
    ]
    openings = _assistant_response_openings(trajectory)
    assert openings == [
        "I have completed the closure for your Green Rewards Card.",
        "I have completed the closure for your EcoCard.",
    ]
    key1 = _normalized_opening_key(trajectory[1].content, 3)
    key2 = _normalized_opening_key(trajectory[2].content, 3)
    assert key1 == key2 == "i have completed"
    assert "HONOR ASSERTION EXEMPTIONS" in RESPONSE_OPENINGS_SYSTEM_PROMPT


def test_all_67_airline_plus_tasks_enforce_nl_assertions_in_reward_basis():
    """No airline_plus task with nl_assertions may omit NL_ASSERTION from reward_basis."""
    tasks = json.loads(
        (REPO_ROOT / "data" / "tau2" / "domains" / "airline_plus" / "tasks.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(tasks) == 67
    for task in tasks:
        ec = task["evaluation_criteria"]
        assert ec.get("nl_assertions"), f"airline_plus task {task['id']} missing nl_assertions"
        assert "NL_ASSERTION" in ec["reward_basis"], (
            f"airline_plus task {task['id']} missing NL_ASSERTION in reward_basis"
        )


def test_s1_banking_client_simulator_prompts_rewrite_legacy_tool_names_to_rest():
    """S1: Banking Client Simulator prompts in REST mode must use REST HTTP contracts
    rather than legacy _NNNN Python tool names."""
    from tau2.hyper.client_api.banking_docs import _operation_index
    from tau2.hyper.client_sim.instructions import resolve_task_client_instructions
    from tau2.hyper.task_loader import load_active_hyper_tau_tasks

    op_index = _operation_index()
    for task in load_active_hyper_tau_tasks():
        if task.source_domain == "banking_knowledge" and task.client_sections:
            prompt = resolve_task_client_instructions(task)
            leaked_legacy = [name for name in op_index if name in prompt]
            assert not leaked_legacy, (
                f"{task.id} Client prompt still contains legacy tool names: {leaked_legacy}"
            )


def test_s2_banking_binary_artifacts_append_discovered_rest_contracts(tmp_path: Path):
    """S2: Binary PDF/PNG artifacts in banking_knowledge (flowcharts, reference PDFs,
    screenshots) must surface their discovered REST operation contracts into sop.md."""
    from tau2.hyper.client_api.banking_docs import (
        append_binary_artifact_banking_contracts,
    )
    from tau2.hyper.task_loader import load_hyper_tau_task

    task = load_hyper_tau_task(
        "019_banking_knowledge_construction_evidence_corpus_hard_live_experiment"
    )
    manifest = json.loads(
        (REPO_ROOT / "data" / task.sop_variant_manifest_path).read_text(encoding="utf-8")
    )
    sop_path = tmp_path / "sop.md"
    sop_path.write_text("# Handbook\n", encoding="utf-8")
    ops = append_binary_artifact_banking_contracts(manifest, sop_path)
    routes = {f"{op.method} {op.path}" for op in ops}
    assert "POST /v1/customers/{customer_id}/bank-accounts" in routes
    assert "POST /v1/debit-cards/{card_id}/activation" in routes
    assert "POST /v1/checking-accounts/{account_id}/debit-card-orders" in routes
    assert "POST /v1/debit-cards/{card_id}/closure" in routes
    assert "PATCH /v1/credit-card-transactions/{transaction_id}/rewards" in routes
    assert "POST /v1/checking-accounts/{account_id}/credits" in routes
    assert "POST /v1/savings-accounts/{account_id}/credits" in routes
    assert "POST /v1/savings-accounts/{account_id}/interest-discrepancy-reports" in routes
    sop_text = sop_path.read_text(encoding="utf-8")
    assert "## Referenced API contracts (diagram & print materials)" in sop_text


def test_s3_seeded_dsec_workspace_does_not_leak_held_temp_limit_facts():
    """S3: Task 052's seeded starting workspace must not pre-bake the Client-held
    temporary debit-card limit increase eligibility rules."""
    agent_py = (
        REPO_ROOT
        / "data/tau2/hyper/workspaces/banking_knowledge/gen_sonnet5_stateless_dsec_medium_001/tree/agent.py"
    ).read_text(encoding="utf-8")
    assert "150 percent of the current limit" not in agent_py
    assert "no overdraft fees in the last 30 days" not in agent_py

