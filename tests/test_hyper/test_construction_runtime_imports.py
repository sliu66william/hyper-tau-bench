"""The sealed candidate runtime must import without tau2.domains.

The construction image (docker/hyper-construction/Dockerfile) deletes
``src/tau2/domains`` and replaces the package ``__init__`` with a minimal
one; ``candidate_server`` is then imported inside that image. Any module in
its import closure that pulls ``tau2.domains`` at import time breaks sealed
scoring for every construction task (observed 2026-08-21 when
``client_api.catalog`` grew top-level domain imports).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STRIP_SCRIPT = REPO_ROOT / "docker" / "hyper-construction" / "strip_runtime_src.py"

_MINIMAL_INIT = '"""Minimal tau2 package init for construction sandbox runtime."""\n'


def test_candidate_server_imports_in_a_domainless_runtime(tmp_path):
    stripped_src = tmp_path / "src"
    shutil.copytree(
        REPO_ROOT / "src" / "tau2",
        stripped_src / "tau2",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.md"),
    )
    strip_res = subprocess.run(
        [sys.executable, str(STRIP_SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert strip_res.returncode == 0, strip_res.stdout + strip_res.stderr
    assert not (stripped_src / "tau2" / "evaluator").exists()
    assert not (stripped_src / "tau2" / "user").exists()
    assert not (stripped_src / "tau2" / "knowledge").exists()

    sub_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(stripped_src),
        "PYTHONUTF8": "1",
        # Keep the interpreter from resolving the repo's real data dir;
        # importing must not need any data at all.
        "TAU2_DATA_DIR": str(tmp_path / "no-data"),
    }
    for win_key in ("SYSTEMROOT", "TEMP", "TMP"):
        if win_key in os.environ:
            sub_env[win_key] = os.environ[win_key]

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tau2.hyper.agent_context, tau2.hyper.client_api, "
                "tau2.hyper.runtime_contract, tau2.hyper.sandbox.candidate_server, "
                "tau2.hyper.sandbox.callback_mcp, tau2.hyper.sandbox.model_gateway, "
                "tau2.hyper.harnesses.codex_driver, tau2.hyper.harnesses.prompt_arg_driver, "
                "tau2.hyper.harnesses.turn_loop_driver, tau2.agent.llm_agent; "
                "from pathlib import Path; "
                "from tau2.data_model.message import AssistantMessage, UserMessage; "
                "import tau2.hyper.agent_context as ac; "
                "ac.generate = lambda **kw: AssistantMessage(role='assistant', content='ok'); "
                "ctx = ac.build_agent_context(domain='telecom', tools=[], resource_root=Path('.'), "
                "model_configs=[{'model': 'gpt-5.4-nano', 'constraints': {'reasoning_effort': 'none'}}]); "
                "ctx.model_gateway.generate(model='gpt-5.4-nano', messages=[UserMessage(role='user', content='hi')]); "
                "print('import ok')"
            ),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=sub_env,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "import ok" in result.stdout
