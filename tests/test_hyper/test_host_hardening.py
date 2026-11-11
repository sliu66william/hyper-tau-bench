"""Host hardening against a Developer who controls the kit and the callback
directory."""

import json
import time

import pytest

from tau2.hyper.live_experiment import LiveExperimentContext
from tau2.hyper.sandbox.callback_broker import CallbackBroker, _write_response_file
from tau2.hyper.sandbox.orchestrator import (
    _classify_workspace_provenance,
    initialize_kit_repository,
)


def _symlink_or_skip(link, target, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires elevated privileges on Windows")
        raise


def test_broker_response_write_rejects_a_symlink(tmp_path):
    target = tmp_path / "secret"
    target.write_text("keep")
    link = tmp_path / ".response-abc.tmp"
    _symlink_or_skip(link, target)
    with pytest.raises(OSError):
        _write_response_file(link, "attacker")
    assert target.read_text() == "keep"


def test_broker_response_write_rejects_an_existing_file(tmp_path):
    path = tmp_path / ".response-abc.tmp"
    path.write_text("old")
    with pytest.raises(OSError):
        _write_response_file(path, "new")


def test_broker_survives_planted_temp_response_and_serves_next_request(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    with CallbackBroker(kit, toolkit=object()) as broker:
        # Plant a colliding .response-bad.tmp file before submitting request-bad.json.
        (broker.callback_dir / ".response-bad.tmp").write_text("planted")
        (broker.callback_dir / "request-bad.json").write_text(
            json.dumps({"token": broker.token, "tool": "submit", "arguments": {}})
        )
        # Follow with a clean request to prove the broker daemon thread stayed alive.
        (broker.callback_dir / "request-good.json").write_text(
            json.dumps({"token": broker.token, "tool": "submit", "arguments": {}})
        )
        deadline = time.monotonic() + 3.0
        good_resp = broker.callback_dir / "response-good.json"
        while time.monotonic() < deadline and not good_resp.exists():
            time.sleep(0.05)
        assert good_resp.exists()
        assert json.loads(good_resp.read_text())["ok"] is True


def test_live_experiment_persist_rejects_symlinked_simulations_dir(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(kit / "simulations", outside, target_is_directory=True)

    ctx = LiveExperimentContext(lambda: '{"ok": true}', workspace_root=kit)
    output = ctx.run()
    assert output == '{"ok": true}'
    assert ctx.report_path is None
    assert list(outside.iterdir()) == []


def test_local_test_artifact_persist_rejects_symlinked_simulations_and_dangling_links(
    tmp_path,
):
    from tau2.hyper.sandbox.local_test import LocalTestService

    kit = tmp_path / "kit"
    kit.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(kit / "simulations", outside, target_is_directory=True)

    service = LocalTestService(kit)
    assert service._persist_local_test_artifact('{"ok": true}') is None
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "config",
    [
        '[core]\n\tfsmonitor = "touch {sentinel}; false"\n',
        '[filter "x"]\n\tclean = "touch {sentinel}; cat"\n',
        '[diff "x"]\n\ttextconv = "touch {sentinel}; cat"\n\tcommand = "touch {sentinel}"\n',
    ],
    ids=["fsmonitor", "clean-filter", "diff-driver"],
)
def test_host_git_runs_nothing_the_developer_configured(tmp_path, config):
    kit = tmp_path / "kit"
    (kit / "workspace").mkdir(parents=True)
    (kit / "workspace" / "agent.py").write_text("x = 1\n")
    (kit / "workspace" / "old.py").write_text("y = 1\n")
    baseline = initialize_kit_repository(kit)

    # The Developer, root in the mount, changes the workspace and plants a
    # program in the kit's git config, with attributes that route files to it.
    sentinel = tmp_path / "ran"
    (kit / "workspace" / "agent.py").write_text("x = 2\n")
    (kit / "workspace" / "old.py").unlink()
    (kit / "workspace" / "new.py").write_text("z = 1\n")
    (kit / ".gitattributes").write_text("workspace/* filter=x diff=x\n")
    with (kit / ".git" / "config").open("a") as handle:
        handle.write("\n" + config.format(sentinel=sentinel))

    provenance, deleted = _classify_workspace_provenance(kit, baseline)
    assert provenance == {"workspace/agent.py": "modified", "workspace/new.py": "new"}
    assert deleted == ["workspace/old.py"]
    assert not sentinel.exists()
