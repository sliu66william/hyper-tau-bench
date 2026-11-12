# Copyright Sierra
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline_plus.tools import AirlinePlusTools
from tau2.domains.airline_plus.utils import (
    AIRLINE_PLUS_DB_PATH,
    AIRLINE_PLUS_POLICY_PATH,
    AIRLINE_PLUS_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline+ domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_PLUS_DB_PATH)
    tools = AirlinePlusTools(db)
    with open(AIRLINE_PLUS_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="airline_plus",
        policy=policy,
        tools=tools,
    )


_ZERO_WRITE_VERIFIED_AIRLINE_IDS = frozenset(
    {
        "0", "1", "2", "4", "5", "6", "9", "10", "26", "27", "28", "31",
        "34", "36", "38", "41", "43", "45", "46", "47", "48", "49", "52", "54",
    }
)
_VERIFIED_APPENDED_AIRLINE_IDS = frozenset(
    {"28", "31", "34", "36", "45", "46", "47", "48", "49"}
)


def _restore_legacy_airline_task(raw: dict) -> dict:
    tid = str(raw.get("id"))
    ec = raw.get("evaluation_criteria")
    if not isinstance(ec, dict):
        return raw
    ec = dict(ec)
    if tid in _ZERO_WRITE_VERIFIED_AIRLINE_IDS:
        ec["reward_basis"] = [
            r for r in (ec.get("reward_basis") or []) if r != "NL_ASSERTION"
        ]
    if tid in _VERIFIED_APPENDED_AIRLINE_IDS and ec.get("nl_assertions"):
        ec["nl_assertions"] = list(ec["nl_assertions"][:-1])
    if tid == "23" and ec.get("nl_assertions"):
        ec["nl_assertions"] = [
            s.replace(
                "outbound flight MER848 on 2024-05-26 and return flights MER626 and MER806 on 2024-05-28",
                "outbound flights MER848 and MER626 on 2024-05-26 and return flight MER806 on 2024-05-28",
            )
            for s in ec["nl_assertions"]
        ]
    return {**raw, "evaluation_criteria": ec}


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    from tau2.hyper.verified import verified_enabled

    tasks = load_file(AIRLINE_PLUS_TASK_SET_PATH)
    if not verified_enabled():
        tasks = [_restore_legacy_airline_task(t) for t in tasks]
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(AIRLINE_PLUS_TASK_SET_PATH).parent
        / f"split_{Path(AIRLINE_PLUS_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
