# Copyright Sierra
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail_plus.tools import RetailPlusTools
from tau2.domains.retail_plus.utils import (
    RETAIL_PLUS_DB_PATH,
    RETAIL_PLUS_POLICY_PATH,
    RETAIL_PLUS_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Retail+ domain does not support solo mode")
    if db is None:
        db = RetailDB.load(RETAIL_PLUS_DB_PATH)
    tools = RetailPlusTools(db)
    with open(RETAIL_PLUS_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="retail_plus",
        policy=policy,
        tools=tools,
    )


def _restore_legacy_retail_task(raw: dict) -> dict:
    tid = str(raw.get("id"))
    ec = raw.get("evaluation_criteria")
    if not isinstance(ec, dict) or tid not in {
        "10", "26", "30", "31", "32", "120", "131"
    }:
        return raw
    ec = dict(ec)
    if tid in {"10", "26", "30", "31", "32"}:
        ec["nl_assertions"] = None
    if tid == "26":
        actions = list(ec.get("actions") or [])
        actions.append(
            {
                "action_id": "26_7",
                "name": "transfer_to_human_agents",
                "arguments": {
                    "summary": "The user wants to refund to the amex credit card, but the agent cannot help."
                },
                "info": None,
                "compare_args": [],
            }
        )
        ec["actions"] = actions
    if tid in {"120", "131"} and ec.get("nl_assertions"):
        ec["nl_assertions"] = list(ec["nl_assertions"][:-1])
    return {**raw, "evaluation_criteria": ec}


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    from tau2.hyper.verified import verified_enabled

    tasks = load_file(RETAIL_PLUS_TASK_SET_PATH)
    if not verified_enabled():
        tasks = [_restore_legacy_retail_task(t) for t in tasks]
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
        Path(RETAIL_PLUS_TASK_SET_PATH).parent
        / f"split_{Path(RETAIL_PLUS_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
