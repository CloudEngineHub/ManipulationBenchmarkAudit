from __future__ import annotations

from dataclasses import dataclass


TASK_ORDER = ("stack", "carrot", "spoon", "eggplant")


@dataclass(frozen=True)
class WidowXTask:
    key: str
    env_task: str
    instruction: str
    official_horizon: int


TASKS: dict[str, WidowXTask] = {
    "stack": WidowXTask(
        key="stack",
        env_task="widowx_stack_cube",
        instruction="stack the green block on the yellow block",
        official_horizon=60,
    ),
    "carrot": WidowXTask(
        key="carrot",
        env_task="widowx_carrot_on_plate",
        instruction="put carrot on plate",
        official_horizon=60,
    ),
    "spoon": WidowXTask(
        key="spoon",
        env_task="widowx_spoon_on_towel",
        instruction="put spoon on towel",
        official_horizon=60,
    ),
    "eggplant": WidowXTask(
        key="eggplant",
        env_task="widowx_put_eggplant_in_basket",
        instruction="put eggplant in basket",
        official_horizon=120,
    ),
}


def parse_tasks(tasks: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if tasks is None:
        out = list(TASK_ORDER)
    elif isinstance(tasks, str):
        out = [x.strip() for x in tasks.split(",") if x.strip()]
    else:
        out = [str(x).strip() for x in tasks if str(x).strip()]
    unknown = sorted(set(out) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown WidowX task keys {unknown}; valid keys are {list(TASK_ORDER)}")
    return out
