from __future__ import annotations

from dataclasses import dataclass


TASK_ORDER = ("stack", "carrot", "spoon", "eggplant")
CONTROL_MODE = "arm_pd_ee_target_base_pose_gripper_pd_joint_pos"


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


def make_env(task_key: str):
    """Use the paper's absolute base-frame pose controller for saved actions."""
    import simpler_env

    env = simpler_env.make(TASKS[task_key].env_task)
    actual_mode = env.unwrapped.control_mode
    if actual_mode != CONTROL_MODE:
        env.close()
        raise RuntimeError(
            f"DSD actions require {CONTROL_MODE!r}, but SimplerEnv selected {actual_mode!r}. "
            "Install the pinned X-VLA forks in data_source_dependency/README.md#environment."
        )
    return env


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
