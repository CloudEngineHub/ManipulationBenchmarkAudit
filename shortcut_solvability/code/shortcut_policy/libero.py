from __future__ import annotations

import argparse
import collections
import glob
import re
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .common import (
    cfg_get,
    device_from_arg,
    expand_placeholders,
    find_checkpoint,
    load_config,
    require_torch,
    save_checkpoint,
    set_seed,
    write_csv,
    write_json,
)
from .models import build_policy_from_cfg


ACTION_DIM = 7
NOOP_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _libero_suite(task_suite: str) -> Any:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[str(task_suite)]()


def _all_tasks(task_suite: str) -> list[int]:
    suite = _libero_suite(task_suite)
    return list(range(int(suite.n_tasks)))


def _scene_prefix_free(name: str) -> str:
    text = name[:-5] if name.endswith(".hdf5") else name
    text = text[:-5] if text.endswith("_demo") else text
    text = re.sub(r"^(kitchen|study)_scene\d+_|^living_room_scene\d+_", "", text, flags=re.IGNORECASE)
    return text.replace("_", " ").strip().lower()


def _lang_to_task_id(task_suite: str) -> dict[str, int]:
    suite = _libero_suite(task_suite)
    out: dict[str, int] = {}
    for task_id in range(int(suite.n_tasks)):
        out[suite.get_task(task_id).language.strip().lower()] = int(task_id)
    return out


def _state_from_official_obs(obs: Any, t: int) -> np.ndarray:
    return np.concatenate([obs["ee_pos"][t], obs["ee_ori"][t], obs["gripper_states"][t]], axis=0).astype(np.float32)


class LiberoOfficialDataset:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.task_suite = str(cfg["task_suite"])
        self.chunk_size = int(cfg["chunk_size"])
        self.tasks = [int(x) for x in cfg_get(cfg, "tasks", [])] or _all_tasks(self.task_suite)
        self.num_tasks = len(_all_tasks(self.task_suite))
        suite = _libero_suite(self.task_suite)
        suite_dir = Path(str(cfg["dataset_dir"])) / self.task_suite
        self.samples: list[tuple[Path, str, int, int]] = []
        action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
        action_sumsq = np.zeros(ACTION_DIM, dtype=np.float64)
        action_count = 0
        for task_id in self.tasks:
            task = suite.get_task(int(task_id))
            demo_file = suite_dir / f"{task.name}_demo.hdf5"
            if not demo_file.exists():
                raise FileNotFoundError(demo_file)
            with h5py.File(demo_file, "r") as f:
                for demo_key in f["data"].keys():
                    actions = f["data"][demo_key]["actions"][()].astype(np.float32)
                    action_sum += actions.sum(axis=0, dtype=np.float64)
                    action_sumsq += np.square(actions, dtype=np.float64).sum(axis=0)
                    action_count += int(actions.shape[0])
                    for t in range(max(0, int(actions.shape[0]) - self.chunk_size)):
                        self.samples.append((demo_file, str(demo_key), int(t), int(task_id)))
        self.action_mean, self.action_std = _mean_std(action_sum, action_sumsq, action_count, "official LIBERO actions")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        demo_file, demo_key, t, task_id = self.samples[int(idx)]
        with h5py.File(demo_file, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]
            return {
                "view_a": np.ascontiguousarray(obs["agentview_rgb"][t]),
                "view_b": np.ascontiguousarray(obs["eye_in_hand_rgb"][t]),
                "state": _state_from_official_obs(obs, t),
                "actions": demo["actions"][t : t + self.chunk_size].astype(np.float32),
                "task_id": np.asarray(task_id, dtype=np.int64),
            }


class LiberoNonoopsDataset:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.task_suite = str(cfg["task_suite"])
        self.chunk_size = int(cfg["chunk_size"])
        self.tasks = [int(x) for x in cfg_get(cfg, "tasks", [])] or _all_tasks(self.task_suite)
        lang_to_id = _lang_to_task_id(self.task_suite)
        self.num_tasks = len(lang_to_id)
        self.samples: list[tuple[Path, str, int, int]] = []
        action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
        action_sumsq = np.zeros(ACTION_DIM, dtype=np.float64)
        action_count = 0
        for path_s in sorted(glob.glob(str(Path(str(cfg["data_root_dir"])) / "*.hdf5"))):
            path = Path(path_s)
            lang = _scene_prefix_free(path.name)
            if lang not in lang_to_id:
                raise KeyError(f"{path.name}: language {lang!r} is not in LIBERO {self.task_suite}")
            task_id = lang_to_id[lang]
            if task_id not in self.tasks:
                continue
            with h5py.File(path, "r") as f:
                for demo_key in f["data"].keys():
                    demo = f["data"][demo_key]
                    actions = demo["actions"][()].astype(np.float32)
                    action_sum += actions.sum(axis=0, dtype=np.float64)
                    action_sumsq += np.square(actions, dtype=np.float64).sum(axis=0)
                    action_count += int(actions.shape[0])
                    for t in range(max(0, int(actions.shape[0]) - self.chunk_size + 1)):
                        self.samples.append((path, str(demo_key), int(t), int(task_id)))
        self.action_mean, self.action_std = _mean_std(action_sum, action_sumsq, action_count, "no-noops LIBERO actions")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path, demo_key, t, task_id = self.samples[int(idx)]
        with h5py.File(path, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]
            state = np.concatenate([obs["ee_states"][t], obs["gripper_states"][t]], axis=0).astype(np.float32)
            return {
                "view_a": np.ascontiguousarray(obs["agentview_rgb"][t]),
                "view_b": np.ascontiguousarray(obs["eye_in_hand_rgb"][t]),
                "state": state,
                "actions": demo["actions"][t : t + self.chunk_size].astype(np.float32),
                "task_id": np.asarray(task_id, dtype=np.int64),
            }


def _mean_std(sum_v: np.ndarray, sumsq_v: np.ndarray, count: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError(f"no {label} found")
    mean = sum_v / float(count)
    var = sumsq_v / float(count) - mean**2
    return mean.astype(np.float32), (np.sqrt(np.maximum(var, 1e-12)) + 1e-6).astype(np.float32)


def build_dataset(cfg: dict[str, Any]) -> LiberoOfficialDataset | LiberoNonoopsDataset:
    if str(cfg_get(cfg, "dataset_type", "original")) == "original":
        return LiberoOfficialDataset(cfg)
    if str(cfg["dataset_type"]) == "nonoops":
        return LiberoNonoopsDataset(cfg)
    raise ValueError(f"unknown dataset_type={cfg['dataset_type']!r}")


def _collate(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "view_a": torch.from_numpy(np.stack([b["view_a"] for b in batch])).permute(0, 3, 1, 2).contiguous(),
        "view_b": torch.from_numpy(np.stack([b["view_b"] for b in batch])).permute(0, 3, 1, 2).contiguous(),
        "state": torch.from_numpy(np.stack([b["state"] for b in batch])).float(),
        "actions": torch.from_numpy(np.stack([b["actions"] for b in batch])).float(),
        "task_id": torch.from_numpy(np.stack([b["task_id"] for b in batch])).long(),
    }


def train(args: argparse.Namespace) -> None:
    torch = require_torch()
    cfg = expand_placeholders(
        load_config(args.config),
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
    )
    set_seed(int(cfg["seed"]))
    device = device_from_arg(args.device)
    dataset = build_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg_get(cfg, "num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    model = build_policy_from_cfg(cfg, benchmark="libero", num_tasks=dataset.num_tasks).to(device)
    main_params = list(model.action_head.parameters())
    if model.proprio_encoder is not None:
        main_params += list(model.proprio_encoder.parameters())
    param_groups = [{"params": main_params, "lr": float(cfg["lr"])}]
    if model.train_visual_encoder:
        param_groups.append({"params": model.visual_encoder.parameters(), "lr": float(cfg["backbone_lr"])})
    optimizer = torch.optim.AdamW(param_groups, lr=float(cfg["lr"]), weight_decay=1e-4)
    start_step = 0
    if args.resume:
        ckpt_path = find_checkpoint(cfg["checkpoint_dir"])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
    action_mean = torch.tensor(dataset.action_mean, device=device)
    action_std = torch.tensor(dataset.action_std, device=device)
    loss_name = str(cfg_get(cfg, "loss_type", "mse"))
    loss_fn = {"mse": torch.nn.functional.mse_loss, "l1": torch.nn.functional.l1_loss}[loss_name]
    step = start_step
    model.train()
    while step < int(cfg["max_steps"]):
        for batch in loader:
            view_a = batch["view_a"].to(device, non_blocking=True).float() / 255.0
            view_b = batch["view_b"].to(device, non_blocking=True).float() / 255.0
            state = batch["state"].to(device, non_blocking=True).float() if bool(cfg_get(cfg, "use_proprio", True)) else None
            actions = batch["actions"].to(device, non_blocking=True).float()
            target = (actions - action_mean) / action_std
            pred = model(view_a=view_a, view_b=view_b, state=state, task_id=batch["task_id"].to(device))
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % int(cfg["checkpoint_interval"]) == 0 or step >= int(cfg["max_steps"]):
                _save_train_checkpoint(cfg, model, optimizer, step, dataset.action_mean, dataset.action_std)
            if step >= int(cfg["max_steps"]):
                return


def _save_train_checkpoint(
    cfg: dict[str, Any],
    model: Any,
    optimizer: Any,
    step: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> Path:
    return save_checkpoint(
        Path(str(cfg["checkpoint_dir"])) / f"step{int(step)}.pt",
        {
            "step": int(step),
            "cfg": dict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "action_mean": np.asarray(action_mean, dtype=np.float32),
            "action_std": np.asarray(action_std, dtype=np.float32),
        },
    )


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = quat.astype(np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * np.arccos(q[3])) / den).astype(np.float32)


def eval(args: argparse.Namespace) -> None:
    torch = require_torch()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = expand_placeholders(
        load_config(args.config),
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
    )
    device = device_from_arg(args.device)
    suite = _libero_suite(str(cfg["task_suite"]))
    ckpt_path = find_checkpoint(cfg["checkpoint_dir"], step=args.step)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_policy_from_cfg(cfg, benchmark="libero", num_tasks=int(suite.n_tasks)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    action_mean = torch.tensor(np.asarray(ckpt["action_mean"], dtype=np.float32), device=device)
    action_std = torch.tensor(np.asarray(ckpt["action_std"], dtype=np.float32), device=device)
    rows: list[dict[str, Any]] = []
    max_steps = int(MAX_STEPS_PER_SUITE[str(cfg["task_suite"])])
    eval_trials = int(cfg_get(cfg, "eval_trials", 50))
    replan_steps = int(cfg["replan_steps"])
    wait_steps = int(cfg_get(cfg, "num_steps_wait", 10))
    for task_id in [int(x) for x in cfg_get(cfg, "tasks", [])] or list(range(int(suite.n_tasks))):
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
            camera_heights=224,
            camera_widths=224,
            render_gpu_device_id=int(args.render_gpu),
        )
        env.seed(int(cfg["seed"]))
        try:
            init_states = suite.get_task_init_states(task_id)
            for trial_idx in range(min(eval_trials, len(init_states))):
                env.reset()
                obs = env.set_init_state(init_states[trial_idx])
                plan: collections.deque[np.ndarray] = collections.deque()
                success = False
                for t in range(max_steps + wait_steps):
                    if t < wait_steps:
                        obs, _, done, _ = env.step(NOOP_ACTION.tolist())
                    else:
                        if not plan:
                            policy_state = np.concatenate(
                                [
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                ],
                                axis=0,
                            ).astype(np.float32)
                            view_a = torch.from_numpy(np.ascontiguousarray(obs["agentview_image"])).permute(2, 0, 1)[None].to(device).float() / 255.0
                            view_b = torch.from_numpy(np.ascontiguousarray(obs["robot0_eye_in_hand_image"])).permute(2, 0, 1)[None].to(device).float() / 255.0
                            state = torch.from_numpy(policy_state)[None].to(device).float() if bool(cfg_get(cfg, "use_proprio", True)) else None
                            task_t = torch.tensor([task_id], device=device)
                            with torch.inference_mode():
                                pred = model(view_a=view_a, view_b=view_b, state=state, task_id=task_t)[0]
                            actions = (pred * action_std + action_mean).detach().cpu().numpy().astype(np.float32)
                            plan.extend(actions[:replan_steps])
                        obs, _, done, _ = env.step(np.asarray(plan.popleft(), dtype=np.float32).tolist())
                    if bool(done):
                        success = True
                        break
                rows.append(
                    {
                        "task_idx": task_id,
                        "task_name": task.name,
                        "trial_idx": trial_idx,
                        "success": int(success),
                    }
                )
        finally:
            env.close()
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "trials.csv", ["task_idx", "task_name", "trial_idx", "success"], rows)
    successes = sum(int(r["success"]) for r in rows)
    write_json(
        out_dir / "summary.json",
        {
            "benchmark": "LIBERO",
            "task_suite": cfg["task_suite"],
            "checkpoint": str(ckpt_path),
            "successes": successes,
            "trials": len(rows),
            "success_rate": 0.0 if not rows else successes / len(rows),
        },
    )


def inspect_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(
        {
            "task_suite": cfg.get("task_suite"),
            "dataset_type": cfg.get("dataset_type"),
            "chunk_size": cfg.get("chunk_size"),
            "replan_steps": cfg.get("replan_steps"),
            "eval_trials": cfg.get("eval_trials"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LIBERO DINO/task-id shortcut policy commands")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ic = sub.add_parser("inspect-config")
    ic.add_argument("config")
    ic.set_defaults(func=inspect_config)
    tr = sub.add_parser("train")
    tr.add_argument("config")
    tr.add_argument("--data-root")
    tr.add_argument("--checkpoint-root")
    tr.add_argument("--output-root")
    tr.add_argument("--device")
    tr.add_argument("--resume", action="store_true")
    tr.set_defaults(func=train)
    ev = sub.add_parser("eval")
    ev.add_argument("config")
    ev.add_argument("--data-root")
    ev.add_argument("--checkpoint-root")
    ev.add_argument("--output-root")
    ev.add_argument("--out-dir", required=True)
    ev.add_argument("--step", type=int)
    ev.add_argument("--device")
    ev.add_argument("--render-gpu", type=int, default=0)
    ev.set_defaults(func=eval)
    args = parser.parse_args()
    t0 = time.time()
    args.func(args)
    if args.cmd != "inspect-config":
        print(f"{args.cmd} completed in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
