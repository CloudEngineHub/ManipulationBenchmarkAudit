from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    cfg_get,
    device_from_arg,
    expand_placeholders,
    find_checkpoint,
    load_config,
    normalize_continuous_actions,
    require_torch,
    save_checkpoint,
    set_seed,
    unnormalize_continuous_actions,
    write_csv,
    write_json,
)
from .models import build_policy_from_cfg


EPISODE_RE = re.compile(r"^episode_(\d{7})\.npz$")


def _episode_paths(root: Path, max_files: int | None = None) -> list[Path]:
    paths = [p for p in root.iterdir() if p.is_file() and EPISODE_RE.match(p.name)]
    paths.sort(key=lambda p: p.name)
    if max_files is not None:
        paths = paths[: int(max_files)]
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz files under {root}")
    return paths


def _episode_index(path: Path) -> int:
    match = EPISODE_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a CALVIN episode filename: {path.name}")
    return int(match.group(1))


def _load_step(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        required = ("robot_obs", "scene_obs", "rel_actions", "actions")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"{path} missing keys {missing}; available={sorted(data.files)}")
        robot_obs = np.asarray(data["robot_obs"], dtype=np.float32).reshape(-1)
        scene_obs = np.asarray(data["scene_obs"], dtype=np.float32).reshape(-1)
        rel = np.asarray(data["rel_actions"], dtype=np.float32).reshape(-1)
        abs_action = np.asarray(data["actions"], dtype=np.float32).reshape(-1)
    if robot_obs.shape[0] != 15 or scene_obs.shape[0] != 24 or rel.shape[0] != 7 or abs_action.shape[0] != 7:
        raise ValueError(
            f"{path.name}: expected robot_obs=15, scene_obs=24, rel_actions=7, actions=7; "
            f"got {robot_obs.shape}, {scene_obs.shape}, {rel.shape}, {abs_action.shape}"
        )
    return _episode_index(path), np.concatenate([robot_obs, scene_obs]).astype(np.float32), rel, abs_action


def prepare_data(args: argparse.Namespace) -> None:
    torch = require_torch()
    root = Path(args.dataset_root).resolve()
    paths = _episode_paths(root, args.max_files)
    n = len(paths)
    episode_idx = np.empty((n,), dtype=np.int64)
    gt_state = np.empty((n, 39), dtype=np.float32)
    rel_actions = np.empty((n, 7), dtype=np.float32)
    actions = np.empty((n, 7), dtype=np.float32)
    for i, path in enumerate(paths):
        idx, state, rel, abs_action = _load_step(path)
        episode_idx[i] = idx
        gt_state[i] = state
        rel_actions[i] = rel
        actions[i] = abs_action
        if (i + 1) % 10000 == 0:
            print(f"packed {i + 1}/{n}", flush=True)
    order = np.argsort(episode_idx, kind="mergesort")
    payload = {
        "episode_idx": torch.from_numpy(episode_idx[order]),
        "gt_state": torch.from_numpy(gt_state[order]),
        "rel_actions": torch.from_numpy(rel_actions[order]),
        "actions": torch.from_numpy(actions[order]),
        "dataset_root": str(root),
        "created_by": "shortcut_solvability.code.shortcut_policy.calvin prepare-data",
        "partial": bool(args.max_files is not None),
    }
    out = root / "states_actions_rel_and_abs.pt"
    tmp = out.with_suffix(out.suffix + f".tmp.{__import__('os').getpid()}")
    torch.save(payload, tmp)
    tmp.replace(out)

    stats_n = min(int(args.stats_max_steps), n)
    _write_normalization_stats(
        root / "normalization_stats.npz",
        rel_actions[order[:stats_n]],
        actions[order[:stats_n]],
        gt_state[order[:stats_n]],
        stats_n=stats_n,
        stats_max_steps=int(args.stats_max_steps),
    )
    print(f"wrote {out} and {root / 'normalization_stats.npz'}", flush=True)


def _write_normalization_stats(
    path: Path,
    rel_actions: np.ndarray,
    actions: np.ndarray,
    gt_state: np.ndarray,
    *,
    stats_n: int,
    stats_max_steps: int,
) -> None:
    if stats_n <= 0:
        raise ValueError("cannot compute stats from zero files")

    def mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.astype(np.float64).mean(axis=0)
        var = x.astype(np.float64).var(axis=0)
        return mean.astype(np.float32), (np.sqrt(np.maximum(var, 1e-12)) + 1e-6).astype(np.float32)

    rel_mean, rel_std = mean_std(rel_actions)
    abs_mean, abs_std = mean_std(actions)
    state_mean, state_std = mean_std(gt_state)
    tmp = str(path) + f".tmp.{__import__('os').getpid()}.npz"
    np.savez(
        tmp,
        action_mean_rel=rel_mean,
        action_std_rel=rel_std,
        action_mean_abs=abs_mean,
        action_std_abs=abs_std,
        gt_state_mean=state_mean,
        gt_state_std=state_std,
        n_files_used=np.int64(stats_n),
        max_steps=np.int64(stats_max_steps),
    )
    Path(tmp).replace(path)


def load_task_vocab(cfg: dict[str, Any]) -> list[str]:
    lang_path = Path(str(cfg["dataset_root"])) / str(cfg_get(cfg, "language_segments_path", "lang_annotations/auto_lang_ann.npy"))
    lang = np.load(lang_path, allow_pickle=True).item()
    return sorted(set(str(x) for x in lang["language"]["task"]))


class CalvinTaskChunkDataset:
    def __init__(self, cfg: dict[str, Any], task_vocab: list[str]) -> None:
        self.cfg = dict(cfg)
        self.root = Path(str(cfg["dataset_root"]))
        self.chunk_size = int(cfg["chunk_size"])
        self.action_key = {"rel": "rel_actions", "abs": "actions"}[str(cfg_get(cfg, "action_space", "rel"))]
        lang_path = self.root / str(cfg_get(cfg, "language_segments_path", "lang_annotations/auto_lang_ann.npy"))
        lang = np.load(lang_path, allow_pickle=True).item()
        segments = np.asarray(lang["info"]["indx"], dtype=np.int64)
        segment_tasks = [str(x) for x in lang["language"]["task"]]
        task_to_id = {task: i for i, task in enumerate(task_vocab)}
        ep_bounds = np.load(self.root / "ep_start_end_ids.npy").astype(np.int64, copy=False)
        ep_bounds = ep_bounds[np.argsort(ep_bounds[:, 1], kind="mergesort")]
        ep_ends = ep_bounds[:, 1]
        starts: list[int] = []
        task_ids: list[int] = []
        for seg_i, (seg_start, seg_end) in enumerate(segments):
            start = int(seg_start) + int(cfg_get(cfg, "start_offset", 0))
            end = int(seg_end) - 1 + int(cfg_get(cfg, "end_offset", 0))
            ep_i = int(np.searchsorted(ep_ends, int(seg_start), side="left"))
            start = max(start, int(ep_bounds[ep_i, 0]))
            end = min(end, int(ep_bounds[ep_i, 1]))
            last = end + 1 - self.chunk_size
            if last < start:
                continue
            task_id = task_to_id[segment_tasks[seg_i]]
            for s in range(start, last + 1):
                starts.append(s)
                task_ids.append(task_id)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.task_ids = np.asarray(task_ids, dtype=np.int64)
        if self.starts.size == 0:
            raise ValueError("no CALVIN training chunks; check offsets/chunk_size/lang annotations")
        torch = require_torch()
        payload = torch.load(self.root / "states_actions_rel_and_abs.pt", map_location="cpu", weights_only=False)
        episode_idx = payload["episode_idx"].numpy()
        self.actions = payload[self.action_key].numpy()
        self.pos_by_episode_idx = np.full((int(episode_idx.max()) + 1,), -1, dtype=np.int64)
        self.pos_by_episode_idx[episode_idx] = np.arange(episode_idx.shape[0], dtype=np.int64)
        self.offsets = np.arange(self.chunk_size, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        start = int(self.starts[int(idx)])
        with np.load(self.root / f"episode_{start:07d}.npz") as obs:
            rgb_static = np.ascontiguousarray(obs["rgb_static"])
            rgb_gripper = np.ascontiguousarray(obs["rgb_gripper"])
            robot_obs_15 = np.asarray(obs["robot_obs"], dtype=np.float32)
        step_idx = start + self.offsets
        pos = self.pos_by_episode_idx[step_idx]
        if np.any(pos < 0):
            raise KeyError(f"states_actions_rel_and_abs.pt is missing actions for steps starting at {start}")
        tcp6 = robot_obs_15[:6].astype(np.float32, copy=False)
        grip_width = robot_obs_15[6:7].astype(np.float32, copy=False)
        return {
            "view_a": rgb_static,
            "view_b": rgb_gripper,
            "state": np.concatenate([tcp6, grip_width], axis=0),
            "actions": self.actions[pos].astype(np.float32, copy=False),
            "task_id": np.asarray(self.task_ids[int(idx)], dtype=np.int64),
        }


def _collate(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "view_a": torch.from_numpy(np.stack([b["view_a"] for b in batch])).permute(0, 3, 1, 2).contiguous(),
        "view_b": torch.from_numpy(np.stack([b["view_b"] for b in batch])).permute(0, 3, 1, 2).contiguous(),
        "state": torch.from_numpy(np.stack([b["state"] for b in batch])).float(),
        "actions": torch.from_numpy(np.stack([b["actions"] for b in batch])).float(),
        "task_id": torch.from_numpy(np.stack([b["task_id"] for b in batch])).long(),
    }


def load_normalization_stats(root: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(root) / "normalization_stats.npz")
    gt_mean = np.asarray(data["gt_state_mean"], dtype=np.float32).reshape(-1)
    gt_std = np.asarray(data["gt_state_std"], dtype=np.float32).reshape(-1)
    return {
        "action_mean_rel": np.asarray(data["action_mean_rel"], dtype=np.float32),
        "action_std_rel": np.asarray(data["action_std_rel"], dtype=np.float32),
        "action_mean_abs": np.asarray(data["action_mean_abs"], dtype=np.float32),
        "action_std_abs": np.asarray(data["action_std_abs"], dtype=np.float32),
        "state_mean": np.concatenate([gt_mean[:6], gt_mean[6:7]], axis=0),
        "state_std": np.concatenate([gt_std[:6], gt_std[6:7]], axis=0),
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
    task_vocab = load_task_vocab(cfg)
    dataset = CalvinTaskChunkDataset(cfg, task_vocab)
    stats = load_normalization_stats(cfg["dataset_root"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg_get(cfg, "num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    model = build_policy_from_cfg(
        cfg,
        benchmark="calvin",
        num_tasks=len(task_vocab),
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
    ).to(device)
    lr = float(cfg["lr"])
    main_params = [p for n, p in model.named_parameters() if not n.startswith("visual_encoder.")]
    param_groups = [{"params": main_params, "lr": lr}]
    if model.train_visual_encoder:
        param_groups.append({"params": model.visual_encoder.parameters(), "lr": float(cfg["backbone_lr"])})
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    start_step = 0
    if args.resume:
        ckpt_path = find_checkpoint(cfg["checkpoint_dir"])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
    mean_key, std_key = {"rel": ("action_mean_rel", "action_std_rel"), "abs": ("action_mean_abs", "action_std_abs")}[str(cfg_get(cfg, "action_space", "rel"))]
    action_mean = torch.tensor(stats[mean_key], device=device)
    action_std = torch.tensor(stats[std_key], device=device)
    loss_fn = {"mse": torch.nn.functional.mse_loss, "l1": torch.nn.functional.l1_loss, "smooth_l1": torch.nn.functional.smooth_l1_loss}[str(cfg_get(cfg, "loss_type", "smooth_l1"))]
    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[str(cfg_get(cfg, "amp_dtype", "fp32"))]
    step = start_step
    model.train()
    while step < int(cfg["max_steps"]):
        for batch in loader:
            view_a = batch["view_a"].to(device, non_blocking=True).float() / 255.0
            view_b = batch["view_b"].to(device, non_blocking=True).float() / 255.0
            state = batch["state"].to(device, non_blocking=True).float() if bool(cfg_get(cfg, "use_proprio", False)) else None
            actions = batch["actions"].to(device, non_blocking=True).float()
            task_id = batch["task_id"].to(device, non_blocking=True)
            target = normalize_continuous_actions(actions, action_mean, action_std, dims=6)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
                pred = model(view_a=view_a, view_b=view_b, state=state, task_id=task_id)
                loss = (6.0 / 7.0) * loss_fn(pred[..., :6], target[..., :6]) + (1.0 / 7.0) * loss_fn(pred[..., 6], target[..., 6])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % int(cfg["checkpoint_interval"]) == 0 or step >= int(cfg["max_steps"]):
                _save_train_checkpoint(cfg, model, optimizer, step, task_vocab, stats[mean_key], stats[std_key], stats["state_mean"], stats["state_std"])
            if step >= int(cfg["max_steps"]):
                return


def _save_train_checkpoint(
    cfg: dict[str, Any],
    model: Any,
    optimizer: Any,
    step: int,
    task_vocab: list[str],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> Path:
    return save_checkpoint(
        Path(str(cfg["checkpoint_dir"])) / f"step{int(step)}.pt",
        {
            "step": int(step),
            "cfg": dict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "task_vocab": list(task_vocab),
            "action_mean": np.asarray(action_mean, dtype=np.float32),
            "action_std": np.asarray(action_std, dtype=np.float32),
            "state_mean": np.asarray(state_mean, dtype=np.float32),
            "state_std": np.asarray(state_std, dtype=np.float32),
        },
    )


@contextlib.contextmanager
def _temp_seed(seed: int) -> Any:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _initial_env_state(initial_condition: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyhash
    except ImportError as exc:
        raise ImportError("CALVIN official initial-state hashing requires pyhash") from exc

    robot_obs = np.array(
        [0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596, 1.50013585, 0.07999963, -1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899, 1.64119093, 0.55344928, 1.0],
        dtype=np.float64,
    )
    block_rot_z_range = (math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01], dtype=np.float64)
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01], dtype=np.float64)
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01], dtype=np.float64),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01], dtype=np.float64),
    ]
    with _temp_seed(pyhash.fnv1_32()(str(initial_condition.values()))):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24, dtype=np.float64)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = float(initial_condition["lightbulb"])
        scene_obs[5] = float(initial_condition["led"])
        placements = [("red_block", 6, 11), ("blue_block", 12, 17), ("pink_block", 18, 23)]
        for name, pos_start, rot_idx in placements:
            loc = initial_condition[name]
            if loc == "slider_right":
                scene_obs[pos_start : pos_start + 3] = block_slider_right
            elif loc == "slider_left":
                scene_obs[pos_start : pos_start + 3] = block_slider_left
            elif name == "blue_block" and initial_condition["red_block"] == "table":
                scene_obs[pos_start : pos_start + 3] = block_table[1]
            elif name == "pink_block":
                scene_obs[pos_start : pos_start + 3] = block_table[1]
            else:
                scene_obs[pos_start : pos_start + 3] = block_table[0]
            scene_obs[rot_idx] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def _add_calvin_to_path(cfg: dict[str, Any]) -> None:
    conf_dir = Path(str(cfg["calvin_conf_dir"]))
    models_root = conf_dir.parent if conf_dir.name == "conf" else conf_dir
    env_root = Path(str(cfg_get(cfg, "calvin_env_dir", models_root.parent / "calvin_env")))
    for p in [models_root, env_root]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _step_without_render(env: Any, action: Any) -> dict[str, Any]:
    env.robot.apply_action(action)
    for _ in range(env.action_repeat):
        env.p.stepSimulation(physicsClientId=env.cid)
    env.scene.step()
    return env.get_info()


def _obs_to_policy(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32)
    return {
        "view_a": np.ascontiguousarray(obs["rgb_obs"]["rgb_static"]),
        "view_b": np.ascontiguousarray(obs["rgb_obs"]["rgb_gripper"]),
        "state": np.concatenate([robot_obs[:6], robot_obs[6:7]], axis=0).astype(np.float32),
    }


def eval(args: argparse.Namespace) -> None:
    torch = require_torch()
    cfg = expand_placeholders(
        load_config(args.config),
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
    )
    _add_calvin_to_path(cfg)
    import hydra
    from calvin_env.envs.play_table_env import get_env
    from omegaconf import OmegaConf

    device = device_from_arg(args.device)
    ckpt_path = find_checkpoint(cfg["checkpoint_dir"], step=args.step)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    task_vocab = list(ckpt["task_vocab"])
    task_to_id = {task: i for i, task in enumerate(task_vocab)}
    eval_cfg = dict(cfg)
    eval_cfg["train_visual_encoder"] = False
    eval_cfg["pretrained_visual"] = False
    model = build_policy_from_cfg(
        eval_cfg,
        benchmark="calvin",
        num_tasks=len(task_vocab),
        state_mean=np.asarray(ckpt["state_mean"], dtype=np.float32),
        state_std=np.asarray(ckpt["state_std"], dtype=np.float32),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    action_mean = torch.tensor(np.asarray(ckpt["action_mean"], dtype=np.float32), device=device)
    action_std = torch.tensor(np.asarray(ckpt["action_std"], dtype=np.float32), device=device)
    conf_dir = Path(str(cfg["calvin_conf_dir"]))
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    env = get_env(Path(str(cfg["calvin_dataset_path"])) / "validation", show_gui=False)
    sequences_path = Path(args.eval_sequences)
    with sequences_path.open("r", encoding="utf-8") as f:
        sequences = json.load(f)[: int(cfg_get(cfg, "num_sequences", 1000))]
    rows: list[dict[str, Any]] = []
    try:
        for seq_idx, (initial_state, eval_sequence) in enumerate(sequences):
            robot_obs0, scene_obs0 = _initial_env_state(initial_state)
            env.reset(robot_obs=robot_obs0, scene_obs=scene_obs0)
            success_len = 0
            for subtask in eval_sequence:
                tid = task_to_id[subtask]
                start_info = env.get_info()
                planned: deque[np.ndarray] = deque()
                success = False
                for _ in range(int(cfg_get(cfg, "ep_len", 360))):
                    if not planned:
                        obs = _obs_to_policy(env.get_obs())
                        view_a = torch.from_numpy(obs["view_a"]).permute(2, 0, 1)[None].to(device).float() / 255.0
                        view_b = torch.from_numpy(obs["view_b"]).permute(2, 0, 1)[None].to(device).float() / 255.0
                        state = torch.from_numpy(obs["state"])[None].to(device).float() if bool(cfg_get(cfg, "use_proprio", False)) else None
                        task_t = torch.tensor([tid], device=device)
                        with torch.inference_mode():
                            pred = model(view_a=view_a, view_b=view_b, state=state, task_id=task_t)[0]
                        acts = unnormalize_continuous_actions(pred, action_mean, action_std, dims=6).detach().cpu().numpy()
                        if str(cfg_get(cfg, "action_space", "rel")) == "rel":
                            acts = np.clip(acts, -1.0, 1.0)
                        acts[..., -1] = np.where(acts[..., -1] > 0.0, 1.0, -1.0)
                        planned.extend(np.asarray(a, dtype=np.float32) for a in acts[: int(cfg["replan_steps"])])
                    info = _step_without_render(env, planned.popleft())
                    if task_oracle.get_task_info_for_set(start_info, info, {subtask}):
                        success = True
                        break
                if success:
                    success_len += 1
                else:
                    break
            rows.append({"sequence_idx": seq_idx, "tasks_completed": success_len})
    finally:
        env.close()
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "trials.csv", ["sequence_idx", "tasks_completed"], rows)
    values = [int(r["tasks_completed"]) for r in rows]
    chain = _count_success(values)
    write_json(
        out_dir / "summary.json",
        {
            "benchmark": "CALVIN",
            "checkpoint": str(ckpt_path),
            "num_sequences": len(values),
            "avg_seq_len": float(np.mean(values)) if values else 0.0,
            "chain_sr": {str(i + 1): float(v) for i, v in enumerate(chain)},
        },
    )


def _count_success(results: list[int]) -> list[float]:
    count = Counter(results)
    return [0.0 if not results else sum(count[j] for j in range(i, 6)) / len(results) for i in range(1, 6)]


def inspect_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(
        {
            "dataset_root": cfg.get("dataset_root"),
            "calvin_dataset_path": cfg.get("calvin_dataset_path"),
            "chunk_size": cfg.get("chunk_size"),
            "replan_steps": cfg.get("replan_steps"),
            "num_sequences": cfg.get("num_sequences"),
            "ep_len": cfg.get("ep_len"),
        }
    )


def inspect_data(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root)
    paths = _episode_paths(root, max_files=1)
    idx, state, rel, action = _load_step(paths[0])
    print({"first_index": idx, "gt_state_shape": state.shape, "rel_shape": rel.shape, "action_shape": action.shape})


def main() -> None:
    parser = argparse.ArgumentParser(description="CALVIN DINO/task-id shortcut policy commands")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ic = sub.add_parser("inspect-config")
    ic.add_argument("config")
    ic.set_defaults(func=inspect_config)
    idata = sub.add_parser("inspect-data")
    idata.add_argument("dataset_root")
    idata.set_defaults(func=inspect_data)
    prep = sub.add_parser("prepare-data")
    prep.add_argument("dataset_root", help="CALVIN training split, e.g. <DATA_ROOT>/task_ABCD_D/training")
    prep.add_argument("--stats-max-steps", type=int, default=50000)
    prep.add_argument("--max-files", type=int, help="debug/smoke only; omit for a real split")
    prep.set_defaults(func=prepare_data)
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
    ev.add_argument("--eval-sequences", default="shortcut_solvability/results/calvin/official_1000_eval_sequences.json")
    ev.add_argument("--step", type=int)
    ev.add_argument("--device")
    ev.set_defaults(func=eval)
    args = parser.parse_args()
    t0 = time.time()
    args.func(args)
    if args.cmd not in {"inspect-config", "inspect-data"}:
        print(f"{args.cmd} completed in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
