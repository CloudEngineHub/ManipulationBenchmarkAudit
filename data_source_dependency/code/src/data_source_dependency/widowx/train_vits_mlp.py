from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data_source_dependency.widowx.data import (
    build_widowx_loader,
    stats_from_summary,
    summarize_dataset,
    write_json,
)
from data_source_dependency.widowx.model import WidowXViTSMLPPolicy
from data_source_dependency.widowx.normalization import normalize_action_tensor, preprocess_proprio_tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(cfg)
    for key in (
        "dataset_root",
        "output_dir",
        "max_steps",
        "batch_size",
        "num_workers",
        "limit_episodes_per_task",
        "tasks",
    ):
        value = getattr(args, key)
        if value is not None:
            out[key] = value
    if args.run_name is not None:
        out["run_name"] = args.run_name
        out["output_dir"] = str(Path(out["output_root"]) / args.run_name)
    if args.preload_to_memory is not None:
        out["preload_to_memory"] = bool(args.preload_to_memory)
    if args.train_visual_encoder is not None:
        out["train_visual_encoder"] = bool(args.train_visual_encoder)
    if args.amp_dtype is not None:
        out["amp_dtype"] = args.amp_dtype
    if args.image_aug is not None:
        out["image_aug"] = args.image_aug
    if args.image_norm is not None:
        out["image_norm"] = args.image_norm
    if args.use_proprio is not None:
        out["use_proprio"] = bool(args.use_proprio)
    if args.action_key is not None:
        out["action_key"] = args.action_key
    if args.action_norm is not None:
        out["action_norm"] = args.action_norm
    if args.proprio_norm is not None:
        out["proprio_norm"] = args.proprio_norm
    if args.proprio_preprocess is not None:
        out["proprio_preprocess"] = args.proprio_preprocess
    if args.init_checkpoint is not None:
        out["init_checkpoint"] = args.init_checkpoint
    if args.stats_checkpoint is not None:
        out["stats_checkpoint"] = args.stats_checkpoint
    return out


def _finalize_action_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    action_key = str(out.get("action_key", "actions_env"))
    action_norm = str(out.get("action_norm", "dataset_standard"))
    if action_key == "actions_env":
        out.setdefault("action_dim", 7)
        out.setdefault("action_normalize_dims", 6)
    elif action_key == "actions_xvla_first10":
        out.setdefault("action_dim", 10)
        out.setdefault("action_normalize_dims", 9)
    else:
        raise ValueError("action_key must be actions_env or actions_xvla_first10")
    if action_norm == "bridge_q01_q99" and action_key != "actions_env":
        raise ValueError("action_norm=bridge_q01_q99 requires action_key=actions_env")
    if action_norm not in {"dataset_standard", "bridge_q01_q99", "none"}:
        raise ValueError("action_norm must be dataset_standard, bridge_q01_q99, or none")
    if str(out.get("proprio_norm", "none")) not in {"none", "dataset_standard"}:
        raise ValueError("proprio_norm must be none or dataset_standard")
    if str(out.get("proprio_preprocess", "none")) not in {"none", "xvla_zero_gripper"}:
        raise ValueError("proprio_preprocess must be none or xvla_zero_gripper")
    return out


def _expand_path_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key in (
        "dataset_root",
        "output_root",
        "output_dir",
        "dino_hub_dir",
        "init_checkpoint",
        "stats_checkpoint",
    ):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = os.path.expanduser(os.path.expandvars(value))
    return out


def _save_checkpoint(
    path: Path,
    step: int,
    model: WidowXViTSMLPPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    task_vocab: list[str],
    stats: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "task_vocab": task_vocab,
            "action_mean": stats["action_mean"],
            "action_std": stats["action_std"],
            "action_min": stats["action_min"],
            "action_max": stats["action_max"],
            "proprio_mean": stats.get("proprio_mean"),
            "proprio_std": stats.get("proprio_std"),
            "proprio_min": stats.get("proprio_min"),
            "proprio_max": stats.get("proprio_max"),
        },
        tmp,
    )
    os.replace(tmp, path)


def _load_init_checkpoint(
    path: str | Path,
    model: WidowXViTSMLPPolicy,
    cfg: dict[str, Any],
    task_vocab: list[str],
    device: torch.device,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"init_checkpoint does not exist: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_task_vocab = list(ckpt["task_vocab"])
    if ckpt_task_vocab != task_vocab:
        raise ValueError(
            f"init_checkpoint task_vocab mismatch: checkpoint={ckpt_task_vocab}, current={task_vocab}"
        )
    ckpt_cfg = dict(ckpt.get("cfg", {}))
    compatibility_keys = (
        "dino_model",
        "visual_feature_mode",
        "patch_pool_size",
        "image_norm",
        "use_proprio",
        "proprio_norm",
        "proprio_preprocess",
        "chunk_size",
        "action_key",
        "action_dim",
        "action_normalize_dims",
        "task_dim",
        "task_embedding_scale",
        "hidden_dim",
    )
    mismatches = []
    for key in compatibility_keys:
        if key in ckpt_cfg and key in cfg and ckpt_cfg[key] != cfg[key]:
            mismatches.append((key, ckpt_cfg[key], cfg[key]))
    if mismatches:
        detail = ", ".join(f"{key}: checkpoint={old!r} current={new!r}" for key, old, new in mismatches)
        raise ValueError(f"init_checkpoint config mismatch: {detail}")
    model.load_state_dict(ckpt["model"], strict=True)
    return {
        "init_checkpoint": str(path),
        "init_step": int(ckpt.get("step", -1)),
        "init_task_vocab": ckpt_task_vocab,
    }


def _load_stats_checkpoint(
    path: str | Path,
    cfg: dict[str, Any],
    reference_stats: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stats_checkpoint does not exist: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_cfg = dict(ckpt.get("cfg", {}))
    compatibility_keys = (
        "use_proprio",
        "proprio_norm",
        "proprio_preprocess",
        "action_key",
        "action_norm",
        "action_dim",
        "action_normalize_dims",
    )
    mismatches = []
    for key in compatibility_keys:
        if key in ckpt_cfg and key in cfg and ckpt_cfg[key] != cfg[key]:
            mismatches.append((key, ckpt_cfg[key], cfg[key]))
    if mismatches:
        detail = ", ".join(f"{key}: checkpoint={old!r} current={new!r}" for key, old, new in mismatches)
        raise ValueError(f"stats_checkpoint config mismatch: {detail}")

    stats: dict[str, np.ndarray] = {}
    required_keys = ("action_mean", "action_std", "action_min", "action_max")
    optional_keys = ("proprio_mean", "proprio_std", "proprio_min", "proprio_max")
    for key in required_keys:
        if key not in ckpt:
            raise ValueError(f"stats_checkpoint missing {key}: {path}")
        value = np.asarray(ckpt[key], dtype=np.float32)
        if value.shape != reference_stats[key].shape:
            raise ValueError(
                f"stats_checkpoint {key} shape mismatch: checkpoint={value.shape} "
                f"dataset={reference_stats[key].shape}"
            )
        stats[key] = value
    for key in optional_keys:
        reference = reference_stats.get(key)
        value = ckpt.get(key)
        if reference is None:
            if value is not None:
                raise ValueError(f"stats_checkpoint has unexpected {key} for current dataset/config")
            continue
        if value is None:
            raise ValueError(f"stats_checkpoint missing {key}: {path}")
        array = np.asarray(value, dtype=np.float32)
        if array.shape != reference.shape:
            raise ValueError(
                f"stats_checkpoint {key} shape mismatch: checkpoint={array.shape} "
                f"dataset={reference.shape}"
            )
        stats[key] = array
    return stats, {
        "stats_checkpoint": str(path),
        "stats_step": int(ckpt.get("step", -1)),
        "stats_task_vocab": list(ckpt.get("task_vocab", [])),
    }


def train(cfg: dict[str, Any]) -> None:
    cfg = _finalize_action_cfg(cfg)
    cfg = _expand_path_cfg(cfg)
    set_seed(int(cfg.get("seed", 0)))
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    summary = summarize_dataset(
        cfg["dataset_root"],
        tasks=cfg.get("tasks"),
        chunk_size=int(cfg["chunk_size"]),
        limit_episodes_per_task=cfg.get("limit_episodes_per_task"),
        action_key=str(cfg.get("action_key", "actions_env")),
        pad_terminal_actions=bool(cfg.get("pad_terminal_actions", False)),
    )
    write_json(output_dir / "dataset_summary.json", summary)
    stats = stats_from_summary(summary)
    if cfg.get("stats_checkpoint"):
        stats, stats_info = _load_stats_checkpoint(cfg["stats_checkpoint"], cfg, stats)
        write_json(output_dir / "stats_checkpoint.json", stats_info)

    loader, dataset = build_widowx_loader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training requires a CUDA-capable device.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model = WidowXViTSMLPPolicy(cfg=cfg, num_tasks=len(dataset.task_vocab)).to(device)
    if cfg.get("init_checkpoint"):
        init_info = _load_init_checkpoint(cfg["init_checkpoint"], model, cfg, dataset.task_vocab, device)
        write_json(output_dir / "init_checkpoint.json", init_info)
    main_params = [p for n, p in model.named_parameters() if not n.startswith("visual_encoder.")]
    param_groups = [{"params": main_params, "lr": float(cfg["lr"])}]
    if model.train_visual_encoder:
        param_groups.append({"params": model.visual_encoder.parameters(), "lr": float(cfg["backbone_lr"])})
    optimizer = torch.optim.AdamW(param_groups, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))

    action_mean = torch.tensor(stats["action_mean"], device=device, dtype=torch.float32)
    action_std = torch.tensor(stats["action_std"], device=device, dtype=torch.float32)
    proprio_mean = None
    proprio_std = None
    if stats.get("proprio_mean") is not None:
        proprio_mean = torch.tensor(stats["proprio_mean"], device=device, dtype=torch.float32)
        proprio_std = torch.tensor(stats["proprio_std"], device=device, dtype=torch.float32)
    action_normalize_dims = int(cfg.get("action_normalize_dims", 6))
    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        str(cfg.get("amp_dtype", "fp16"))
    ]
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Requested bf16 AMP, but this CUDA device does not support bf16; using fp16 AMP.")
        amp_dtype = torch.float16
    loss_fn = {
        "mse": nn.functional.mse_loss,
        "l1": nn.functional.l1_loss,
        "smooth_l1": nn.functional.smooth_l1_loss,
    }[str(cfg.get("loss_type", "smooth_l1"))]

    log_path = output_dir / "train_log.jsonl"
    step = 0
    start = time.time()
    max_steps = int(cfg["max_steps"])
    checkpoint_interval = int(cfg.get("checkpoint_interval", 1000))
    max_runtime_hours = float(cfg.get("max_runtime_hours", 2.0))

    with tqdm(total=max_steps) as pbar:
        while step < max_steps:
            data_start = time.time()
            for batch in loader:
                data_wait_sec = time.time() - data_start
                step_start = time.time()
                if time.time() - start >= max_runtime_hours * 3600.0:
                    _save_checkpoint(
                        checkpoint_dir / f"step{step}.pt",
                        step,
                        model,
                        optimizer,
                        cfg,
                        dataset.task_vocab,
                        stats,
                    )
                    return

                image = batch["image"].to(device, non_blocking=True).float() / 255.0
                task_id = batch["task_id"].to(device, non_blocking=True)
                actions_raw = batch["actions"].to(device, non_blocking=True).float()
                proprio = batch.get("proprio")
                if proprio is not None:
                    proprio = proprio.to(device, non_blocking=True).float()
                    proprio = preprocess_proprio_tensor(proprio, cfg, proprio_mean, proprio_std)
                actions_gt = normalize_action_tensor(actions_raw, cfg, action_mean, action_std)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    pred = model(image=image, task_id=task_id, proprio=proprio)
                    loss = loss_fn(pred, actions_gt)
                    loss_cont = loss_fn(pred[..., :action_normalize_dims], actions_gt[..., :action_normalize_dims])
                    if action_normalize_dims < pred.shape[-1]:
                        loss_grip = loss_fn(pred[..., action_normalize_dims:], actions_gt[..., action_normalize_dims:])
                    else:
                        loss_grip = torch.zeros((), device=device, dtype=loss.dtype)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1

                if step % int(cfg.get("log_interval", 10)) == 0:
                    record = {
                        "step": step,
                        "loss": float(loss.item()),
                        "loss_cont": float(loss_cont.item()),
                        "loss_grip": float(loss_grip.item()),
                        "elapsed_sec": time.time() - start,
                        "data_wait_sec": data_wait_sec,
                        "step_sec": time.time() - step_start,
                    }
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, sort_keys=True) + "\n")
                    pbar.set_postfix(loss=f"{record['loss']:.4f}")
                    pbar.update(int(cfg.get("log_interval", 10)))

                if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                    _save_checkpoint(
                        checkpoint_dir / f"step{step}.pt",
                        step,
                        model,
                        optimizer,
                        cfg,
                        dataset.task_vocab,
                        stats,
                    )
                if step >= max_steps:
                    break
                data_start = time.time()

    _save_checkpoint(
        checkpoint_dir / f"step{step}.pt",
        step,
        model,
        optimizer,
        cfg,
        dataset.task_vocab,
        stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WidowX ViT-S + MLP BC policy.")
    parser.add_argument("config", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-episodes-per-task", type=int)
    parser.add_argument("--tasks", type=str)
    parser.add_argument("--preload-to-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-visual-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image-aug", choices=["none", "random_crop_erasing"])
    parser.add_argument("--image-norm", choices=["dinov2", "imagenet", "clip"])
    parser.add_argument("--action-key", choices=["actions_env", "actions_xvla_first10"])
    parser.add_argument("--action-norm", choices=["dataset_standard", "bridge_q01_q99", "none"])
    parser.add_argument("--proprio-norm", choices=["none", "dataset_standard"])
    parser.add_argument("--proprio-preprocess", choices=["none", "xvla_zero_gripper"])
    parser.add_argument("--init-checkpoint", type=str)
    parser.add_argument("--stats-checkpoint", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    cfg = _apply_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()
