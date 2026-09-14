from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import cfg_get, require_torch, require_torchvision

try:  # Keep module importable on machines that only inspect configs/results.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on runtime env
    torch = None
    nn = None
    F = None


DINO_DIMS = {
    "dinov2_vitb14": 768,
    "dinov2_vits14": 384,
}


def _image_stats(kind: str) -> tuple[list[float], list[float]]:
    if kind == "clip":
        return [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    if kind == "imagenet":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(f"unknown image_norm={kind!r}")


if nn is not None:

    class MildCropErasing(nn.Module):
        def __init__(
            self,
            *,
            crop_scale_min: float,
            erasing_p: float,
            erasing_scale_min: float,
            erasing_scale_max: float,
        ) -> None:
            super().__init__()
            require_torchvision()
            from torchvision.transforms import v2 as transforms

            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224, scale=(float(crop_scale_min), 1.0), antialias=True),
                    transforms.RandomErasing(
                        p=float(erasing_p),
                        scale=(float(erasing_scale_min), float(erasing_scale_max)),
                        ratio=(0.3, 1.0),
                        value="random",
                    ),
                ]
            )

        def forward(self, x: Any) -> Any:
            return self.transform(x)


    class TwoViewTaskPolicy(nn.Module):
        def __init__(
            self,
            *,
            action_dim: int,
            chunk_size: int,
            num_tasks: int,
            state_dim: int,
            use_proprio: bool,
            vision_backbone: str,
            train_visual_encoder: bool,
            pretrained_visual: bool,
            image_norm: str,
            image_aug: str,
            flip_images: bool = False,
            resize_antialias: bool = True,
            mild_crop_erasing_crop_scale_min: float = 0.85,
            mild_crop_erasing_erasing_p: float = 0.25,
            mild_crop_erasing_erasing_scale_min: float = 0.01,
            mild_crop_erasing_erasing_scale_max: float = 0.05,
            task_embedding_scale: float = 1.0,
            dino_model: str = "dinov2_vitb14",
            dino_feature_mode: str = "cls",
            dino_hub_dir: str | None = None,
            use_addition: bool = False,
            hidden_dim: int = 1024,
            head_depth: int = 2,
            task_embedding_name: str = "task_embed",
            state_mean: np.ndarray | None = None,
            state_std: np.ndarray | None = None,
        ) -> None:
            super().__init__()
            self.action_dim = int(action_dim)
            self.chunk_size = int(chunk_size)
            self.num_tasks = int(num_tasks)
            self.use_proprio = bool(use_proprio)
            self.train_visual_encoder = bool(train_visual_encoder)
            self.vision_backbone = str(vision_backbone)
            self.dino_feature_mode = str(dino_feature_mode)
            self.use_addition = bool(use_addition)
            self.flip_images = bool(flip_images)
            self.resize_antialias = bool(resize_antialias)
            self.image_aug = str(image_aug or "none")
            if self.image_aug == "vincent":
                self.image_aug = "mild_crop_erasing"
            if self.image_aug not in {"none", "mild_crop_erasing"}:
                raise ValueError(f"unknown image_aug={self.image_aug!r}")

            mean, std = _image_stats(image_norm)
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

            if self.vision_backbone == "dino":
                if dino_model not in DINO_DIMS:
                    raise ValueError(f"unsupported dino_model={dino_model!r}")
                hub_root = dino_hub_dir
                if hub_root is None:
                    import os

                    hub_root = str(Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser() / "hub")
                self.visual_encoder = torch.hub.load(
                    str((Path(hub_root) / "facebookresearch_dinov2_main").resolve()),
                    dino_model,
                    pretrained=bool(pretrained_visual),
                    source="local",
                )
                feat_dim = DINO_DIMS[dino_model]
                if self.dino_feature_mode == "patch_pool4":
                    feat_dim *= 17
                elif self.dino_feature_mode != "cls":
                    raise ValueError(f"unsupported dino_feature_mode={self.dino_feature_mode!r}")
            elif self.vision_backbone in {"resnet", "resnet18"}:
                tv = require_torchvision()
                weights = tv.models.ResNet18_Weights.DEFAULT if pretrained_visual else None
                backbone = tv.models.resnet18(weights=weights)
                feat_dim = int(backbone.fc.in_features)
                backbone.fc = nn.Identity()
                self.visual_encoder = backbone
            else:
                raise ValueError(f"unknown vision_backbone={self.vision_backbone!r}")

            if self.train_visual_encoder:
                self.visual_encoder.train()
                for p in self.visual_encoder.parameters():
                    p.requires_grad = True
            else:
                self.visual_encoder.eval()
                for p in self.visual_encoder.parameters():
                    p.requires_grad = False

            input_dim = 2 * feat_dim
            if self.use_proprio:
                self.proprio_encoder = nn.Sequential(nn.Linear(int(state_dim), 64), nn.ReLU(), nn.Linear(64, 64))
                input_dim += 64
            else:
                self.proprio_encoder = None

            task_dim = input_dim if self.use_addition else 32
            if task_embedding_name not in {"task_embed", "task_embedding"}:
                raise ValueError(f"unsupported task_embedding_name={task_embedding_name!r}")
            self.task_embedding_name = task_embedding_name
            task_module = nn.Embedding(self.num_tasks, task_dim)
            setattr(self, task_embedding_name, task_module)
            with torch.no_grad():
                task_module.weight.copy_(F.normalize(task_module.weight, dim=-1) * float(task_embedding_scale))
            if self.use_addition:
                if task_dim != input_dim:
                    raise ValueError("task addition requires task_dim == input_dim")
            else:
                input_dim += task_dim

            layers: list[Any] = [nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()]
            for _ in range(int(head_depth) - 1):
                layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU()])
            layers.append(nn.Linear(int(hidden_dim), self.action_dim * self.chunk_size))
            self.action_head = nn.Sequential(*layers)
            self.aug = (
                MildCropErasing(
                    crop_scale_min=float(mild_crop_erasing_crop_scale_min),
                    erasing_p=float(mild_crop_erasing_erasing_p),
                    erasing_scale_min=float(mild_crop_erasing_erasing_scale_min),
                    erasing_scale_max=float(mild_crop_erasing_erasing_scale_max),
                )
                if self.image_aug == "mild_crop_erasing"
                else None
            )

            if state_mean is not None and state_std is not None:
                self.register_buffer("state_mean", torch.from_numpy(np.asarray(state_mean, dtype=np.float32)).view(1, -1))
                self.register_buffer("state_std", torch.from_numpy(np.asarray(state_std, dtype=np.float32)).view(1, -1))
            else:
                self.state_mean = None
                self.state_std = None

        def _encode(self, x: Any) -> Any:
            if self.resize_antialias:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            else:
                x = F.interpolate(x, size=(224, 224), mode="bilinear")
            if self.flip_images:
                x = torch.flip(x, dims=(-2, -1))
            if self.training and self.aug is not None:
                x = self.aug(x)
            x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
            if self.vision_backbone == "dino" and self.dino_feature_mode == "patch_pool4":
                with torch.set_grad_enabled(self.train_visual_encoder):
                    feats = self.visual_encoder.forward_features(x)
                cls = feats["x_norm_clstoken"].unsqueeze(1)
                patch = feats["x_norm_patchtokens"]
                side = int(patch.shape[1] ** 0.5)
                if side * side != patch.shape[1]:
                    raise ValueError(f"expected square ViT patch grid, got {patch.shape[1]} patches")
                b, _n, d = patch.shape
                patch = patch.transpose(1, 2).reshape(b, d, side, side)
                patch = F.adaptive_avg_pool2d(patch, output_size=(4, 4)).flatten(2).transpose(1, 2)
                return F.normalize(torch.cat([cls, patch], dim=1), dim=-1).flatten(1)
            with torch.set_grad_enabled(self.train_visual_encoder):
                feat = self.visual_encoder(x)
            if self.vision_backbone == "dino":
                feat = F.normalize(feat, dim=-1)
            return feat

        def forward(self, view_a: Any, view_b: Any, task_id: Any, state: Any | None = None) -> Any:
            feat = torch.cat([self._encode(view_a), self._encode(view_b)], dim=-1)
            if self.use_proprio:
                if state is None:
                    raise ValueError("state is required when use_proprio=true")
                if self.state_mean is not None and self.state_std is not None:
                    state = (state - self.state_mean.to(state.device, state.dtype)) / self.state_std.to(state.device, state.dtype)
                feat = torch.cat([feat, self.proprio_encoder(state)], dim=-1)
            task_module = getattr(self, self.task_embedding_name)
            task = task_module(task_id.to(feat.device, dtype=torch.long)).to(feat.dtype)
            feat = feat + task if self.use_addition else torch.cat([feat, task], dim=-1)
            out = self.action_head(feat)
            return out.view(out.shape[0], self.chunk_size, self.action_dim)

else:

    class TwoViewTaskPolicy:  # pragma: no cover - depends on runtime env
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def build_policy_from_cfg(
    cfg: dict[str, Any],
    *,
    benchmark: str,
    num_tasks: int,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
) -> Any:
    require_torch()
    if benchmark == "libero":
        state_dim = 8
        image_norm = "imagenet"
        hidden_dim = 512
        head_depth = 2
        use_addition = False
        task_embedding_scale = 1.0
        task_embedding_name = "task_embedding"
        flip_images = bool(cfg_get(cfg, "flip_images", False))
        resize_antialias = False
        aug_params = {
            "mild_crop_erasing_crop_scale_min": float(cfg_get(cfg, "mild_crop_erasing_crop_scale_min", 0.85)),
            "mild_crop_erasing_erasing_p": float(cfg_get(cfg, "mild_crop_erasing_erasing_p", 0.25)),
            "mild_crop_erasing_erasing_scale_min": float(cfg_get(cfg, "mild_crop_erasing_erasing_scale_min", 0.01)),
            "mild_crop_erasing_erasing_scale_max": float(cfg_get(cfg, "mild_crop_erasing_erasing_scale_max", 0.05)),
        }
    elif benchmark == "calvin":
        state_dim = 7
        image_norm = "clip"
        hidden_dim = 1024
        head_depth = 4
        use_addition = bool(cfg_get(cfg, "use_addition", False))
        task_embedding_scale = float(cfg_get(cfg, "task_embedding_scale", 10.0))
        task_embedding_name = "task_embed"
        flip_images = False
        resize_antialias = True
        aug_params = {
            "mild_crop_erasing_crop_scale_min": 0.6,
            "mild_crop_erasing_erasing_p": 0.5,
            "mild_crop_erasing_erasing_scale_min": 0.02,
            "mild_crop_erasing_erasing_scale_max": 0.1,
        }
    else:
        raise ValueError(f"unknown benchmark={benchmark!r}")
    return TwoViewTaskPolicy(
        action_dim=7,
        chunk_size=int(cfg["chunk_size"]),
        num_tasks=int(num_tasks),
        state_dim=state_dim,
        use_proprio=bool(cfg_get(cfg, "use_proprio", False)),
        vision_backbone=str(cfg_get(cfg, "vision_backbone", "dino")),
        train_visual_encoder=bool(cfg_get(cfg, "train_visual_encoder", False)),
        pretrained_visual=bool(cfg_get(cfg, "pretrained_visual", True)),
        image_norm=str(cfg_get(cfg, "image_norm", image_norm)),
        image_aug=str(cfg_get(cfg, "image_aug", "none")),
        flip_images=flip_images,
        resize_antialias=resize_antialias,
        **aug_params,
        task_embedding_scale=task_embedding_scale,
        dino_model=str(cfg_get(cfg, "dino_model", "dinov2_vitb14")),
        dino_feature_mode=str(cfg_get(cfg, "dino_feature_mode", "cls")),
        dino_hub_dir=cfg_get(cfg, "dino_hub_dir", None),
        use_addition=use_addition,
        hidden_dim=hidden_dim,
        head_depth=head_depth,
        task_embedding_name=task_embedding_name,
        state_mean=state_mean,
        state_std=state_std,
    )
