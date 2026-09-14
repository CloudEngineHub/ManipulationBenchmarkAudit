from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


_DINO_MODEL_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}

_IMAGE_NORMS = {
    "dinov2": (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
    "imagenet": (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
    "clip": (
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ),
}


class RandomCropEraseAugmentation:
    def __init__(self, size: int = 224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.6, 1.0), antialias=True),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 1.0), value="random"),
            ]
        )

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(img_tensor)


class WidowXViTSMLPPolicy(nn.Module):
    """Single-camera DINOv2 ViT + task-conditioned MLP action-chunk policy."""

    def __init__(self, cfg: dict[str, Any], num_tasks: int) -> None:
        super().__init__()
        self.chunk_size = int(cfg["chunk_size"])
        self.num_tasks = int(num_tasks)
        self.dino_model = str(cfg.get("dino_model", "dinov2_vits14"))
        self.action_dim = int(cfg.get("action_dim", 7))
        self.train_visual_encoder = bool(cfg.get("train_visual_encoder", True))
        self.use_proprio = bool(cfg.get("use_proprio", False))
        self.image_aug = str(cfg.get("image_aug", "random_crop_erasing"))
        self.proprio_dim = int(cfg.get("proprio_dim", 20))
        self.visual_feature_mode = str(cfg.get("visual_feature_mode", "cls")).lower()
        self.patch_pool_size = int(cfg.get("patch_pool_size", 4))
        task_dim = int(cfg.get("task_dim", 32))
        hidden_dim = int(cfg.get("hidden_dim", 1024))

        if self.dino_model not in _DINO_MODEL_DIMS:
            raise ValueError(f"Unsupported dino_model={self.dino_model}; valid={sorted(_DINO_MODEL_DIMS)}")
        if self.visual_feature_mode not in {"cls", "patch_pool", "cls_patch_pool"}:
            raise ValueError(
                "visual_feature_mode must be one of: cls, patch_pool, cls_patch_pool"
            )
        if self.patch_pool_size <= 0:
            raise ValueError("patch_pool_size must be positive")
        raw_dino_hub_dir = os.environ.get("DSD_DINO_HUB_DIR", cfg.get("dino_hub_dir"))
        if raw_dino_hub_dir is None:
            raise ValueError("Set dino_hub_dir in the config or DSD_DINO_HUB_DIR in the environment")
        dino_hub_dir = Path(os.path.expanduser(os.path.expandvars(str(raw_dino_hub_dir))))
        if not dino_hub_dir.is_dir():
            raise FileNotFoundError(
                f"DINOv2 hub checkout not found at {dino_hub_dir}. "
                "Set DSD_DINO_HUB_DIR to a local facebookresearch/dinov2 checkout."
            )
        os.environ.setdefault("TORCH_HOME", str(dino_hub_dir.parent))
        torch.hub.set_dir(str(dino_hub_dir.parent))
        self.visual_encoder = torch.hub.load(
            str(dino_hub_dir),
            self.dino_model,
            pretrained=bool(cfg.get("pretrained_visual", True)),
            source="local",
        )
        base_feat_dim = _DINO_MODEL_DIMS[self.dino_model]
        if self.visual_feature_mode == "cls":
            feat_dim = base_feat_dim
        elif self.visual_feature_mode == "patch_pool":
            feat_dim = base_feat_dim * self.patch_pool_size * self.patch_pool_size
        else:
            feat_dim = base_feat_dim * (1 + self.patch_pool_size * self.patch_pool_size)
        if self.train_visual_encoder:
            self.visual_encoder.train()
            for p in self.visual_encoder.parameters():
                p.requires_grad = True
        else:
            self.visual_encoder.eval()
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        input_dim = feat_dim + task_dim
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )
            input_dim += 64
        else:
            self.proprio_encoder = None

        self.task_embed = nn.Embedding(self.num_tasks, task_dim)
        with torch.no_grad():
            scale = float(cfg.get("task_embedding_scale", 10.0))
            self.task_embed.weight.copy_(F.normalize(self.task_embed.weight, dim=-1) * scale)

        self.action_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim * self.chunk_size),
        )

        image_norm = str(cfg.get("image_norm", "dinov2")).lower()
        if image_norm not in _IMAGE_NORMS:
            raise ValueError(f"Unsupported image_norm={image_norm}; valid={sorted(_IMAGE_NORMS)}")
        mean, std = _IMAGE_NORMS[image_norm]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.random_crop_erase_aug = RandomCropEraseAugmentation(size=224)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        if self.training and self.image_aug == "random_crop_erasing":
            image = self.random_crop_erase_aug(image)
        image = (image - self.mean) / self.std
        if self.visual_feature_mode == "cls":
            if self.train_visual_encoder:
                feat = self.visual_encoder(image)
            else:
                with torch.no_grad():
                    feat = self.visual_encoder(image)
            return F.normalize(feat, dim=-1)

        if self.train_visual_encoder:
            features = self.visual_encoder.forward_features(image)
        else:
            with torch.no_grad():
                features = self.visual_encoder.forward_features(image)
        patch_tokens = features["x_norm_patchtokens"]
        side = int(patch_tokens.shape[1] ** 0.5)
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(f"Expected square DINO patch grid, got {patch_tokens.shape[1]} patches")
        patch_grid = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            side,
            side,
        )
        pooled = F.adaptive_avg_pool2d(
            patch_grid,
            output_size=(self.patch_pool_size, self.patch_pool_size),
        ).flatten(1)
        pooled = F.normalize(pooled, dim=-1)
        if self.visual_feature_mode == "patch_pool":
            return pooled
        cls = F.normalize(features["x_norm_clstoken"], dim=-1)
        feat = torch.cat([cls, pooled], dim=-1)
        return F.normalize(feat, dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        task_id: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self._encode_image(image)
        if self.use_proprio:
            if proprio is None:
                raise ValueError("proprio is required when use_proprio=True")
            if self.proprio_encoder is None:
                raise RuntimeError("proprio_encoder is missing")
            feat = torch.cat([feat, self.proprio_encoder(proprio)], dim=-1)
        task_feat = self.task_embed(task_id).to(dtype=feat.dtype)
        feat = torch.cat([feat, task_feat], dim=-1)
        out = self.action_head(feat)
        return out.view(out.shape[0], self.chunk_size, self.action_dim)

    def infer_chunk(
        self,
        image_np: np.ndarray,
        task_id: int,
        proprio_np: np.ndarray | None = None,
    ) -> np.ndarray:
        device = next(self.parameters()).device
        image = torch.from_numpy(np.ascontiguousarray(image_np)).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        task = torch.tensor([task_id], device=device, dtype=torch.long)
        proprio = None
        if self.use_proprio:
            if proprio_np is None:
                raise ValueError("proprio_np is required when use_proprio=True")
            proprio = torch.from_numpy(np.asarray(proprio_np, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = self.forward(image=image, task_id=task, proprio=proprio)
        return pred[0].detach().cpu().numpy()
