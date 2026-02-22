# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
import math
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.manual_cast

from torch import Tensor

try:
    from timm.layers import DropPath, trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import DropPath, trunc_normal_

# Dtype debug logging — enable with DEBUG_COMFYUI_SAM3=1
_SAM3_DEBUG = os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes")
_sam3_log = logging.getLogger("sam3")

def _dtype_debug(label, **tensors):
    """Log dtype/shape of tensors at component boundaries."""
    if not _SAM3_DEBUG:
        return
    parts = [f"{label}:"]
    for name, t in tensors.items():
        if t is None:
            parts.append(f"  {name}=None")
        elif isinstance(t, (list, tuple)):
            dtypes = [x.dtype for x in t if hasattr(x, 'dtype')]
            parts.append(f"  {name}=[{', '.join(str(d) for d in dtypes)}]")
        elif hasattr(t, 'dtype'):
            parts.append(f"  {name}={t.dtype} {list(t.shape)}")
    _sam3_log.warning(" ".join(parts))

def _build_linear_stack(input_dim, hidden_dim, output_dim, num_layers, *, dtype, device, operations):
    h = [hidden_dim] * (num_layers - 1)
    return nn.ModuleList(
        operations.Linear(n, k, dtype=dtype, device=device)
        for n, k in zip([input_dim] + h, h + [output_dim])
    )

# ---------------------------------------------------------------------------
# Device cache mixin
# ---------------------------------------------------------------------------

class _DeviceCacheMixin:
    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)


# ---------------------------------------------------------------------------
# VitMlp — replacement for timm.layers.Mlp using operations.Linear
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# VitMlp — replacement for timm.layers.Mlp using operations.Linear
# ---------------------------------------------------------------------------

class VitMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=(0.0, 0.0),
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, dtype=dtype, device=device)
        self.act = act_layer()
        drop1 = drop[0] if isinstance(drop, (tuple, list)) else drop
        drop2 = drop[1] if isinstance(drop, (tuple, list)) else drop
        self.drop1 = nn.Dropout(drop1)
        self.fc2 = operations.Linear(hidden_features, out_features, dtype=dtype, device=device)
        self.drop2 = nn.Dropout(drop2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------------------------------------------------------
# DotProductScoring (from model/model_misc.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MLP (from model/model_misc.py)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = _build_linear_stack(
            input_dim, hidden_dim, output_dim, num_layers,
            dtype=dtype, device=device, operations=operations,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


# ---------------------------------------------------------------------------
# SamMLP (from sam/mask_decoder.py — renamed to avoid conflict with MLP above)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SamMLP (from sam/mask_decoder.py — renamed to avoid conflict with MLP above)
# ---------------------------------------------------------------------------

class SamMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = _build_linear_stack(
            input_dim, hidden_dim, output_dim, num_layers,
            dtype=dtype, device=device, operations=operations,
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# TransformerWrapper (from model/model_misc.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TransformerWrapper (from model/model_misc.py)
# ---------------------------------------------------------------------------

class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",
        pos_enc_at_input_dec=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type
        self.d_model = d_model


# ---------------------------------------------------------------------------
# PositionEmbeddingSine (from model/position_encoding.py) — no learnable params
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PositionEmbeddingSine (from model/position_encoding.py) — no learnable params
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        precompute_resolution: Optional[int] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}
        if precompute_resolution is not None:
            precompute_sizes = [
                (precompute_resolution // 4, precompute_resolution // 4),
                (precompute_resolution // 8, precompute_resolution // 8),
                (precompute_resolution // 16, precompute_resolution // 16),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            for size in precompute_sizes:
                tensors = torch.zeros((1, 1) + size)
                self.forward(tensors)
                self.cache[size] = self.cache[size].clone().detach()

    def _encode_xy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        if cache_key is not None:
            self.cache[cache_key] = pos[0]
        return pos


# ---------------------------------------------------------------------------
# PositionEmbeddingRandom (from sam/prompt_encoder.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PositionEmbeddingRandom (from sam/prompt_encoder.py)
# ---------------------------------------------------------------------------

class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=self.positional_encoding_gaussian_matrix.dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)


# ---------------------------------------------------------------------------
# ViT utility functions (from model/vitdet.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ViT utility functions (from model/vitdet.py)
# ---------------------------------------------------------------------------

def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
            align_corners=False,
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0).to(rel_pos.dtype)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def get_abs_pos(
    abs_pos: Tensor,
    has_cls_token: bool,
    hw: Tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> Tensor:
    if retain_cls_token:
        assert has_cls_token
    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num
    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile(
                [1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])]
            )[:, :, :h, :w]
        else:
            new_abs_pos = F.interpolate(
                new_abs_pos.float(), size=(h, w), mode="bicubic", align_corners=False,
            ).to(abs_pos.dtype)
        if not retain_cls_token:
            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            assert has_cls_token
            return torch.cat(
                [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)],
                dim=1,
            )
    else:
        if not retain_cls_token:
            return abs_pos.reshape(1, h, w, -1)
        else:
            assert has_cls_token
            return torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(
    q: Tensor,
    k: Tensor,
    q_hw: Tuple[int, int],
    k_hw: Tuple[int, int],
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    rescale: bool = False,
    relative_coords: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    q_h, q_w = q_hw
    k_h, k_w = k_hw
    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"
    if relative_coords is not None:
        Rh = rel_pos_h[relative_coords].to(q.dtype)
        Rw = rel_pos_w[relative_coords].to(q.dtype)
    else:
        Rh = get_rel_pos(q_h, k_h, rel_pos_h).to(q.dtype)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w).to(q.dtype)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale
    scale_ratio = new_scale / old_scale
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale
    eye_h = torch.eye(k_h, dtype=q.dtype, device=q.device)
    eye_w = torch.eye(k_w, dtype=q.dtype, device=q.device)
    eye_h = eye_h.view(1, k_h, 1, k_h).expand([B, k_h, k_w, k_h])
    eye_w = eye_w.view(1, 1, k_w, k_w).expand([B, k_h, k_w, k_w])
    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(
        B, k_h * k_w, -1
    )
    return q, k


# ---------------------------------------------------------------------------
# PatchEmbed (from model/vitdet.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PatchEmbed (from model/vitdet.py)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.proj = operations.Conv2d(
            in_chans, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
            dtype=dtype, device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        return x


# ---------------------------------------------------------------------------
# ViTAttention (from model/vitdet.py — renamed from Attention)
# ---------------------------------------------------------------------------

