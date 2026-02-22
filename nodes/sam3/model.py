# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# ComfyUI-native model module.
# Consolidated from: model/*.py, sam/mask_decoder.py, sam/prompt_encoder.py

import datetime
import logging
import math
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.model_management

ops = comfy.ops.manual_cast

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


def is_image_type(resource_path):
    """Return True if resource_path points to a single image file."""
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return (
        isinstance(resource_path, str)
        and os.path.splitext(resource_path)[-1].lower() in _IMAGE_EXTS
    )


try:
    from timm.layers import DropPath, trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import DropPath, trunc_normal_

from copy import copy
from torch import Tensor
from torchvision.ops import masks_to_boxes
from torchvision.ops.roi_align import RoIAlign
from tqdm.auto import tqdm
from typing_extensions import override

from .attention import (
    sam3_attention,
    SplitMultiheadAttention,
    Attention as SAMAttention,
    RoPEAttention,
    TwoWayAttentionBlock,
    TwoWayTransformer,
    MLPBlock,
    LayerNorm2d,
    compute_axial_cis,
    apply_rotary_enc,
)
from .text_encoder import LayerScale
from .utils import (
    box_cxcywh_to_xyxy,
    box_xywh_to_cxcywh,
    box_xyxy_to_xywh,
    fast_diag_box_iou,
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    gen_sineembed_for_position,
    get_valid_ratio,
    get_1d_sine_pe,
    select_closest_cond_frames,
    fill_holes_in_mask_scores,
    mask_to_box,
    SAM3Output,
    Prompt,
    concat_padded_sequences,
    BatchedDatapoint,
    FindStage,
    convert_my_tensors,
    copy_data_to_device,
    load_resource_as_video_frames,
    IMAGE_EXTS,
    rle_encode,
    load_video_frames,
    SAM2Transforms,
)
from . import perflib
from .perflib import mask_iou
from .perflib import masks_to_boxes as perf_masks_to_boxes

log = logging.getLogger("sam3")
logger = logging.getLogger(__name__)


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

class DotProductScoring(nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp
        self.prompt_proj = operations.Linear(d_model, d_proj, dtype=dtype, device=device)
        self.hs_proj = operations.Linear(d_model, d_proj, dtype=dtype, device=device)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        is_valid = (~prompt_mask).to(prompt.dtype).permute(1, 0)[..., None]
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)
        proj_hs = self.hs_proj(hs)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)
        return scores


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
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            operations.Linear(n, k, dtype=dtype, device=device)
            for n, k in zip([input_dim] + h, h + [output_dim])
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
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            operations.Linear(n, k, dtype=dtype, device=device)
            for n, k in zip([input_dim] + h, h + [output_dim])
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

class ViTAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings and 2d-rope."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        cls_token: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_interp: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cls_token = cls_token

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        self._setup_rel_pos(rel_pos_zero_init)
        self._setup_rope_freqs()

    def _setup_rel_pos(self, rel_pos_zero_init: bool = True) -> None:
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            return
        assert self.input_size is not None
        assert self.cls_token is False, "not supported"
        self.rel_pos_h = nn.Parameter(
            torch.zeros(2 * self.input_size[0] - 1, self.head_dim)
        )
        self.rel_pos_w = nn.Parameter(
            torch.zeros(2 * self.input_size[1] - 1, self.head_dim)
        )
        if not rel_pos_zero_init:
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)
        H, W = self.input_size
        q_coords = torch.arange(H)[:, None]
        k_coords = torch.arange(W)[None, :]
        relative_coords = (q_coords - k_coords) + (H - 1)
        self.register_buffer("relative_coords", relative_coords.long())

    def _setup_rope_freqs(self) -> None:
        if not self.use_rope:
            self.freqs_cis = None
            return
        assert self.input_size is not None
        if self.rope_pt_size is None:
            self.rope_pt_size = self.input_size
        self.compute_cis = partial(
            compute_axial_cis, dim=self.head_dim, theta=self.rope_theta,
        )
        scale_pos = 1.0
        if self.rope_interp:
            scale_pos = self.rope_pt_size[0] / self.input_size[0]
        freqs_cis = self.compute_cis(
            end_x=self.input_size[0], end_y=self.input_size[1], scale_pos=scale_pos,
        )
        if self.cls_token:
            t = torch.zeros(self.head_dim // 2, dtype=torch.float32, device=freqs_cis.device)
            cls_freqs_cis = torch.polar(torch.ones_like(t), t)[None, :]
            freqs_cis = torch.cat([cls_freqs_cis, freqs_cis], dim=0)
        self.register_buffer("freqs_cis", freqs_cis)

    def _apply_rope(self, q, k) -> Tuple[Tensor, Tensor]:
        if not self.use_rope:
            return q, k
        assert self.freqs_cis is not None
        return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)

    def forward(self, x: Tensor) -> Tensor:
        s = 1 if self.cls_token else 0
        if x.ndim == 4:
            B, H, W, _ = x.shape
            assert s == 0
            L = H * W
            ndim = 4
        else:
            assert x.ndim == 3
            B, L, _ = x.shape
            ndim = 3
            H = W = math.sqrt(L - s)

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            q, k = concat_rel_pos(
                q.flatten(0, 1), k.flatten(0, 1),
                (H, W), x.shape[1:3],
                self.rel_pos_h, self.rel_pos_w,
                rescale=True, relative_coords=self.relative_coords,
            )
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)

        # Both paths use optimized_attention (skip_reshape since q/k/v are [B, H, L, D])
        x = sam3_attention(q, k, v, self.num_heads)

        if ndim == 4:
            x = (
                x.view(B, self.num_heads, H, W, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Block (from model/vitdet.py)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer blocks with support of window attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_rope: bool = False,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        use_ve_rope: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
        init_values: Optional[float] = None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.attn = ViTAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
            cls_token=cls_token,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = VitMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.ls1(self.attn(x))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
        return x


# ---------------------------------------------------------------------------
# ViT (from model/vitdet.py)
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """Vision Transformer (ViT) backbone for object detection (ViTDet)."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        rel_pos_blocks: Union[Tuple[int, ...], bool] = (2, 5, 8, 11),
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_att_blocks: Tuple[int, ...] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: Optional[int] = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        init_values: Optional[float] = None,
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        window_block_indexes = [i for i in range(depth) if i not in global_att_blocks]
        self.full_attn_ids = list(global_att_blocks)
        self.rel_pos_blocks = [False] * depth
        if isinstance(rel_pos_blocks, bool) and rel_pos_blocks:
            self.rel_pos_blocks = [True] * depth
        else:
            for i in rel_pos_blocks:
                self.rel_pos_blocks[i] = True

        self.retain_cls_token = retain_cls_token
        if self.retain_cls_token:
            assert pretrain_use_cls_token
            assert len(window_block_indexes) == 0, "windowing not supported with cls token"
            assert sum(self.rel_pos_blocks) == 0, "rel pos not supported with cls token"
            scale = embed_dim**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias_patch_embed,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.tile_abs_pos = tile_abs_pos
        self.use_abs_pos = use_abs_pos
        if self.tile_abs_pos:
            assert self.use_abs_pos
        if self.use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth, device="cpu")]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                act_layer=act_layer,
                use_rel_pos=self.rel_pos_blocks[i],
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=(
                    (window_size, window_size)
                    if rope_pt_size is None
                    else (rope_pt_size, rope_pt_size)
                ),
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            self.blocks.append(block)

        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

        self.ln_pre = (
            operations.LayerNorm(embed_dim, dtype=dtype, device=device)
            if ln_pre
            else nn.Identity()
        )
        self.ln_post = (
            operations.LayerNorm(embed_dim, dtype=dtype, device=device)
            if ln_post
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _dtype_debug("ViT.forward IN", x=x)
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            x = torch.cat([self.class_embedding.to(x.dtype), x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token,
                (h, w), self.retain_cls_token, tiling=self.tile_abs_pos,
            ).to(x.dtype)

        x = self.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)
                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    h = w = math.sqrt(feats.shape[1])
                    feats = feats.reshape(
                        feats.shape[0], h, w, feats.shape[-1]
                    ).permute(0, 3, 1, 2)
                outputs.append(feats)

        _dtype_debug("ViT.forward OUT", features=outputs)
        return outputs

    def get_layer_id(self, layer_name: str) -> int:
        num_layers = self.get_num_layers()
        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("ln_pre") != -1:
            return 0
        elif layer_name.find("pos_embed") != -1 or layer_name.find("cls_token") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)


# ---------------------------------------------------------------------------
# TransformerEncoderLayer (from model/encoder.py)
# ---------------------------------------------------------------------------

class _TransformerSelfCrossAttnLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = operations.Linear(d_model, dim_feedforward, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = operations.Linear(dim_feedforward, d_model, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(d_model, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(d_model, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(d_model, dtype=dtype, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx = None

    def _cross_attn(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        attn_bias=None,
    ):
        if attn_bias is None:
            return self.cross_attn_image(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )[0]
        return self.cross_attn_image(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
        )[0]

    def forward_post(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        pos=None, query_pos=None, attn_bias=None, **kwargs,
    ):
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self._cross_attn(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        pos=None, query_pos=None, attn_bias=None, **kwargs,
    ):
        if dac:
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2:]
            tgt = tgt[:tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        tgt2 = self._cross_attn(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoderLayer(_TransformerSelfCrossAttnLayer):
    def forward(
        self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        pos=None, query_pos=None,
    ):
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt, memory, dac=dac, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos, query_pos=query_pos,
        )


# ---------------------------------------------------------------------------
# TransformerEncoder (from model/encoder.py)
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        with torch.no_grad():
            reference_points_list = []
            for lvl, (H_, W_) in enumerate(spatial_shapes):
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                    torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                )
                ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
                ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            reference_points = torch.cat(reference_points_list, 1)
            reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        assert len(srcs) == self.num_feature_levels
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1).to(pos_embed.dtype)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2), device=src_flatten.device,
            )
        return src_flatten, mask_flatten, lvl_pos_embed_flatten, level_start_index, valid_ratios, spatial_shapes

    def forward(
        self, src, src_key_padding_masks=None, pos=None,
        prompt=None, prompt_key_padding_mask=None, encoder_extra_kwargs=None,
    ):
        assert len(src) == self.num_feature_levels
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels
        (
            src_flatten, key_padding_masks_flatten, lvl_pos_embed_flatten,
            level_start_index, valid_ratios, spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src_flatten.device
        )
        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}
            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten
            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = layer(**layer_kwargs)
        return (
            output.transpose(0, 1),
            (key_padding_masks_flatten.transpose(0, 1) if key_padding_masks_flatten is not None else None),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


# ---------------------------------------------------------------------------
# TransformerEncoderFusion (from model/encoder.py)
# ---------------------------------------------------------------------------

class TransformerEncoderFusion(TransformerEncoder):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: Optional[str] = None,
        dtype=None,
        device=None,
        operations=ops,
        **kwargs,
    ):
        super().__init__(layer, num_layers, d_model, num_feature_levels, **kwargs)
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = operations.Linear(d_model, d_model, dtype=dtype, device=device)
        self.pool_text_with_mask = pool_text_with_mask

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        return None

    def forward(
        self, src, prompt, src_key_padding_mask=None, src_pos=None,
        prompt_key_padding_mask=None, prompt_pos=None,
        feat_sizes=None, encoder_extra_kwargs=None,
    ):
        _dtype_debug("Encoder.forward IN", src=src, prompt=prompt)
        bs = src[0].shape[1]
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                    if src_key_padding_mask[i] is not None else None
                )
        else:
            assert all(x.dim == 4 for x in src)
        if self.add_pooled_text_to_img_feat:
            pooled_text = pool_text_feat(
                prompt, prompt_key_padding_mask, self.pool_text_with_mask
            )
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
            src = [x.add_(pooled_text) for x in src]
        (out, key_padding_masks_flatten, lvl_pos_embed_flatten,
         level_start_index, spatial_shapes, valid_ratios) = super().forward(
            src, src_key_padding_masks=src_key_padding_mask, pos=src_pos,
            prompt=prompt.transpose(0, 1),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        _dtype_debug("Encoder.forward OUT", memory=out, pos_embed=lvl_pos_embed_flatten, memory_text=prompt)
        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    if not pool_with_mask:
        return prompt.mean(dim=0)
    assert prompt_mask.dim() == 2
    is_valid = (~prompt_mask).to(prompt.dtype).permute(1, 0)[..., None]
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text


# ---------------------------------------------------------------------------
# TransformerDecoderLayer (from model/decoder.py)
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        cross_attention: nn.Module,
        n_heads: int,
        use_text_cross_attention: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = operations.LayerNorm(d_model, dtype=dtype, device=device)

        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = SplitMultiheadAttention(
                d_model, n_heads, dropout=dropout,
                dtype=dtype, device=device, operations=operations,
            )
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = operations.LayerNorm(d_model, dtype=dtype, device=device)

        self.self_attn = SplitMultiheadAttention(
            d_model, n_heads, dropout=dropout,
            dtype=dtype, device=device, operations=operations,
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = operations.LayerNorm(d_model, dtype=dtype, device=device)

        self.linear1 = operations.Linear(d_model, dim_feedforward, dtype=dtype, device=device)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = operations.Linear(dim_feedforward, d_model, dtype=dtype, device=device)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = operations.LayerNorm(d_model, dtype=dtype, device=device)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        input_dtype = tgt.dtype
        tgt = tgt.float()
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt.to(input_dtype)

    def forward(
        self, tgt=None, tgt_query_pos=None, tgt_query_sine_embed=None,
        tgt_key_padding_mask=None, tgt_reference_points=None,
        memory_text=None, text_attention_mask=None,
        memory=None, memory_key_padding_mask=None,
        memory_level_start_index=None, memory_spatial_shapes=None,
        memory_pos=None, self_attn_mask=None, cross_attn_mask=None,
        dac=False, dac_use_selfatt_ln=True, presence_token=None,
        identity=0.0, **kwargs,
    ):
        if self.self_attn is not None:
            if dac:
                assert tgt.shape[0] % 2 == 0
                num_o2o_queries = tgt.shape[0] // 2
                tgt_o2o = tgt[:num_o2o_queries]
                tgt_query_pos_o2o = tgt_query_pos[:num_o2o_queries]
                tgt_o2m = tgt[num_o2o_queries:]
            else:
                tgt_o2o = tgt
                tgt_query_pos_o2o = tgt_query_pos

            if presence_token is not None:
                tgt_o2o = torch.cat([presence_token, tgt_o2o], dim=0)
                tgt_query_pos_o2o = torch.cat(
                    [torch.zeros_like(presence_token), tgt_query_pos_o2o], dim=0
                )
                tgt_query_pos = torch.cat(
                    [torch.zeros_like(presence_token), tgt_query_pos], dim=0
                )

            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
            tgt2 = self.self_attn(q, k, tgt_o2o, attn_mask=self_attn_mask)[0]
            tgt_o2o = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    tgt_o2o = self.norm2(tgt_o2o)
                tgt = torch.cat((tgt_o2o, tgt_o2m), dim=0)
                if dac_use_selfatt_ln:
                    tgt = self.norm2(tgt)
            else:
                tgt = tgt_o2o
                tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text, memory_text, key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if presence_token is not None:
            presence_token_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = torch.cat([presence_token_mask, cross_attn_mask], dim=1)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1)
                if memory_key_padding_mask is not None else None
            ),
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:1]
            tgt = tgt[1:]
        return tgt, presence_token_out


# ---------------------------------------------------------------------------
# TransformerDecoder (from model/decoder.py)
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        interaction_layer,
        layer,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        num_o2m_queries: int = 0,
        dac: bool = False,
        boxRPB: str = "none",
        instance_query: bool = False,
        num_instances: int = 1,
        dac_use_selfatt_ln: bool = True,
        use_act_checkpoint: bool = False,
        compile_mode=None,
        presence_token: bool = False,
        clamp_presence_logits: bool = True,
        clamp_presence_logit_max_val: float = 10.0,
        use_normed_output_consistently: bool = True,
        separate_box_head_instance: bool = False,
        separate_norm_instance: bool = False,
        resolution: Optional[int] = None,
        stride: Optional[int] = None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.fine_layers = (
            get_clones(interaction_layer, num_layers) if interaction_layer is not None
            else [None] * num_layers
        )
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = operations.LayerNorm(d_model, dtype=dtype, device=device)
        self.return_intermediate = return_intermediate
        self.bbox_embed = MLP(d_model, d_model, 4, 3, dtype=dtype, device=device, operations=operations)
        self.query_embed = operations.Embedding(tot_num_queries, d_model, dtype=dtype, device=device)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        self.use_normed_output_consistently = use_normed_output_consistently

        self.instance_norm = (
            operations.LayerNorm(d_model, dtype=dtype, device=device)
            if separate_norm_instance else None
        )
        self.instance_bbox_embed = None
        if separate_box_head_instance:
            self.instance_bbox_embed = MLP(d_model, d_model, 4, 3, dtype=dtype, device=device, operations=operations)
        if instance_query:
            self.instance_query_embed = operations.Embedding(num_instances, d_model, dtype=dtype, device=device)
        self.box_refine = box_refine
        if box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            self.reference_points = operations.Embedding(num_queries, 4, dtype=dtype, device=device)
            if instance_query:
                self.instance_reference_points = operations.Embedding(num_instances, 4, dtype=dtype, device=device)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads
            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2, dtype=dtype, device=device, operations=operations)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2, dtype=dtype, device=device, operations=operations)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache = {}
            if resolution is not None and stride is not None:
                feat_size = resolution // stride
                self.compilable_stored_size = (feat_size, feat_size)

        self.roi_pooler = (
            RoIAlign(output_size=7, spatial_scale=1, sampling_ratio=-1, aligned=True)
            if interaction_layer is not None else None
        )
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.presence_token = None
        self.clamp_presence_logits = clamp_presence_logits
        self.clamp_presence_logit_max_val = clamp_presence_logit_max_val
        if presence_token:
            self.presence_token = operations.Embedding(1, d_model, dtype=dtype, device=device)
            self.presence_token_head = MLP(d_model, d_model, 1, 3, dtype=dtype, device=device, operations=operations)
            self.presence_token_out_norm = operations.LayerNorm(d_model, dtype=dtype, device=device)

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2, dtype=dtype, device=device, operations=operations)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln

        nn.init.normal_(self.query_embed.weight.data)
        if self.instance_query_embed is not None:
            nn.init.normal_(self.instance_query_embed.weight.data)

        assert self.roi_pooler is None
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        self.compile_mode = compile_mode
        self.compiled = False
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W, device):
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
        bs, num_queries, _ = boxes_xyxy.shape
        if self.compilable_cord_cache is None:
            self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
            self.compilable_stored_size = (H, W)
        if self.compilable_stored_size == (H, W):
            coords_h, coords_w = self.compilable_cord_cache
        else:
            if feat_size not in self.coord_cache:
                self.coord_cache[feat_size] = self._get_coords(H, W, reference_boxes.device)
            coords_h, coords_w = self.coord_cache[feat_size]
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)
        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8
            deltas_x_log = (
                torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / np.log2(8)
            )
            deltas_y_log = deltas_y * 8
            deltas_y_log = (
                torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / np.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)
        deltas_x = self.boxRPB_embed_x(x=deltas_x)
        deltas_y = self.boxRPB_embed_y(x=deltas_y)
        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
        B = B.flatten(2, 3)
        B = B.permute(0, 3, 1, 2)
        B = B.contiguous()
        return B

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        pos=None, reference_boxes=None, level_start_index=None,
        spatial_shapes=None, valid_ratios=None,
        memory_text=None, text_attention_mask=None,
        apply_dac=None, is_instance_prompt=False,
        decoder_extra_kwargs=None,
        obj_roi_memory_feat=None, obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        _dtype_debug("Decoder.forward IN", tgt=tgt, memory=memory, memory_text=memory_text)
        if memory_mask is not None:
            assert self.boxRPB == "none"
        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert (tgt.shape[0] == self.num_queries) or (
                self.use_instance_query
                and (tgt.shape[0] == self.instance_query_embed.num_embeddings)
            )
            tgt = tgt.repeat(2, 1, 1)
            if reference_boxes is not None:
                assert (reference_boxes.shape[0] == self.num_queries) or (
                    self.use_instance_query
                    and (reference_boxes.shape[0] == self.instance_query_embed.num_embeddings)
                )
                reference_boxes = reference_boxes.repeat(2, 1, 1)
        bs = tgt.shape[1]
        intermediate = []
        intermediate_presence_logits = []
        presence_feats = None
        if self.box_refine:
            if reference_boxes is None:
                reference_boxes = self.reference_points.weight.unsqueeze(1).to(tgt.dtype)
                reference_boxes = (
                    reference_boxes.repeat(2, bs, 1) if apply_dac
                    else reference_boxes.repeat(1, bs, 1)
                )
                reference_boxes = reference_boxes.sigmoid()
            intermediate_ref_boxes = [reference_boxes]
        else:
            reference_boxes = None
            intermediate_ref_boxes = None
        output = tgt
        presence_out = None
        if self.presence_token is not None and is_instance_prompt is False:
            presence_out = self.presence_token.weight[None].expand(1, bs, -1).to(tgt.dtype)
        box_head = self.bbox_embed
        if is_instance_prompt and self.instance_bbox_embed is not None:
            box_head = self.instance_bbox_embed
        out_norm = self.norm
        if is_instance_prompt and self.instance_norm is not None:
            out_norm = self.instance_norm
        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )
            query_pos = self.ref_point_head(query_sine_embed).to(output.dtype)
            if self.boxRPB != "none" and reference_boxes is not None:
                assert spatial_shapes.shape[0] == 1
                memory_mask = self._get_rpb_matrix(
                    reference_boxes, (spatial_shapes[0, 0], spatial_shapes[0, 1]),
                )
                memory_mask = memory_mask.flatten(0, 1).to(output.dtype)
            output, presence_out = layer(
                tgt=output, tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text, text_attention_mask=text_attention_mask,
                memory=memory, memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos, self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask, dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                presence_token=presence_out,
                **(decoder_extra_kwargs or {}),
                obj_roi_memory_feat=obj_roi_memory_feat,
                obj_roi_memory_mask=obj_roi_memory_mask,
            )
            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if box_head_trk is None:
                    if not self.use_normed_output_consistently:
                        delta_unsig = box_head(output)
                    else:
                        delta_unsig = box_head(out_norm(output))
                else:
                    Q_det = decoder_extra_kwargs["Q_det"]
                    assert output.size(0) >= Q_det
                    delta_unsig_det = self.bbox_embed(output[:Q_det])
                    delta_unsig_trk = box_head_trk(output[Q_det:])
                    delta_unsig = torch.cat([delta_unsig_det, delta_unsig_trk], dim=0)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_boxes = new_reference_points.detach()
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                raise NotImplementedError("not implemented yet")
            intermediate.append(out_norm(output))
            if self.presence_token is not None and is_instance_prompt is False:
                intermediate_layer_presence_logits = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)
                if self.clamp_presence_logits:
                    intermediate_layer_presence_logits.clamp(
                        min=-self.clamp_presence_logit_max_val,
                        max=self.clamp_presence_logit_max_val,
                    )
                intermediate_presence_logits.append(intermediate_layer_presence_logits)
                presence_feats = presence_out.clone()
        stacked_intermediate = torch.stack(intermediate)
        stacked_ref_boxes = torch.stack(intermediate_ref_boxes)
        stacked_presence = (
            torch.stack(intermediate_presence_logits)
            if self.presence_token is not None and is_instance_prompt is False else None
        )
        _dtype_debug("Decoder.forward OUT", output=stacked_intermediate, ref_boxes=stacked_ref_boxes, presence=stacked_presence)
        return (
            stacked_intermediate,
            stacked_ref_boxes,
            stacked_presence,
            presence_feats,
        )


# ---------------------------------------------------------------------------
# TransformerEncoderCrossAttention (from model/decoder.py)
# ---------------------------------------------------------------------------

class TransformerEncoderCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,
        remove_cross_attention_layers: Optional[list] = None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = operations.LayerNorm(d_model, dtype=dtype, device=device)
        self.pos_enc_at_input = pos_enc_at_input
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)
        self.batch_first = batch_first
        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)
        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None
                self.layers[i].dropout2 = None

    def forward(
        self, src, prompt, src_mask=None, prompt_mask=None,
        src_key_padding_mask=None, prompt_key_padding_mask=None,
        src_pos=None, prompt_pos=None, feat_sizes=None,
        num_obj_ptr_tokens: int = 0,
    ):
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = src[0], src_key_padding_mask[0], src_pos[0]
        assert src.shape[1] == prompt.shape[1]
        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            output = layer(
                tgt=output, memory=prompt, tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos, query_pos=src_pos, dac=False,
                attn_bias=None,
                **kwds,
            )
            normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
        }


# ---------------------------------------------------------------------------
# TransformerDecoderLayerv1 (from model/decoder.py)
# ---------------------------------------------------------------------------

class TransformerDecoderLayerv1(_TransformerSelfCrossAttnLayer):
    def forward(
        self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        pos=None, query_pos=None, attn_bias=None, **kwds,
    ):
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt, memory, dac=dac, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos, query_pos=query_pos, attn_bias=attn_bias, **kwds,
        )


# ---------------------------------------------------------------------------
# TransformerDecoderLayerv2 (from model/decoder.py)
# ---------------------------------------------------------------------------

class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    def __init__(self, cross_attention_first=False, *args, **kwds):
        super().__init__(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        if self.cross_attn_image is None:
            return tgt
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory, **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward_pre(
        self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        pos=None, query_pos=None, attn_bias=None, num_k_exclude_rope=0,
    ):
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None
        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args, **kwds):
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sam3DualViTDetNeck (from model/necks.py)
# ---------------------------------------------------------------------------

class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()
            if scale == 4.0:
                current.add_module("dconv_2x2_0", operations.ConvTranspose2d(
                    dim, dim // 2, kernel_size=2, stride=2, dtype=dtype, device=device,
                ))
                current.add_module("gelu", nn.GELU())
                current.add_module("dconv_2x2_1", operations.ConvTranspose2d(
                    dim // 2, dim // 4, kernel_size=2, stride=2, dtype=dtype, device=device,
                ))
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module("dconv_2x2", operations.ConvTranspose2d(
                    dim, dim // 2, kernel_size=2, stride=2, dtype=dtype, device=device,
                ))
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            current.add_module("conv_1x1", operations.Conv2d(
                out_dim, d_model, kernel_size=1, bias=use_bias, dtype=dtype, device=device,
            ))
            current.add_module("conv_3x3", operations.Conv2d(
                d_model, d_model, kernel_size=3, padding=1, bias=use_bias, dtype=dtype, device=device,
            ))
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            self.sam2_convs = deepcopy(self.convs)

    def forward(self, tensor_list):
        _dtype_debug("FPN_Neck.forward IN", input=tensor_list)
        xs = self.trunk(tensor_list)
        _dtype_debug("FPN_Neck.forward after trunk", trunk_out=xs)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        x = xs[-1]
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)
            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        _dtype_debug("FPN_Neck.forward OUT", sam3_out=sam3_out, sam3_pos=sam3_pos)
        return sam3_out, sam3_pos, sam2_out, sam2_pos


# ---------------------------------------------------------------------------
# SimpleMaskDownSampler (from model/memory.py)
# ---------------------------------------------------------------------------

class SimpleMaskDownSampler(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
        interpol_size=None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(operations.Conv2d(
                mask_in_chans, mask_out_chans,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dtype=dtype, device=device,
            ))
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans
        self.encoder.append(operations.Conv2d(
            mask_out_chans, embed_dim, kernel_size=1, dtype=dtype, device=device,
        ))
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            assert isinstance(self.interpol_size, (list, tuple))
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(self, x):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            _dtype = x.dtype
            x = F.interpolate(
                x.float(), size=self.interpol_size,
                align_corners=False, mode="bilinear", antialias=True,
            ).to(_dtype)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# CXBlock (from model/memory.py)
# ---------------------------------------------------------------------------

class CXBlock(nn.Module):
    """ConvNeXt Block."""

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.dwconv = operations.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding,
            groups=dim if use_dwconv else 1,
            dtype=dtype, device=device,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = operations.Linear(dim, 4 * dim, dtype=dtype, device=device)
        self.act = nn.GELU()
        self.pwconv2 = operations.Linear(4 * dim, dim, dtype=dtype, device=device)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.to(x.dtype) * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# SimpleFuser (from model/memory.py)
# ---------------------------------------------------------------------------

class SimpleFuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)
        if input_projection:
            assert dim is not None
            self.proj = operations.Conv2d(dim, dim, kernel_size=1, dtype=dtype, device=device)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# SimpleMaskEncoder (from model/memory.py)
# ---------------------------------------------------------------------------

class SimpleMaskEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = operations.Conv2d(in_dim, in_dim, kernel_size=1, dtype=dtype, device=device)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = operations.Conv2d(in_dim, out_dim, kernel_size=1, dtype=dtype, device=device)

    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        pix_feat = pix_feat.to(masks.device)
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x).to(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}


# ---------------------------------------------------------------------------
# MaskEncoder (from model/geometry_encoders.py)
# ---------------------------------------------------------------------------

class MaskEncoder(nn.Module):
    def __init__(self, mask_downsampler, position_encoding):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.position_encoding = position_encoding

    def forward(self, masks, *args, **kwargs):
        masks = self.mask_downsampler(masks)
        masks_pos = self.position_encoding(masks).to(masks.dtype)
        return masks, masks_pos


# ---------------------------------------------------------------------------
# SequenceGeometryEncoder (from model/geometry_encoders.py)
# ---------------------------------------------------------------------------

class SequenceGeometryEncoder(nn.Module):
    def __init__(
        self,
        encode_boxes_as_points: bool,
        points_direct_project: bool,
        points_pool: bool,
        points_pos_enc: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        mask_encoder: MaskEncoder = None,
        add_mask_label: bool = False,
        use_act_ckpt: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = operations.Embedding(num_labels, self.d_model, dtype=dtype, device=device)

        self.cls_embed = None
        if add_cls:
            self.cls_embed = operations.Embedding(1, self.d_model, dtype=dtype, device=device)

        assert points_direct_project or points_pos_enc or points_pool
        assert encode_boxes_as_points or boxes_direct_project or boxes_pos_enc or boxes_pool

        self.points_direct_project = None
        if points_direct_project:
            self.points_direct_project = operations.Linear(2, self.d_model, dtype=dtype, device=device)
        self.points_pool_project = None
        if points_pool:
            self.points_pool_project = operations.Linear(self.d_model, self.d_model, dtype=dtype, device=device)
        self.points_pos_enc_project = None
        if points_pos_enc:
            self.points_pos_enc_project = operations.Linear(self.d_model, self.d_model, dtype=dtype, device=device)

        self.boxes_direct_project = None
        self.boxes_pool_project = None
        self.boxes_pos_enc_project = None
        if not encode_boxes_as_points:
            if boxes_direct_project:
                self.boxes_direct_project = operations.Linear(4, self.d_model, dtype=dtype, device=device)
            if boxes_pool:
                self.boxes_pool_project = operations.Conv2d(
                    self.d_model, self.d_model, self.roi_size, dtype=dtype, device=device,
                )
            if boxes_pos_enc:
                self.boxes_pos_enc_project = operations.Linear(
                    self.d_model + 2, self.d_model, dtype=dtype, device=device,
                )

        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = operations.Linear(self.d_model, self.d_model, dtype=dtype, device=device)
            self.norm = operations.LayerNorm(self.d_model, dtype=dtype, device=device)

        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = operations.LayerNorm(self.d_model, dtype=dtype, device=device)

        self.encode = None
        if num_layers > 0:
            assert add_cls
            self.encode = get_clones(layer, num_layers)
            self.encode_norm = operations.LayerNorm(self.d_model, dtype=dtype, device=device)

        if mask_encoder is not None:
            assert isinstance(mask_encoder, MaskEncoder)
            if add_mask_label:
                self.mask_label_embed = operations.Embedding(2, self.d_model, dtype=dtype, device=device)
        self.add_mask_label = add_mask_label
        self.mask_encoder = mask_encoder

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]
        if self.points_direct_project is not None:
            proj = self.points_direct_project(points)
            assert points_embed is None
            points_embed = proj
        if self.points_pool_project is not None:
            grid = points.transpose(0, 1).unsqueeze(2)
            grid = (grid * 2) - 1
            sampled = torch.nn.functional.grid_sample(img_feats, grid, align_corners=False)
            assert list(sampled.shape) == [bs, self.d_model, n_points, 1]
            sampled = sampled.squeeze(-1).permute(2, 0, 1)
            proj = self.points_pool_project(sampled)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj
        if self.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(n_points, bs, enc_x.shape[-1])
            enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
            enc = torch.cat([enc_x, enc_y], -1)
            proj = self.points_pos_enc_project(enc)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj
        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        import torchvision
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]
        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes)
            assert boxes_embed is None
            boxes_embed = proj
        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
            if boxes_xyxy.device.type == "cuda":
                scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)
            else:
                scale = scale.to(device=boxes_xyxy.device)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            input_dtype = img_feats.dtype
            sampled = torchvision.ops.roi_align(
                img_feats.float(), boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
            )
            sampled = sampled.to(input_dtype)
            assert list(sampled.shape) == [bs * n_boxes, self.d_model, self.roi_size, self.roi_size]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj
        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
            proj = self.boxes_pos_enc_project(enc)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj
        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    def _encode_masks(self, masks, attn_mask, mask_labels, img_feats=None):
        n_masks, bs = masks.shape[:2]
        assert n_masks == 1
        assert list(attn_mask.shape) == [bs, n_masks]
        masks, pos = self.mask_encoder(masks=masks.flatten(0, 1).to(img_feats.dtype), pix_feat=img_feats)
        H, W = masks.shape[-2:]
        n_tokens_per_mask = H * W
        masks = masks + pos
        masks = masks.view(n_masks, bs, *masks.shape[1:]).flatten(-2)
        masks = masks.permute(0, 3, 1, 2).flatten(0, 1)
        attn_mask = attn_mask.repeat_interleave(n_tokens_per_mask, dim=1)
        if self.add_mask_label:
            masks = masks + self.mask_label_embed(mask_labels.long())
        return masks, attn_mask

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        points = geo_prompt.point_embeddings
        points_mask = geo_prompt.point_mask
        points_labels = geo_prompt.point_labels
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        masks = geo_prompt.mask_embeddings
        masks_mask = geo_prompt.mask_mask
        masks_labels = geo_prompt.mask_labels
        seq_first_img_feats = img_feats[-1]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )
        if self.points_pool_project or self.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat
        if self.encode_boxes_as_points:
            assert boxes is not None
            assert geo_prompt.box_mask is not None
            assert geo_prompt.box_labels is not None
            assert boxes.shape[-1] == 4
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)
            labels_tl = geo_prompt.box_labels + 2
            labels_br = geo_prompt.box_labels + 4
            points, _ = concat_padded_sequences(points, points_mask, top_left, boxes_mask)
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1), points_mask,
                labels_tl.unsqueeze(-1), boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)
            points, _ = concat_padded_sequences(points, points_mask, bottom_right, boxes_mask)
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1), points_mask,
                labels_br.unsqueeze(-1), boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)
        final_embeds, final_mask = self._encode_points(
            points=points, points_mask=points_mask,
            points_labels=points_labels, img_feats=img_feats,
        )
        if not self.encode_boxes_as_points:
            boxes_embeds, boxes_mask = self._encode_boxes(
                boxes=boxes, boxes_mask=boxes_mask,
                boxes_labels=boxes_labels, img_feats=img_feats,
            )
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, boxes_embeds, boxes_mask
            )
        if masks is not None and self.mask_encoder is not None:
            masks_embed, masks_mask = self._encode_masks(
                masks=masks, attn_mask=masks_mask,
                mask_labels=masks_labels, img_feats=img_feats,
            )
            if points.size(0) == boxes.size(0) == 0:
                return masks_embed, masks_mask
        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        if self.cls_embed is not None:
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1).to(final_embeds.dtype)
            cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=final_mask.device)
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )
        if self.final_proj is not None:
            final_embeds = self.norm(self.final_proj(final_embeds))
        if self.encode is not None:
            # Cast geometry embeddings to match visual features dtype.
            # Embedding layers output fp32 (integer indices bypass manual_cast).
            final_embeds = final_embeds.to(seq_first_img_feats.dtype)
            for lay in self.encode:
                final_embeds = lay(
                    tgt=final_embeds, memory=seq_first_img_feats,
                    tgt_key_padding_mask=final_mask,
                    pos=seq_first_img_pos_embeds,
                )
            final_embeds = self.encode_norm(final_embeds)
        if masks is not None and self.mask_encoder is not None:
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, masks_embed, masks_mask
            )
        return final_embeds, final_mask


# ---------------------------------------------------------------------------
# PromptEncoder (from sam/prompt_encoder.py)
# ---------------------------------------------------------------------------

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            operations.Embedding(1, embed_dim, dtype=dtype, device=device) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = operations.Embedding(1, embed_dim, dtype=dtype, device=device)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            operations.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2, dtype=dtype, device=device),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            operations.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2, dtype=dtype, device=device),
            LayerNorm2d(mask_in_chans),
            activation(),
            operations.Conv2d(mask_in_chans, embed_dim, kernel_size=1, dtype=dtype, device=device),
        )
        self.no_mask_embed = operations.Embedding(1, embed_dim, dtype=dtype, device=device)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device(),
            dtype=self.point_embeddings[0].weight.dtype,
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


# ---------------------------------------------------------------------------
# MaskDecoder (from sam/mask_decoder.py)
# ---------------------------------------------------------------------------

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = operations.Embedding(1, transformer_dim, dtype=dtype, device=device)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = operations.Embedding(self.num_mask_tokens, transformer_dim, dtype=dtype, device=device)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = operations.Embedding(1, transformer_dim, dtype=dtype, device=device)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            operations.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2,
                dtype=dtype, device=device,
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            operations.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2,
                dtype=dtype, device=device,
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = operations.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1,
                dtype=dtype, device=device,
            )
            self.conv_s1 = operations.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1,
                dtype=dtype, device=device,
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                SamMLP(transformer_dim, transformer_dim, transformer_dim // 8, 3,
                       dtype=dtype, device=device, operations=operations)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = SamMLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
            dtype=dtype, device=device, operations=operations,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = operations.Linear(transformer_dim, 1, dtype=dtype, device=device)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = SamMLP(
                    transformer_dim, transformer_dim, 1, 3,
                    dtype=dtype, device=device, operations=operations,
                )

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings.to(src.dtype)
        import os as _os
        if _os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes"):
            import logging as _logging
            _logging.getLogger("sam3").warning(
                "predict_masks: tokens=%s image_embed=%s dense_pe=%s src=%s",
                tokens.dtype, image_embeddings.dtype,
                dense_prompt_embeddings.dtype, src.dtype,
            )
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        if _os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes"):
            import logging as _logging
            _logging.getLogger("sam3").warning(
                "predict_masks: hs=%s mask_tokens=%s hyper_in=%s upscaled=%s",
                hs.dtype, mask_tokens_out.dtype, hyper_in.dtype, upscaled_embedding.dtype,
            )
        masks = (hyper_in @ upscaled_embedding.to(hyper_in.dtype).view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out


# ---------------------------------------------------------------------------
# Segmentation heads (from model/maskformer_segmentation.py)
# ---------------------------------------------------------------------------

class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model, dtype=None, device=None, operations=ops):
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(
            nn.Identity(),
            nn.Identity(),
            operations.Linear(d_model, 1, dtype=dtype, device=device),
        )

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim, mask_dim, dtype=None, device=None, operations=ops):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3,
                              dtype=dtype, device=device, operations=operations)

    def forward(self, obj_queries, pixel_embed):
        pixel_embed = pixel_embed.to(obj_queries.dtype)
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                mask_preds = torch.einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            if pixel_embed.ndim == 3:
                mask_preds = torch.einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
        return mask_preds


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(operations.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1,
                                                  dtype=dtype, device=device))
            norms.append(operations.GroupNorm(8, self.hidden_dim, dtype=dtype, device=device))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.hidden_dim

    def forward(self, backbone_feats: List[torch.Tensor]):
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(
                prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode
            )
            if self.shared_conv:
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))
        return prev_fpn


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
                dtype=dtype, device=device, operations=operations,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = operations.Conv2d(
                hidden_dim, 1, kernel_size=3, stride=1, padding=1,
                dtype=dtype, device=device,
            )
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim,
                                                 dtype=dtype, device=device, operations=operations)

        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)

    def _embed_pixels(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids,
        encoder_hidden_states,
    ) -> torch.Tensor:
        feature_device = backbone_feats[0].device
        model_device = self.device
        image_ids_ = image_ids.to(feature_device)
        if self.use_encoder_inputs:
            if backbone_feats[0].shape[0] > 1:
                backbone_visual_feats = []
                for feat in backbone_feats:
                    backbone_visual_feats.append(feat[image_ids_, ...].to(model_device))
            else:
                backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
                -1, *backbone_feats[-1].shape[1:]
            )
            backbone_visual_feats[-1] = encoder_visual_embed
            pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats = [x.to(model_device) for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
            dtype=dtype, device=device, operations=operations,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else LinearPresenceHead(self.d_model, dtype=dtype, device=device, operations=operations)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = operations.LayerNorm(self.d_model, dtype=dtype, device=device)

        self.semantic_seg_head = operations.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1,
                                                    dtype=dtype, device=device)
        self.instance_seg_head = operations.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1,
            dtype=dtype, device=device,
        )

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]
        _dtype_debug("SegHead.forward IN", backbone_feats=backbone_feats, obj_queries=obj_queries, encoder_hidden_states=encoder_hidden_states)

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        semantic_seg = self.semantic_seg_head(pixel_embed)
        _dtype_debug("SegHead.forward OUT", pred_masks=mask_pred, semantic_seg=semantic_seg, presence_logit=presence_logit)
        return {
            "pred_masks": mask_pred,
            "semantic_seg": semantic_seg,
            "presence_logit": presence_logit,
        }


# ---------------------------------------------------------------------------
# SAM3VLBackbone (from model/vl_combiner.py)
# ---------------------------------------------------------------------------

class SAM3VLBackbone(nn.Module):
    """Combines a vision backbone and a language backbone without fusion."""

    def __init__(
        self,
        visual,
        text,
        scalp=0,
        **kwargs,
    ):
        super().__init__()
        self.vision_backbone = visual
        self.language_backbone = text
        self.scalp = scalp

    def forward(
        self,
        samples: torch.Tensor,
        captions: List[str],
        input_boxes: Optional[torch.Tensor] = None,
        additional_text: Optional[List[str]] = None,
    ):
        output = self.forward_image(samples)
        device = output["vision_features"].device
        output.update(self.forward_text(captions, input_boxes, additional_text, device))
        return output

    def forward_image(self, samples: torch.Tensor):
        _dtype_debug("Backbone.forward_image IN", samples=samples)
        if samples.dtype == torch.uint8:
            samples = samples.float() / 255.0
        try:
            expected_device = next(self.vision_backbone.parameters()).device
            if samples.device != expected_device:
                samples = samples.to(expected_device)
        except StopIteration:
            pass
        result = self._forward_image_no_act_ckpt(samples)
        _dtype_debug("Backbone.forward_image OUT",
                     vision_features=result.get("vision_features"),
                     backbone_fpn=result.get("backbone_fpn"))
        return result

    def _forward_image_no_act_ckpt(self, samples):
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples
        )
        if self.scalp > 0:
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }
        return output

    def forward_text(
        self, captions, input_boxes=None, additional_text=None, device=None
    ):
        return self._forward_text_no_ack_ckpt(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
        )

    def _forward_text_no_ack_ckpt(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device=None,
    ):
        output = {}

        text_to_encode = copy(captions)
        if additional_text is not None:
            text_to_encode += additional_text

        text_attention_mask, text_memory, text_embeds = self.language_backbone(
            text_to_encode, input_boxes, device=device
        )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output


# ---------------------------------------------------------------------------
# Sam3Image (from model/sam3_image.py)
# ---------------------------------------------------------------------------

def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(torch.nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor=None,
        dtype=None,
        device=None,
        operations=ops,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring = dot_prod_scoring
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                self.instance_dot_prod_scoring = deepcopy(dot_prod_scoring)
        else:
            self.class_embed = operations.Linear(self.hidden_dim, 1, dtype=dtype, device=device)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = deepcopy(self.class_embed)

        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output

        self.inst_interactive_predictor = inst_interactive_predictor

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)

    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
            img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_pos_embeds = [
                x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
            ]
            return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

        img_batch = backbone_out["img_batch_all_stages"]
        if img_ids.numel() > 1:
            unique_ids, _ = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, _ = img_ids, slice(None)
        if isinstance(img_batch, torch.Tensor):
            image = img_batch[unique_ids]
        elif unique_ids.numel() == 1:
            image = img_batch[unique_ids.item()].unsqueeze(0)
        else:
            image = torch.stack([img_batch[i] for i in unique_ids.tolist()])
        image = image.to(dtype=torch.float32, device=self.device)
        id_mapping = torch.full(
            (len(img_batch),), -1, dtype=torch.long, device=self.device
        )
        id_mapping[unique_ids] = torch.arange(len(unique_ids), device=self.device)
        backbone_out = {
            **backbone_out,
            **self.backbone.forward_image(image),
            "id_mapping": id_mapping,
        }
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
    ):
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Cast text features to match visual features dtype (text encoder
        # outputs fp32 because Embedding receives integer indices; manual_cast
        # can't infer target dtype from integers).
        txt_feats = txt_feats.to(img_feats[-1].dtype)

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device, dtype=geo_feats.dtype
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1).to(memory.dtype)

        apply_dac = self.transformer.decoder.dac and self.training
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
            )
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)

        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs

    def _update_scores_and_boxes(
        self,
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        dec_presence_out=None,
        is_instance_prompt=False,
    ):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)
        out["queries"] = hs[-1][:, :num_o2o]
        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)

        box_head = self.transformer.decoder.bbox_embed
        if (
            is_instance_prompt
            and self.transformer.decoder.instance_bbox_embed is not None
        ):
            box_head = self.transformer.decoder.instance_bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        if dec_presence_out is not None:
            _update_out(
                out, "presence_logit_dec", dec_presence_out, update_aux=self.training
            )

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()
            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        _update_out(
            out, "pred_logits", outputs_class[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out, "pred_boxes", outputs_coord[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out,
            "pred_boxes_xyxy",
            outputs_boxes_xyxy[:, :, :num_o2o],
            update_aux=self.training,
        )
        if num_o2m > 0 and self.training:
            _update_out(
                out,
                "pred_logits_o2m",
                outputs_class[:, :, num_o2o:],
                update_aux=self.training,
            )
            _update_out(
                out,
                "pred_boxes_o2m",
                outputs_coord[:, :, num_o2o:],
                update_aux=self.training,
            )
            _update_out(
                out,
                "pred_boxes_xyxy_o2m",
                outputs_boxes_xyxy[:, :, num_o2o:],
                update_aux=self.training,
            )

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
    ):
        apply_dac = self.transformer.decoder.dac and self.training
        if self.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = self.segmentation_head(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if self.o2m_mask_predict and num_o2m > 0:
                        _update_out(
                            out, f"{k}_o2m", v[:, num_o2o:], auxiliary=aux_masks
                        )
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

    def _get_best_mask(self, out):
        prev_mask_idx = out["pred_logits"].argmax(dim=1).squeeze(1)
        batch_idx = torch.arange(
            out["pred_logits"].shape[0], device=prev_mask_idx.device
        )
        prev_mask_pred = out["pred_masks"][batch_idx, prev_mask_idx][:, None]
        prev_mask_pred = self.geometry_encoder.mask_encoder.mask_downsampler(
            prev_mask_pred
        )
        prev_mask_pred = prev_mask_pred.flatten(-2).permute(2, 0, 1)
        return prev_mask_pred

    def forward_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt: Prompt,
    ):
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        _dtype_debug("forward_grounding: after _encode_prompt", prompt=prompt, prompt_mask=prompt_mask)
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        _dtype_debug("forward_grounding: after _run_encoder",
                     encoder_hidden_states=encoder_out.get("encoder_hidden_states"),
                     pos_embed=encoder_out.get("pos_embed"))
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
        _dtype_debug("forward_grounding: after _run_decoder",
                     encoder_hidden_states=out.get("encoder_hidden_states"),
                     pred_boxes=out.get("pred_boxes"))

        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        _dtype_debug("forward_grounding: after _run_segmentation_heads",
                     pred_logits=out.get("pred_logits"),
                     pred_masks=out.get("pred_masks"),
                     pred_boxes=out.get("pred_boxes"))

        return out

    def _postprocess_out(self, out: Dict, multimask_output: bool = False):
        num_mask_boxes = out["pred_boxes"].size(1)
        if not self.training and multimask_output and num_mask_boxes > 1:
            out["multi_pred_logits"] = out["pred_logits"]
            if "pred_masks" in out:
                out["multi_pred_masks"] = out["pred_masks"]
            out["multi_pred_boxes"] = out["pred_boxes"]
            out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

            best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
            batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)

            out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(1)
            if "pred_masks" in out:
                out["pred_masks"] = out["pred_masks"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][
                batch_idx, best_mask_idx
            ].unsqueeze(1)

        return out

    def _get_dummy_prompt(self, num_prompts=1):
        device = self.device
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=device),
            box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
        )
        return geometric_prompt

    def forward(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        num_frames = len(input.find_inputs)
        assert num_frames == 1

        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )

        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        if find_input.input_points is not None and find_input.input_points.numel() > 0:
            log.warning("Point prompts are ignored in PCS.")

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )

        stage_outs = []
        for cur_step in range(num_interactive_steps + 1):
            if cur_step > 0:
                geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                    geo_prompt=geometric_prompt,
                    find_target=find_target,
                    previous_out=stage_outs[-1],
                )
            out = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt.clone(),
            )
            stage_outs.append(out)

        previous_stages_out.append(stage_outs)
        return previous_stages_out

    def predict_inst(
        self,
        inference_state,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h, orig_w = (
            inference_state["original_height"],
            inference_state["original_width"],
        )
        backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.inst_interactive_predictor.model._prepare_backbone_features(
            backbone_out
        )

        vision_feats[-1] = (
            vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
        )
        import os as _os
        if _os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes"):
            import logging as _logging
            _logging.getLogger("sam3").warning(
                "predict_inst: vision_feats[-1] dtype=%s, no_mem_embed dtype=%s",
                vision_feats[-1].dtype,
                self.inst_interactive_predictor.model.no_mem_embed.dtype,
            )
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self.inst_interactive_predictor._bb_feat_sizes[::-1]
            )
        ][::-1]

        self.inst_interactive_predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self.inst_interactive_predictor._is_image_set = True
        self.inst_interactive_predictor._orig_hw = [(orig_h, orig_w)]
        res = self.inst_interactive_predictor.predict(**kwargs)
        self.inst_interactive_predictor._features = None
        self.inst_interactive_predictor._is_image_set = False
        return res

    def predict_inst_batch(
        self,
        inference_state,
        *args,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.inst_interactive_predictor.model._prepare_backbone_features(
            backbone_out
        )
        vision_feats[-1] = (
            vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
        )
        batch_size = vision_feats[-1].shape[1]
        orig_heights, orig_widths = (
            inference_state["original_heights"],
            inference_state["original_widths"],
        )
        assert (
            batch_size == len(orig_heights) == len(orig_widths)
        ), f"Batch size mismatch in predict_inst_batch. Got {batch_size}, {len(orig_heights)}, {len(orig_widths)}"
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self.inst_interactive_predictor._bb_feat_sizes[::-1]
            )
        ][::-1]
        self.inst_interactive_predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self.inst_interactive_predictor._is_image_set = True
        self.inst_interactive_predictor._is_batch = True
        self.inst_interactive_predictor._orig_hw = [
            (orig_h, orig_w) for orig_h, orig_w in zip(orig_heights, orig_widths)
        ]
        res = self.inst_interactive_predictor.predict_batch(*args, **kwargs)
        self.inst_interactive_predictor._features = None
        self.inst_interactive_predictor._is_image_set = False
        self.inst_interactive_predictor._is_batch = False
        return res


import os

class Sam3ImageOnVideoMultiGPU(Sam3Image):
    def __init__(
        self, *args, async_all_gather=True, gather_backbone_out=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.async_all_gather = async_all_gather

        if gather_backbone_out is None:
            gather_backbone_out = isinstance(self.backbone, SAM3VLBackbone)
        self.gather_backbone_out = gather_backbone_out

    def forward_video_grounding_multigpu(
        self,
        backbone_out,
        find_inputs,
        geometric_prompt: Prompt,
        frame_idx,
        num_frames,
        multigpu_buffer,
        track_in_reverse=False,
        return_sam2_backbone_feats=False,
        return_tracker_backbone_feats=False,
        run_nms=False,
        nms_prob_thresh=None,
        nms_iou_thresh=None,
        **kwargs,
    ):
        from .perflib import nms_masks

        frame_idx_curr_b = frame_idx - frame_idx % self.world_size
        frame_idx_curr_e = min(frame_idx_curr_b + self.world_size, num_frames)
        if frame_idx not in multigpu_buffer:
            with torch.profiler.record_function("build_multigpu_buffer_next_chunk1"):
                self._build_multigpu_buffer_next_chunk(
                    backbone_out=backbone_out,
                    find_inputs=find_inputs,
                    geometric_prompt=geometric_prompt,
                    frame_idx_begin=frame_idx_curr_b,
                    frame_idx_end=frame_idx_curr_e,
                    num_frames=num_frames,
                    multigpu_buffer=multigpu_buffer,
                    run_nms=run_nms,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                )

        out = {}
        for k, (v, handle) in multigpu_buffer[frame_idx].items():
            if k.startswith("sam2_backbone_") and not return_sam2_backbone_feats:
                continue
            if k.startswith("tracker_backbone_") and not return_tracker_backbone_feats:
                continue
            if handle is not None:
                handle.wait()
            out[k] = v

        if not track_in_reverse and frame_idx_curr_b - self.world_size >= 0:
            frame_idx_prev_e = frame_idx_curr_b
            frame_idx_prev_b = frame_idx_curr_b - self.world_size
        elif track_in_reverse and frame_idx_curr_e < num_frames:
            frame_idx_prev_b = frame_idx_curr_e
            frame_idx_prev_e = min(frame_idx_prev_b + self.world_size, num_frames)
        else:
            frame_idx_prev_b = frame_idx_prev_e = None
        if frame_idx_prev_b is not None:
            for frame_idx_rm in range(frame_idx_prev_b, frame_idx_prev_e):
                multigpu_buffer.pop(frame_idx_rm, None)

        if not track_in_reverse and frame_idx_curr_e < num_frames:
            frame_idx_next_b = frame_idx_curr_e
            frame_idx_next_e = min(frame_idx_next_b + self.world_size, num_frames)
        elif track_in_reverse and frame_idx_curr_b - self.world_size >= 0:
            frame_idx_next_e = frame_idx_curr_b
            frame_idx_next_b = frame_idx_curr_b - self.world_size
        else:
            frame_idx_next_b = frame_idx_next_e = None
        if frame_idx_next_b is not None and frame_idx_next_b not in multigpu_buffer:
            with torch.profiler.record_function("build_multigpu_buffer_next_chunk2"):
                self._build_multigpu_buffer_next_chunk(
                    backbone_out=backbone_out,
                    find_inputs=find_inputs,
                    geometric_prompt=geometric_prompt,
                    frame_idx_begin=frame_idx_next_b,
                    frame_idx_end=frame_idx_next_e,
                    num_frames=num_frames,
                    multigpu_buffer=multigpu_buffer,
                    run_nms=run_nms,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                )

        return out, backbone_out

    def _build_multigpu_buffer_next_chunk(
        self,
        backbone_out,
        find_inputs,
        geometric_prompt: Prompt,
        frame_idx_begin,
        frame_idx_end,
        num_frames,
        multigpu_buffer,
        run_nms=False,
        nms_prob_thresh=None,
        nms_iou_thresh=None,
    ):
        from .perflib import nms_masks

        frame_idx_local_gpu = min(frame_idx_begin + self.rank, frame_idx_end - 1)
        with torch.profiler.record_function("forward_grounding"):
            out_local = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_inputs[frame_idx_local_gpu],
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
        if run_nms:
            with torch.profiler.record_function("nms_masks"):
                assert nms_prob_thresh is not None and nms_iou_thresh is not None
                pred_probs = out_local["pred_logits"].squeeze(-1).sigmoid()
                pred_masks = out_local["pred_masks"]
                for prompt_idx in range(pred_probs.size(0)):
                    keep = nms_masks(
                        pred_probs=pred_probs[prompt_idx],
                        pred_masks=pred_masks[prompt_idx],
                        prob_threshold=nms_prob_thresh,
                        iou_threshold=nms_iou_thresh,
                    )
                    out_local["pred_logits"][prompt_idx, :, 0] -= 1e4 * (~keep).float()

        if self.gather_backbone_out:
            feats = out_local["prev_encoder_out"]["backbone_out"]["sam2_backbone_out"]
            assert len(feats["backbone_fpn"]) == 3
            if feats["backbone_fpn"][0].device.type == "cuda":
                backbone_fpn_bf16 = [x.to(torch.bfloat16) for x in feats["backbone_fpn"]]
            else:
                backbone_fpn_bf16 = list(feats["backbone_fpn"])
            fpn0, fpn_handle0 = self._gather_tensor(backbone_fpn_bf16[0])
            fpn1, fpn_handle1 = self._gather_tensor(backbone_fpn_bf16[1])
            fpn2, fpn_handle2 = self._gather_tensor(backbone_fpn_bf16[2])
            vision_pos_enc = feats["vision_pos_enc"]

        out_local = {
            "pred_logits": out_local["pred_logits"],
            "pred_boxes": out_local["pred_boxes"],
            "pred_boxes_xyxy": out_local["pred_boxes_xyxy"],
            "pred_masks": out_local["pred_masks"],
        }

        out_gathered = {k: self._gather_tensor(v) for k, v in out_local.items()}
        for rank in range(self.world_size):
            frame_idx_to_save = frame_idx_begin + rank
            if frame_idx_to_save >= num_frames:
                continue
            frame_buffer = {
                k: (v[rank], handle) for k, (v, handle) in out_gathered.items()
            }
            if self.gather_backbone_out:
                frame_buffer["tracker_backbone_fpn_0"] = (fpn0[rank], fpn_handle0)
                frame_buffer["tracker_backbone_fpn_1"] = (fpn1[rank], fpn_handle1)
                frame_buffer["tracker_backbone_fpn_2"] = (fpn2[rank], fpn_handle2)
                frame_buffer["tracker_backbone_pos_enc"] = (vision_pos_enc, None)
            multigpu_buffer[frame_idx_to_save] = frame_buffer

    def _gather_tensor(self, x):
        if self.world_size == 1:
            return [x], None
        async_op = self.async_all_gather
        x = x.contiguous()
        output_list = [torch.empty_like(x) for _ in range(self.world_size)]
        handle = torch.distributed.all_gather(output_list, x, async_op=async_op)
        return output_list, handle


# ---------------------------------------------------------------------------
# Section 6: SAM3 Tracker
# ---------------------------------------------------------------------------

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class Sam3TrackerBase(torch.nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        maskmem_backbone,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        max_cond_frames_in_attn=-1,
        keep_first_cond_frame=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        forward_backbone_per_frame_for_eval=False,
        memory_temporal_stride_for_eval=1,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        non_overlap_masks_for_mem_enc=False,
        max_obj_ptrs_in_encoder=16,
        sam_mask_decoder_extra_args=None,
        compile_all_components=False,
        use_memory_selection=False,
        mf_threshold=0.01,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        # Store for _build_sam_heads
        self._dtype = dtype
        self._device_arg = device
        self._operations = operations

        # Part 1: the image backbone
        self.backbone = backbone
        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        self.mask_downsample = operations.Conv2d(1, 1, kernel_size=4, stride=4, dtype=dtype, device=device)

        # Part 2: encoder-only transformer
        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        # Part 3: memory encoder
        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            self.mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem

        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking

        # Part 4: SAM-style prompt encoder and mask decoder
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        trunc_normal_(self.no_obj_ptr, std=0.02)
        self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame

        self.use_memory_selection = use_memory_selection
        self.mf_threshold = mf_threshold

        self.compile_all_components = compile_all_components
        if self.compile_all_components:
            self._compile_all_components()

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None, dummy=False):
        if dummy:
            return torch.zeros(len(rel_pos_list), self.mem_dim, device=device)

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        if device is not None and device.type == "cuda":
            pos_enc = (
                torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)
                / t_diff_max
            )
        else:
            pos_enc = (
                torch.tensor(rel_pos_list).to(device=device)
                / t_diff_max
            )
        tpos_dim = self.hidden_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        pos_enc = self.obj_ptr_tpos_proj(pos_enc)

        return pos_enc

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        _dtype = self._dtype
        _device = self._device_arg
        _ops = self._operations

        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
            dtype=_dtype,
            device=_device,
            operations=_ops,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                dtype=_dtype,
                device=_device,
                operations=_ops,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dtype=_dtype,
            device=_device,
            operations=_ops,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        self.obj_ptr_proj = SamMLP(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, 3,
            dtype=_dtype, device=_device, operations=_ops,
        )
        self.obj_ptr_tpos_proj = _ops.Linear(
            self.hidden_dim, self.mem_dim, dtype=_dtype, device=_device,
        )

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        gt_masks=None,
    ):
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        sparse_embeddings = self._maybe_clone(sparse_embeddings)
        dense_embeddings = self._maybe_clone(dense_embeddings)
        image_pe = self._maybe_clone(self.sam_prompt_encoder.get_dense_pe())
        with torch.profiler.record_function("sam_mask_decoder"):
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,
                high_res_features=high_res_features,
            )
        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)
        object_score_logits = self._maybe_clone(object_score_logits)

        if self.training and self.teacher_force_obj_scores_for_mem:
            is_obj_appearing = torch.any(gt_masks.float().flatten(1) > 0, dim=1)
            is_obj_appearing = is_obj_appearing[..., None]
        else:
            is_obj_appearing = object_score_logits > 0

        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()

        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(
                high_res_masks.size(-2) // self.backbone_stride * 4,
                high_res_masks.size(-1) // self.backbone_stride * 4,
            ),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward(self, input: BatchedDatapoint, is_inference=False):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM3VideoPredictor for inference. "
            "See examples/sam3_dense_video_tracking.ipynb for an inference example."
        )

    def forward_image(self, img_batch):
        backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = self._maybe_clone(
                backbone_out["backbone_fpn"][i]
            )
            backbone_out["vision_pos_enc"][i] = self._maybe_clone(
                backbone_out["vision_pos_enc"][i]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def cal_mem_score(self, object_score_logits, iou_score):
        object_score_norm = torch.where(
            object_score_logits > 0,
            object_score_logits.sigmoid() * 2 - 1,
            torch.zeros_like(object_score_logits),
        )
        score_per_frame = (object_score_norm * iou_score).mean()
        return score_per_frame

    def frame_filter(self, output_dict, track_in_reverse, frame_idx, num_frames, r):
        if (frame_idx == 0 and not track_in_reverse) or (
            frame_idx == num_frames - 1 and track_in_reverse
        ):
            return []

        max_num = min(num_frames, self.max_obj_ptrs_in_encoder)

        if not track_in_reverse:
            start = frame_idx - 1
            end = 0
            step = -r
            must_include = frame_idx - 1
        else:
            start = frame_idx + 1
            end = num_frames
            step = r
            must_include = frame_idx + 1

        valid_indices = []
        for i in range(start, end, step):
            if (
                i not in output_dict["non_cond_frame_outputs"]
                or "eff_iou_score" not in output_dict["non_cond_frame_outputs"][i]
            ):
                continue

            score_per_frame = output_dict["non_cond_frame_outputs"][i]["eff_iou_score"]

            if score_per_frame > self.mf_threshold:
                valid_indices.insert(0, i)

            if len(valid_indices) >= max_num - 1:
                break

        if must_include not in valid_indices:
            valid_indices.append(must_include)

        return valid_indices

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if not is_init_cond_frame and use_prev_mem_frame:
            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )
            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]
            r = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )

            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if self.use_memory_selection:
                    if t_rel > len(valid_indices):
                        continue
                    prev_frame_idx = valid_indices[-t_rel]
                else:
                    if t_rel == 1:
                        if not track_in_reverse:
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        if not track_in_reverse:
                            prev_frame_idx = ((frame_idx - 2) // r) * r
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                        else:
                            prev_frame_idx = -(-(frame_idx + 2) // r) * r
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * r

                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=torch.cuda.is_available())
                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_prompt_mask.append(
                    torch.zeros(B, seq_len, device=device, dtype=bool)
                )
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):
                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding

                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                )
                to_cat_prompt_pos_embed.append(maskmem_enc)

            if (
                self.training
                and self.prob_to_dropout_spatial_mem > 0
                and self.rng.random() < self.prob_to_dropout_spatial_mem
            ):
                num_spatial_mem_keep = self.rng.integers(len(to_cat_prompt) + 1)
                keep = self.rng.choice(
                    range(len(to_cat_prompt)), num_spatial_mem_keep, replace=False
                ).tolist()
                to_cat_prompt = [to_cat_prompt[i] for i in keep]
                to_cat_prompt_mask = [to_cat_prompt_mask[i] for i in keep]
                to_cat_prompt_pos_embed = [to_cat_prompt_pos_embed[i] for i in keep]

            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            if not self.training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_ptrs = [
                (
                    (frame_idx - t) * tpos_sign_mul,
                    out["obj_ptr"],
                    True,
                )
                for t, out in ptr_cond_outputs.items()
            ]

            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if not self.use_memory_selection:
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                else:
                    if -t_diff <= -len(valid_indices):
                        break
                    t = valid_indices[-t_diff]

                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))

            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = (
                        obj_ptrs
                        + self.cond_frame_obj_ptr_embedding
                        * torch.tensor(is_selected_cond_frame_list, device=device)[
                            ..., None, None
                        ].float()
                    )
                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                    device=device,
                )
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

            to_cat_prompt = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_prompt_mask = [torch.zeros(B, 1, device=device, dtype=bool)]
            to_cat_prompt_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        output_dict=None,
        is_init_cond_frame=False,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        if is_mask_from_pts and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            pix_feat = pix_feat.view_as(pix_feat)
            maskmem_out = self.maskmem_backbone(
                pix_feat, mask_for_mem, skip_mask_sigmoid=True
            )
        else:
            maskmem_out = self.maskmem_backbone(image, pix_feat, mask_for_mem)
        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]
        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features += (
            1 - is_obj_appearing[..., None, None]
        ) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc

    def forward_tracking(self, backbone_out, input, return_dict=False):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        for stage_id in processing_order:
            img_ids = input.find_inputs[stage_id].img_ids
            if img_feats_already_computed:
                current_image = input.img_batch[img_ids]
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                (
                    current_image,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(input.img_batch, img_ids)
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        image,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )
            if prev_sam_mask_logits is not None:
                assert self.iter_use_prev_mask_pred
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        (
            _,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if self.use_memory_selection:
            current_out["object_score_logits"] = object_score_logits
            iou_score = ious.max(-1)[0]
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(
                object_score_logits, iou_score
            )
        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if self.offload_output_to_cpu_for_eval and not self.training:
            trimmed_out = {
                "pred_masks": current_out["pred_masks"].cpu(),
                "pred_masks_high_res": current_out["pred_masks_high_res"].cpu(),
                "obj_ptr": current_out["obj_ptr"],
                "object_score_logits": current_out["object_score_logits"],
            }
            if run_mem_encoder and self.num_maskmem > 0:
                trimmed_out["maskmem_features"] = maskmem_features.cpu()
                trimmed_out["maskmem_pos_enc"] = [x.cpu() for x in maskmem_pos_enc]
            if self.use_memory_selection:
                trimmed_out["iou_score"] = current_out["iou_score"].cpu()
                trimmed_out["eff_iou_score"] = current_out["eff_iou_score"].cpu()
            current_out = trimmed_out

        def _trim_past_out(past_out, current_out):
            if past_out is None:
                return None
            return {
                "pred_masks": past_out["pred_masks"],
                "obj_ptr": past_out["obj_ptr"],
                "object_score_logits": past_out["object_score_logits"],
            }

        if self.trim_past_non_cond_mem_for_eval and not self.training:
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx, None)

            if past_out is not None:
                log.info(past_out.get("eff_iou_score", 0))
                if (
                    self.use_memory_selection
                    and past_out.get("eff_iou_score", 0) < self.mf_threshold
                ) or not self.use_memory_selection:
                    output_dict["non_cond_frame_outputs"][past_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

            if (
                self.use_memory_selection and not self.offload_output_to_cpu_for_eval
            ):
                far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
                past_out = output_dict["non_cond_frame_outputs"].get(
                    far_old_frame_idx, None
                )
                if past_out is not None:
                    output_dict["non_cond_frame_outputs"][far_old_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _compile_all_components(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        from .perflib import compile_wrapper

        logging.info("Compiling all components. First time may be very slow.")

        self.maskmem_backbone.forward = compile_wrapper(
            self.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self.transformer.encoder.forward = compile_wrapper(
            self.transformer.encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,
        )
        self.sam_mask_decoder.forward = compile_wrapper(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    def _maybe_clone(self, x):
        return x.clone() if self.compile_all_components else x


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


# ---------------------------------------------------------------------------
# Section 7: Sam3TrackerPredictor
# (from model/sam3_tracking_predictor.py)
# ---------------------------------------------------------------------------

class Sam3TrackerPredictor(Sam3TrackerBase):
    """
    The demo class that extends the `Sam3TrackerBase` to handle user interactions
    and manage inference states, with support for multi-object tracking.
    """

    def __init__(
        self,
        clear_non_cond_mem_around_input=False,
        clear_non_cond_mem_for_multi_obj=False,
        fill_hole_area=0,
        always_start_from_first_ann_frame=False,
        max_point_num_in_prompt_enc=16,
        non_overlap_masks_for_output=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

        self.iter_use_prev_mask_pred = True
        self.add_all_frames_to_correct_as_cond = True

    @torch.inference_mode()
    def init_state(
        self,
        video_height=None,
        video_width=None,
        num_frames=None,
        video_path=None,
        cached_features=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize a inference state."""
        inference_state = {}
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["device"] = self.device
        if offload_state_to_cpu or self.device.type != "cuda":
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = self.device

        if video_path is not None:
            images, video_height, video_width = load_video_frames(
                video_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=inference_state["storage_device"],
            )
            inference_state["images"] = images
            inference_state["num_frames"] = len(images)
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
        else:
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
            inference_state["num_frames"] = num_frames
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = (
            {} if cached_features is None else cached_features
        )
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["first_ann_frame_idx"] = None
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        self.clear_all_points_in_video(inference_state)
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        rel_coordinates=True,
        use_prev_mem_frame=False,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if rel_coordinates:
            if points is not None:
                points = points * self.image_size
            if box is not None:
                box = box * self.image_size

        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        num_points = point_inputs["point_coords"].size(1)
        if num_points > self.max_point_num_in_prompt_enc > 0:
            num_first = self.max_point_num_in_prompt_enc // 2
            num_last = self.max_point_num_in_prompt_enc - num_first
            point_inputs["point_coords"] = torch.cat(
                [
                    point_inputs["point_coords"][:, :num_first],
                    point_inputs["point_coords"][:, -num_last:],
                ],
                dim=1,
            )
            point_inputs["point_labels"] = torch.cat(
                [
                    point_inputs["point_labels"][:, :num_first],
                    point_inputs["point_labels"][:, -num_last:],
                ],
                dim=1,
            )
            logging.warning(
                f"Too many points ({num_points}) are provided on frame {frame_idx}. Only "
                f"the first {num_first} points and the last {num_last} points will be used."
            )
        prev_sam_mask_logits = None
        if self.iter_use_prev_mask_pred:
            prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
                if prev_out is None:
                    prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

            if prev_out is not None and prev_out["pred_masks"] is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(inference_state["device"], non_blocking=torch.cuda.is_available())
                prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=use_prev_mem_frame,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        add_mask_to_memory=False,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        if mask_H != self.input_mask_size or mask_W != self.input_mask_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.input_mask_size, self.input_mask_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        else:
            mask_inputs = mask_inputs_orig

        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        if mask_H != video_H or mask_W != video_W:
            mask_inputs_video_res = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(video_H, video_W),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        else:
            mask_inputs_video_res = mask_inputs_orig
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        mask_inputs_per_frame[frame_idx] = mask_inputs_video_res
        point_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        current_out["pred_masks"] = None
        current_out["pred_masks_video_res"] = torch.where(
            mask_inputs_video_res, -NO_OBJ_SCORE, NO_OBJ_SCORE
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        for obj_idx2, obj_temp_output_dict2 in temp_output_dict_per_obj.items():
            if obj_idx2 == obj_idx:
                continue
            current_out2 = obj_temp_output_dict2[storage_key].get(frame_idx, None)
            if current_out2 is not None and "pred_masks_video_res" in current_out2:
                current_out2["pred_masks_video_res"] = torch.where(
                    mask_inputs_video_res,
                    NO_OBJ_SCORE,
                    current_out2["pred_masks_video_res"],
                )

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=torch.cuda.is_available())
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        if self.fill_hole_area > 0 and len(video_res_masks) > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks, self.fill_hole_area
            )
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.low_res_mask_size
            consolidated_mask_key = "pred_masks"

        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        if self.use_memory_selection:
            consolidated_out["iou_score"] = torch.full(
                size=(batch_size, 1),
                fill_value=0.0,
                dtype=torch.float32,
                device=inference_state["device"],
            )
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            obj_mask = out.get("pred_masks_video_res", out["pred_masks"])
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                is_downsampling = "pred_masks_video_res" in out
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias=is_downsampling,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]
            if self.use_memory_selection:
                consolidated_out["iou_score"][obj_idx : obj_idx + 1] = out["iou_score"]
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=torch.cuda.is_available()),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"],
        )

        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=None,
            mask_inputs=mask_inputs,
            gt_masks=None,
            frames_to_add_correction_pt=[],
            output_dict={
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=run_mem_encoder,
                )
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = min(
                input_frames_inds, default=None
            )
        if (
            inference_state["first_ann_frame_idx"]
            not in output_dict["cond_frame_outputs"]
        ):
            inference_state["first_ann_frame_idx"] = min(
                output_dict["cond_frame_outputs"], default=None
            )

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        num_frames = inference_state["num_frames"]
        if self.always_start_from_first_ann_frame:
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = [0]
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        tqdm_disable=False,
        obj_ids=None,
        run_mem_encoder=True,
        propagate_preflight=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        if propagate_preflight:
            self.propagate_in_video_preflight(inference_state)
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if obj_ids is not None:
            raise NotImplementedError(
                "Per-object tracking yet for batched inference if not implemented."
            )
        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        processing_order = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        )

        for frame_idx in tqdm(
            processing_order, desc="propagate in video", disable=tqdm_disable
        ):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                obj_scores = current_out["object_score_logits"]
                output_dict[storage_key][frame_idx] = current_out
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if self.use_memory_selection:
                obj_out["iou_score"] = current_out["iou_score"][obj_slice]
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_points_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    @torch.inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            if self.backbone is None:
                raise RuntimeError(
                    f"Image features for frame {frame_idx} are not cached. "
                    "Please run inference on this frame first."
                )
            else:
                image = inference_state["images"][frame_idx].to(inference_state["device"]).float().unsqueeze(0)
                backbone_out = self.forward_image(image)
                inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        if "tracker_backbone_out" in backbone_out:
            backbone_out = backbone_out["tracker_backbone_out"]

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            feat = feat.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["backbone_fpn"][i] = feat
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=use_prev_mem_frame,
        )

        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            if maskmem_features.device.type == "cuda":
                maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=torch.cuda.is_available())
        pred_masks_gpu = current_out["pred_masks"]
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=torch.cuda.is_available())
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        if self.use_memory_selection:
            compact_current_out["iou_score"] = current_out["iou_score"]
            compact_current_out["eff_iou_score"] = current_out["eff_iou_score"]
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        image, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            image=image,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        storage_device = inference_state["storage_device"]
        if maskmem_features.device.type == "cuda":
            maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=torch.cuda.is_available())
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"], updated_frames

        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                if self.use_memory_selection:
                    out["iou_score"] = out["iou_score"][remain_old_obj_inds]
                    out["eff_iou_score"] = self.cal_mem_score(
                        out["object_score_logits"], out["iou_score"]
                    )
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)

    def _suppress_shrinked_masks(
        self, pred_masks, new_pred_masks, shrink_threshold=0.3
    ):
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(
            keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0)
        )
        return pred_masks_after

    def _suppress_object_pw_area_shrinkage(self, pred_masks):
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks
        )
        pred_masks = self._suppress_shrinked_masks(
            pred_masks, pixel_level_non_overlapping_masks
        )
        return pred_masks

    def _apply_object_wise_non_overlapping_constraints(
        self, pred_masks, obj_scores, background_value=-10.0
    ):
        pred_masks_single_score = torch.where(
            pred_masks > 0, obj_scores[..., None, None], background_value
        )
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks


# ---------------------------------------------------------------------------
# Section 8: MaskletConfirmationStatus + Sam3VideoBase
# (from model/sam3_video_base.py)
# ---------------------------------------------------------------------------

class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1
    CONFIRMED = 2


class Sam3VideoBase(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        tracker: nn.Module,
        score_threshold_detection=0.5,
        det_nms_thresh=0.0,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.0,
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=0,
        max_trk_keep_alive=8,
        min_trk_keep_alive=-4,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=False,
        o2o_matching_masklets_enable=False,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        masklet_confirmation_enable=False,
        masklet_confirmation_consecutive_det_thresh=3,
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None

        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000
            num_obj_for_compile = 16
        logger.info(f"setting {max_num_objects=} and {num_obj_for_compile=}")
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def _init_dist_pg_cpu(self):
        timeout_sec = int(os.getenv("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180"))
        timeout = datetime.timedelta(seconds=timeout_sec)
        self._dist_pg_cpu = dist.new_group(backend="gloo", timeout=timeout)

    def broadcast_python_obj_cpu(self, python_obj_list, src):
        if self._dist_pg_cpu is None:
            self._init_dist_pg_cpu()
        dist.broadcast_object_list(python_obj_list, src=src, group=self._dist_pg_cpu)

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
        allow_new_detections: bool = True,
    ):
        det_out = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            allow_new_detections=allow_new_detections,
        )

        if tracker_metadata_prev == {}:
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = (
            self.run_tracker_propagation(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                tracker_states_local=tracker_states_local,
                tracker_metadata_prev=tracker_metadata_prev,
            )
        )

        tracker_update_plan, tracker_metadata_new = (
            self.run_tracker_update_planning_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_states_local=tracker_states_local,
                is_image_only=is_image_only,
            )
        )

        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())
        det_to_matched_trk_obj_ids = tracker_update_plan.get(
            "det_to_matched_trk_obj_ids", {}
        )

        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
        )

        if self.rank == 0:
            obj_id_to_mask = self.build_outputs(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_update_plan=tracker_update_plan,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                reconditioned_obj_ids=reconditioned_obj_ids,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            )
            obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        else:
            obj_id_to_mask, obj_id_to_score = {}, {}
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj_per_gpu"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        if tracker_obj_scores_global.shape[0] > 0:
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(dict(zip(tracker_obj_ids, tracker_obj_scores_global)))
        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    def _suppress_detections_close_to_boundary(self, boxes, margin=0.025):
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )
        return keep

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        num_frames: int,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        feature_cache: Dict,
        reverse: bool,
        allow_new_detections: bool,
    ):
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.detector.backbone.forward_text(
                input_batch.find_text_batch, device=self.device
            )
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        if "multigpu_buffer" not in feature_cache:
            feature_cache["multigpu_buffer"] = {}

        tracking_bounds = feature_cache.get("tracking_bounds", {})
        max_frame_num_to_track = tracking_bounds.get("max_frame_num_to_track")
        start_frame_idx = tracking_bounds.get("propagate_in_video_start_frame_idx")

        sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
            backbone_out={
                "img_batch_all_stages": input_batch.img_batch,
                **text_outputs,
            },
            find_inputs=input_batch.find_inputs,
            geometric_prompt=geometric_prompt,
            frame_idx=frame_idx,
            num_frames=num_frames,
            multigpu_buffer=feature_cache["multigpu_buffer"],
            track_in_reverse=reverse,
            return_tracker_backbone_feats=True,
            run_nms=self.det_nms_thresh > 0.0,
            nms_prob_thresh=self.score_threshold_detection,
            nms_iou_thresh=self.det_nms_thresh,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=start_frame_idx,
        )
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 1e8
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]

        pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)

        det_out = {
            "bbox": pred_boxes_xyxy[pos_pred_idx[0], pos_pred_idx[1]],
            "mask": pred_masks[pos_pred_idx[0], pos_pred_idx[1]],
            "scores": pred_probs[pos_pred_idx[0], pos_pred_idx[1]],
        }

        backbone_cache = {}
        sam_mask_decoder = self.tracker.sam_mask_decoder
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(sam3_image_out["tracker_backbone_fpn_0"]),
            sam_mask_decoder.conv_s1(sam3_image_out["tracker_backbone_fpn_1"]),
            sam3_image_out["tracker_backbone_fpn_2"],
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": sam3_image_out["tracker_backbone_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        backbone_cache["tracker_backbone_out"] = tracker_backbone_out
        feature_cache[frame_idx] = (
            input_batch.img_batch[frame_idx],
            backbone_cache,
        )
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out

    def run_tracker_propagation(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, npt.NDArray],
    ):
        obj_ids_local, low_res_masks_local, obj_scores_local = (
            self._propogate_tracker_one_frame_local_gpu(
                tracker_states_local, frame_idx=frame_idx, reverse=reverse
            )
        )

        assert np.all(
            obj_ids_local == tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        ), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        )

        _, H_mask, W_mask = low_res_masks_local.shape
        if self.world_size > 1:
            low_res_masks_local = low_res_masks_local.float().contiguous()
            obj_scores_local = obj_scores_local.float().contiguous()
            num_obj_this_gpu = tracker_metadata_prev["num_obj_per_gpu"][self.rank]
            assert low_res_masks_local.size(0) == num_obj_this_gpu
            assert obj_scores_local.size(0) == num_obj_this_gpu
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            obj_scores_peers = [
                obj_scores_local.new_empty(num_obj)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            dist.all_gather(obj_scores_peers, obj_scores_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
            obj_scores_global = torch.cat(obj_scores_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
            obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: Dict[str, Tensor],
        trk_id_to_max_iou_high_conf_det: List[int],
        tracker_states_local: List[Any],
        tracker_metadata: Dict[str, npt.NDArray],
        tracker_obj_scores_global: Tensor,
    ):
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            input_mask_res = self.tracker.input_mask_size
            new_mask_binary = (
                F.interpolate(
                    new_mask.unsqueeze(1),
                    size=(input_mask_res, input_mask_res),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)[0]
                > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids_all_gpu"] == trk_obj_id)[
                0
            ].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    and obj_score > HIGH_CONF_THRESH
                ):
                    logger.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        mask=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)

            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(
                    tracker_states_local[idx], run_mem_encoder=True
                )
        return tracker_states_local

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_states_local: List[Any],
        is_image_only: bool = False,
    ):
        tracker_metadata_new = {
            "obj_ids_per_gpu": deepcopy(tracker_metadata_prev["obj_ids_per_gpu"]),
            "obj_ids_all_gpu": None,
            "num_obj_per_gpu": deepcopy(tracker_metadata_prev["num_obj_per_gpu"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(
                tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]
            ),
            "obj_id_to_last_occluded": {},
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        reconditioned_obj_ids = set()

        det_mask_preds: Tensor = det_out["mask"]
        det_scores_np: npt.NDArray = det_out["scores"].float().cpu().numpy()
        det_bbox_xyxy: Tensor = det_out["bbox"]
        if self.rank == 0:
            (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            ) = self._associate_det_trk(
                det_masks=det_mask_preds,
                det_scores_np=det_scores_np,
                trk_masks=tracker_low_res_masks_global,
                trk_obj_ids=tracker_metadata_prev["obj_ids_all_gpu"],
            )

            if self.suppress_det_close_to_boundary:
                keep = self._suppress_detections_close_to_boundary(
                    det_bbox_xyxy[new_det_fa_inds]
                )
                new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

            prev_obj_num = np.sum(tracker_metadata_prev["num_obj_per_gpu"])
            new_det_num = len(new_det_fa_inds)
            num_obj_dropped_due_to_limit = 0
            if not is_image_only and prev_obj_num + new_det_num > self.max_num_objects:
                logger.warning(
                    f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}"
                )
                new_det_num_to_keep = self.max_num_objects - prev_obj_num
                num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                new_det_fa_inds = self._drop_new_det_with_obj_limit(
                    new_det_fa_inds, det_scores_np, new_det_num_to_keep
                )
                assert len(new_det_fa_inds) == new_det_num_to_keep
                new_det_num = len(new_det_fa_inds)

            new_det_start_obj_id = tracker_metadata_prev["max_obj_id"] + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            prev_workload_per_gpu = tracker_metadata_prev["num_obj_per_gpu"]
            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=new_det_num,
                prev_workload_per_gpu=prev_workload_per_gpu,
            )

            rank0_metadata_new = deepcopy(tracker_metadata_prev["rank0_metadata"])
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                obj_ids_newly_removed, rank0_metadata_new = self._process_hotstart(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                    new_det_obj_ids=new_det_obj_ids,
                    empty_trk_obj_ids=empty_trk_obj_ids,
                    unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                    rank0_metadata=rank0_metadata_new,
                    tracker_metadata=tracker_metadata_prev,
                )
            else:
                obj_ids_newly_removed = set()
            tracker_metadata_new["rank0_metadata"] = rank0_metadata_new

        NUM_BROADCAST_ITEMS = 9
        if self.rank == 0 and self.world_size > 1:
            num_obj_per_gpu_on_rank0 = tracker_metadata_prev["num_obj_per_gpu"]
            update_plan = [
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ]
            assert (
                len(update_plan) == NUM_BROADCAST_ITEMS
            ), f"Manually update NUM_BROADCAST_ITEMS to be: {len(update_plan)}"
            self.broadcast_python_obj_cpu(update_plan, src=0)
        elif self.rank > 0 and self.world_size > 1:
            update_plan = [None] * NUM_BROADCAST_ITEMS
            self.broadcast_python_obj_cpu(update_plan, src=0)
            (
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ) = update_plan
            if not np.all(
                num_obj_per_gpu_on_rank0 == tracker_metadata_prev["num_obj_per_gpu"]
            ):
                raise RuntimeError(
                    f"{self.rank=} received {num_obj_per_gpu_on_rank0=}, which is inconsistent with local record "
                    f"{tracker_metadata_prev['num_obj_per_gpu']=}. There's likely a bug in update planning or execution."
                )

        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,
            "new_det_obj_ids": new_det_obj_ids,
            "new_det_gpu_ids": new_det_gpu_ids,
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,
            "obj_ids_newly_removed": obj_ids_newly_removed,
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,
            "reconditioned_obj_ids": reconditioned_obj_ids,
        }

        should_recondition_iou = False

        if (
            self.reconstruction_bbox_iou_thresh > 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        ):
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]

                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids_all_gpu"]).index(
                        trk_obj_id
                    )
                except ValueError:
                    continue

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue

                tracker_box_pixels = (
                    mask_to_box(mask_binary.unsqueeze(0).unsqueeze(0))
                    .squeeze(0)
                    .squeeze(0)
                )
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )

                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = fast_diag_box_iou(det_box_batch, tracker_box_batch)[0]

                if (
                    iou < self.reconstruction_bbox_iou_thresh
                    and det_score >= self.reconstruction_bbox_det_score
                ):
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )

        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    tracker_low_res_masks_global = (
                        self._suppress_overlapping_based_on_recent_occlusion(
                            frame_idx,
                            tracker_low_res_masks_global,
                            tracker_metadata_prev,
                            tracker_metadata_new,
                            obj_ids_newly_removed,
                            reverse,
                        )
                    )

            self._tracker_update_memories(
                tracker_states_local,
                frame_idx,
                tracker_metadata=tracker_metadata_prev,
                low_res_masks=tracker_low_res_masks_global,
            )

        for rank in range(self.world_size):
            new_det_obj_ids_this_gpu = new_det_obj_ids[new_det_gpu_ids == rank]
            updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids_per_gpu"][rank]
            if len(new_det_obj_ids_this_gpu) > 0:
                updated_obj_ids_this_gpu = np.concatenate(
                    [updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu]
                )
            if len(obj_ids_newly_removed) > 0:
                is_removed = np.isin(
                    updated_obj_ids_this_gpu, list(obj_ids_newly_removed)
                )
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
            tracker_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            tracker_metadata_new["num_obj_per_gpu"][rank] = len(
                updated_obj_ids_this_gpu
            )
        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata_new["obj_ids_per_gpu"]
        )
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"],
                np.max(new_det_obj_ids),
            )
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][
                obj_id
            ] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        assert ("rank0_metadata" in tracker_metadata_new) == (self.rank == 0)
        if self.rank == 0 and self.masklet_confirmation_enable:
            rank0_metadata = self.update_masklet_confirmation_status(
                rank0_metadata=tracker_metadata_new["rank0_metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids_all_gpu"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids_all_gpu"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["rank0_metadata"] = rank0_metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_prev: Dict[str, Any],
        tracker_metadata_new: Dict[str, Any],
        obj_ids_newly_removed: Set[int],
        reverse: bool = False,
    ):
        obj_ids_global = tracker_metadata_prev["obj_ids_all_gpu"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert (
                len(obj_ids_global) == batch_size
            ), f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                NEVER_OCCLUDED
                                if obj_id not in obj_ids_newly_removed
                                else ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1]
                for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            NO_OBJ_LOGIT = -10
            tracker_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_states_local: List[Any],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        new_det_gpu_ids: npt.NDArray = tracker_update_plan["new_det_gpu_ids"]
        is_on_this_gpu: npt.NDArray = new_det_gpu_ids == self.rank
        new_det_obj_ids_local: npt.NDArray = new_det_obj_ids[is_on_this_gpu]
        new_det_fa_inds_local: npt.NDArray = new_det_fa_inds[is_on_this_gpu]
        obj_ids_newly_removed: Set[int] = tracker_update_plan["obj_ids_newly_removed"]

        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: Tensor = det_out["mask"][new_det_fa_inds_local_t]
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    def build_outputs(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        reconditioned_obj_ids: set = None,
        det_to_matched_trk_obj_ids: dict = None,
    ):
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}

        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
        existing_masklet_video_res_masks = F.interpolate(
            tracker_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )
        existing_masklet_binary = existing_masklet_video_res_masks > 0
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask

        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        if len(new_det_fa_inds) > 0:
            new_det_low_res_masks = fill_holes_in_mask_scores(
                new_det_low_res_masks,
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
        new_masklet_video_res_masks = F.interpolate(
            new_det_low_res_masks,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )

        new_masklet_binary = new_masklet_video_res_masks > 0
        assert len(new_det_obj_ids) == len(new_masklet_video_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_masklet_binary):
            obj_id_to_mask[obj_id] = mask

        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get(
                "trk_id_to_max_iou_high_conf_det", {}
            )

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    det_mask = det_out["mask"][det_idx]
                    det_mask = det_mask.unsqueeze(0).unsqueeze(0)
                    det_mask_resized = (
                        F.interpolate(
                            det_mask.float(),
                            size=(orig_vid_height, orig_vid_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    )

                    det_mask_final = det_mask_resized.squeeze(0)
                    obj_id_to_mask[obj_id] = det_mask_final

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: List[int],
        obj_ids: List[int],
        frame_idx: int = None,
        reverse: bool = False,
    ):
        assert (
            binary_low_res_masks.dtype == torch.bool
        ), f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)

        mask_iou_thresh = (
            iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)

        last_occ_expanded_i = last_occluded.unsqueeze(1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_i, last_occ_expanded_j)
            & (last_occ_expanded_j > -1)
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (last_occ_expanded_i > -1)
        )
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        if (
            self.rank == 0
            and logger.isEnabledFor(logging.DEBUG)
            and frame_idx is not None
        ):
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            batch_size = suppress_i_mask.shape[0]

            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(
        self,
        inference_states: List[Any],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = False,
    ):
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue

            num_frames_propagated = 0
            for out in self.tracker.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                reverse=reverse,
                tqdm_disable=True,
                run_mem_encoder=run_mem_encoder,
            ):
                out_frame_idx, out_obj_ids, out_low_res_masks, _, out_obj_scores = out
                num_frames_propagated += 1

            assert (
                num_frames_propagated == 1 and out_frame_idx == frame_idx
            ), f"num_frames_propagated: {num_frames_propagated}, out_frame_idx: {out_frame_idx}, frame_idx: {frame_idx}"
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        H_mask = W_mask = self.tracker.low_res_mask_size
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)

            low_res_masks_local = fill_holes_in_mask_scores(
                low_res_masks_local.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores_np: npt.NDArray,
        trk_masks: Tensor,
        trk_obj_ids: npt.NDArray,
    ):
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert (
            trk_masks.size(0) == len(trk_obj_ids)
        ), f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        if trk_masks.size(0) == 0:
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary, trk_masks_binary)

        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            cost_matrix = 1 - ious_np
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]

        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}
        HIGH_CONF_THRESH = 0.8
        HIGH_IOU_THRESH = 0.8
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(
            np.nonzero(det_is_high_conf & det_is_high_iou)[0]
        )
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d

        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        workload_per_gpu: npt.NDArray = prev_workload_per_gpu.copy()
        new_det_gpu_ids = np.zeros(new_det_num, np.int64)

        for i in range(len(new_det_gpu_ids)):
            min_gpu = np.argmin(workload_per_gpu)
            new_det_gpu_ids[i] = min_gpu
            workload_per_gpu[min_gpu] += 1
        return new_det_gpu_ids

    def _process_hotstart(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
        empty_trk_obj_ids: npt.NDArray,
        unmatched_trk_obj_ids: npt.NDArray,
        rank0_metadata: Dict[str, Any],
        tracker_metadata: Dict[str, Any],
    ):
        obj_first_frame_idx = rank0_metadata["obj_first_frame_idx"]
        unmatched_frame_inds = rank0_metadata["unmatched_frame_inds"]
        trk_keep_alive = rank0_metadata["trk_keep_alive"]
        overlap_pair_to_frame_inds = rank0_metadata["overlap_pair_to_frame_inds"]
        removed_obj_ids = rank0_metadata["removed_obj_ids"]
        suppressed_obj_ids = rank0_metadata["suppressed_obj_ids"][frame_idx]

        obj_ids_newly_removed = set()
        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )

        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            trk_keep_alive[obj_id] = min(
                self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1
            )
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            trk_keep_alive[obj_id] = max(
                self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
            )
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                trk_keep_alive[obj_id] = max(
                    self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
                )

        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id] > hotstart_diff and not reverse
                ) or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    logger.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                logger.debug(
                    f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched"
                )
                suppressed_obj_ids.add(obj_id)

        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    logger.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)

        return obj_ids_newly_removed, rank0_metadata

    def _tracker_update_memories(
        self,
        tracker_inference_states: List[Any],
        frame_idx: int,
        tracker_metadata: Dict[str, Any],
        low_res_masks: Tensor,
    ):
        if len(tracker_inference_states) == 0:
            return
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(high_res_H, high_res_W),
            mode="bilinear",
            align_corners=False,
        )
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker._suppress_object_pw_area_shrinkage(
                high_res_masks
            )
        object_score_logits = torch.where(
            (high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0
        )

        start_idx_gpu = sum(tracker_metadata["num_obj_per_gpu"][: self.rank])
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[
                start_idx_state:end_idx_state
            ]
            local_batch_size = local_high_res_masks.size(0)

            encoded_mem = self.tracker._run_memory_encoder(
                tracker_state,
                frame_idx,
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            output_dict = tracker_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = (
                    local_maskmem_features
                )
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [
                    pos for pos in local_maskmem_pos_enc
                ]
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: List[int],
        new_obj_masks: Tensor,
        tracker_states_local: List[Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        prev_tracker_state = (
            tracker_states_local[0] if len(tracker_states_local) > 0 else None
        )

        new_tracker_state = self.tracker.init_state(
            cached_features=feature_cache,
            video_height=orig_vid_height,
            video_width=orig_vid_width,
            num_frames=num_frames,
        )
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None)
            if prev_tracker_state is not None
            else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        input_mask_res = self.tracker.input_mask_size
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(1),
            size=(input_mask_res, input_mask_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        new_obj_masks = new_obj_masks > 0

        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_mask(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                mask=new_mask,
                add_mask_to_memory=True,
            )
        self.tracker.propagate_in_video_preflight(
            new_tracker_state, run_mem_encoder=True
        )
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_object(self, tracker_states_local: List[Any], obj_id: int):
        tracker_states_local_before_removal = tracker_states_local.copy()
        tracker_states_local.clear()
        for tracker_inference_state in tracker_states_local_before_removal:
            new_obj_ids, _ = self.tracker.remove_object(
                tracker_inference_state, obj_id, strict=False, need_output=False
            )
            if len(new_obj_ids) > 0:
                tracker_states_local.append(tracker_inference_state)

    def _tracker_remove_objects(
        self, tracker_states_local: List[Any], obj_ids: list[int]
    ):
        for obj_id in obj_ids:
            self._tracker_remove_object(tracker_states_local, obj_id)

    def _initialize_metadata(self):
        tracker_metadata = {
            "obj_ids_per_gpu": [np.array([], np.int64) for _ in range(self.world_size)],
            "obj_ids_all_gpu": np.array([], np.int64),
            "num_obj_per_gpu": np.zeros(self.world_size, np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        if self.rank == 0:
            rank0_metadata = {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(int),
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(set),
            }
            if self.masklet_confirmation_enable:
                rank0_metadata["masklet_confirmation"] = {
                    "status": np.array([], np.int64),
                    "consecutive_det_num": np.array([], np.int64),
                }
            tracker_metadata["rank0_metadata"] = rank0_metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        rank0_metadata: Dict[str, Any],
        obj_ids_all_gpu_prev: npt.NDArray,
        obj_ids_all_gpu_updated: npt.NDArray,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
    ):
        confirmation_data = rank0_metadata["masklet_confirmation"]

        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert (
            status_prev.shape == obj_ids_all_gpu_prev.shape
        ), f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"

        obj_id_to_updated_idx = {
            obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)
        }
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        unconfirmed_val = MaskletConfirmationStatus.UNCONFIRMED.value
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[
            prev_elem_is_in_updated
        ]

        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)

        change_to_confirmed = (
            consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        )
        status[change_to_confirmed] = MaskletConfirmationStatus.CONFIRMED.value

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return rank0_metadata

    def _encode_prompt(self, **kwargs):
        return self.detector._encode_prompt(**kwargs)

    def _drop_new_det_with_obj_limit(self, new_det_fa_inds, det_scores_np, num_to_keep):
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds

        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds


# ---------------------------------------------------------------------------
# Sam3VideoInference (from model/sam3_video_inference.py)
# ---------------------------------------------------------------------------

class Sam3VideoInference(Sam3VideoBase):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1

    def __init__(
        self,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
        **kwargs,
    ):
        """
        hotstart_delay: int, the delay (in #frames) before the model starts to yield output, 0 to disable hotstart delay.
        hotstart_unmatch_thresh: int, remove the object if it has this many unmatched frames within its hotstart_delay period.
            If `hotstart_delay` is set to 0, this parameter is ignored.
        hotstart_dup_thresh: int, remove the object if it has overlapped with another object this many frames within its hotstart_delay period.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        video_loader_type="cv2",
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""
        # Get actual current device from model parameters
        device = next(self.parameters()).device

        images, orig_height, orig_width = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            device=device,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        # the original video height and width, used for resizing final output scores
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # inputs on each frame
        self._construct_initial_input_batch(inference_state, images)
        # initialize extra states
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []  # for logging user actions
        inference_state["is_image_only"] = is_image_type(resource_path)
        return inference_state

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Revert `inference_state` to what it was right after initialization."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            # constructing an output list in inference state (we start with an empty list)
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0

        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        inference_state["tracker_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"].clear()
        inference_state["action_history"].clear()  # for logging user actions

    def _construct_initial_input_batch(self, inference_state, images):
        """Construct an initial `BatchedDatapoint` instance as input."""
        # 1) img_batch
        num_frames = len(images)
        # Get actual current device from model parameters (handles dynamic device changes)
        device = next(self.parameters()).device

        # 2) find_text_batch
        # "<text placeholder>" will be replaced by the actual text prompt when adding prompts
        find_text_batch = ["<text placeholder>", "visual"]

        # 3) find_inputs
        input_box_embedding_dim = 258  # historical default
        input_points_embedding_dim = 257  # historical default
        stages = [
            FindStage(
                img_ids=[stage_id],
                text_ids=[0],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]
        for i in range(len(stages)):
            stages[i] = convert_my_tensors(stages[i])

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = BatchedDatapoint(
            img_batch=images,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            find_metadatas=[None] * num_frames,
        )
        input_batch = copy_data_to_device(input_batch, device, non_blocking=torch.cuda.is_available())
        inference_state["input_batch"] = input_batch

        # construct the placeholder interactive prompts and tracking queries
        bs = 1
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=device),
            box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=device),
            point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        )

        # constructing an output list in inference state (we start with an empty list)
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames

        # placeholders for cached outputs
        # (note: currently, a single visual prompt embedding is shared for all frames)
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        """
        Handle the case of visual prompt. Currently, in the inference API we do not
        explicitly distinguish between initial box as visual prompt vs subsequent boxes
        or boxes after inference for refinement.
        """
        # If the frame hasn't had any inference results before (prompting or propagation),
        # we treat the first added box prompt as a visual prompt; otherwise, we treat
        # the first box just as a refinement prompt.
        is_new_visual_prompt = (
            inference_state["per_frame_visual_prompt"][frame_idx] is None
            and inference_state["previous_stages_out"][frame_idx] is None
        )
        if is_new_visual_prompt:
            if boxes_cxcywh.size(0) != 1:
                raise RuntimeError(
                    "visual prompts (box as an initial prompt) should only have one box, "
                    f"but got {boxes_cxcywh.shape=}"
                )
            if not box_labels.item():
                logging.warning("A negative box is added as a visual prompt.")
            # take the first box prompt as a visual prompt
            device = self.device
            new_visual_prompt = Prompt(
                box_embeddings=boxes_cxcywh[None, 0:1, :].to(device),  # (seq, bs, 4)
                box_mask=None,
                box_labels=box_labels[None, 0:1].to(device),  # (seq, bs)
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )
            inference_state["per_frame_visual_prompt"][frame_idx] = new_visual_prompt
        else:
            new_visual_prompt = None

        # `boxes_cxcywh` and `box_labels` contains all the raw box inputs added so far
        # strip any visual prompt from the input boxes (for geometric prompt encoding)
        if inference_state["per_frame_visual_prompt"][frame_idx] is not None:
            boxes_cxcywh = boxes_cxcywh[1:]
            box_labels = box_labels[1:]

        return boxes_cxcywh, box_labels, new_visual_prompt

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at least one frame before propagation."
            )
        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t for t, out in enumerate(previous_stages_out) if out is not None
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = start_frame_idx - max_frame_num_to_track
            end_frame_idx = max(end_frame_idx, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = start_frame_idx + max_frame_num_to_track
            end_frame_idx = min(end_frame_idx, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """
        Propagate the prompts to get grounding results for the entire video. This method
        is a generator and yields inference outputs for all frames in the range specified
        by `start_frame_idx`, `max_frame_num_to_track`, and `reverse`.
        """
        # compile the model (it's a no-op if the model is already compiled)
        # note that it's intentionally added to `self.propagate_in_video`, so that the first
        # `self.add_prompt` call will be done in eager mode to fill in the decoder buffers
        # such as positional encoding cache)
        self._compile_model()

        processing_order, end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )

        # Store max_frame_num_to_track in feature_cache for downstream methods
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }

        hotstart_buffer = []
        hotstart_removed_obj_ids = set()
        # when deciding whether to output a masklet on `yield_frame_idx`, we check whether the object is confirmed
        # in a future frame (`unconfirmed_frame_delay` frames after the current frame). For example, if we require
        # an object to be detected in 3 consecutive frames to be confirmed, then we look 2 frames in the future --
        # e.g., we output an object on frame 4 only if it becomes confirmed on frame 6.
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1
        unconfirmed_obj_ids_per_frame = {}  # frame_idx -> hidden_obj_ids
        for frame_idx in tqdm(
            processing_order, desc="propagate_in_video", disable=self.rank > 0
        ):
            out = self._run_single_frame_inference(inference_state, frame_idx, reverse)

            if self.hotstart_delay > 0:
                # accumulate the outputs for the first `hotstart_delay` frames
                hotstart_buffer.append([frame_idx, out])
                # update the object IDs removed by hotstart so that we don't output them
                if self.rank == 0:
                    hotstart_removed_obj_ids.update(out["removed_obj_ids"])
                    unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                    if unconfirmed_obj_ids is not None:
                        unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids

                if frame_idx == end_frame_idx:
                    # we reached the end of propagation -- yield all frames in the buffer
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.hotstart_delay:
                    # we have enough frames -- yield and remove the first (oldest) frame from the buffer
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    # not enough frames yet -- skip yielding
                    yield_list = []
            else:
                yield_list = [(frame_idx, out)]  # output the current frame

            for yield_frame_idx, yield_out in yield_list:
                # post-process the output and yield it
                if self.rank == 0:
                    suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                    unconfirmed_status_frame_idx = (
                        yield_frame_idx + unconfirmed_status_delay
                        if not reverse
                        else yield_frame_idx - unconfirmed_status_delay
                    )

                    # Clamp the frame index to stay within video bounds
                    num_frames = inference_state["num_frames"]
                    unconfirmed_status_frame_idx = max(
                        0, min(unconfirmed_status_frame_idx, num_frames - 1)
                    )

                    unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(
                        unconfirmed_status_frame_idx, None
                    )
                    postprocessed_out = self._postprocess_output(
                        inference_state,
                        yield_out,
                        hotstart_removed_obj_ids,
                        suppressed_obj_ids,
                        unconfirmed_obj_ids,
                    )

                    self._cache_frame_outputs(
                        inference_state,
                        yield_frame_idx,
                        yield_out["obj_id_to_mask"],
                        suppressed_obj_ids=suppressed_obj_ids,
                        removed_obj_ids=hotstart_removed_obj_ids,
                        unconfirmed_obj_ids=unconfirmed_obj_ids,
                    )
                else:
                    postprocessed_out = None  # no output on other GPUs
                yield yield_frame_idx, postprocessed_out

    def _run_single_frame_inference(self, inference_state, frame_idx, reverse):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        # prepare inputs
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = (
            inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        )
        # run inference for the current frame
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=(
                inference_state["constants"]["empty_geometric_prompt"]
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=inference_state["is_image_only"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # update inference state
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        # use a dummy string in "previous_stages_out" to indicate this frame has outputs
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        if self.rank == 0:
            self._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # first frame detection score
            "obj_id_to_tracker_score": tracker_metadata_new[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx],
        }
        # removed_obj_ids is only needed on rank 0 to handle hotstart delay buffer
        if self.rank == 0:
            rank0_metadata = tracker_metadata_new["rank0_metadata"]
            removed_obj_ids = rank0_metadata["removed_obj_ids"]
            out["removed_obj_ids"] = removed_obj_ids
            out["suppressed_obj_ids"] = rank0_metadata["suppressed_obj_ids"][frame_idx]
            out["frame_stats"] = frame_stats
            if self.masklet_confirmation_enable:
                status = rank0_metadata["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][
                    is_unconfirmed
                ].tolist()
            else:
                out["unconfirmed_obj_ids"] = []

        return out

    def _postprocess_output(
        self,
        inference_state,
        out,
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        obj_id_to_mask = out["obj_id_to_mask"]  # low res masks
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        H_video, W_video = inference_state["orig_height"], inference_state["orig_width"]
        if len(curr_obj_ids) == 0:
            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_probs = torch.zeros(0, dtype=torch.float32)
            out_binary_masks = torch.zeros(0, H_video, W_video, dtype=torch.bool)
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
        else:
            out_obj_ids = torch.tensor(curr_obj_ids, dtype=torch.int64)
            out_probs = torch.tensor(
                [out["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids]
            )
            out_tracker_probs = torch.tensor(
                [
                    (
                        out["obj_id_to_tracker_score"][obj_id]
                        if obj_id in out["obj_id_to_tracker_score"]
                        else 0.0
                    )
                    for obj_id in curr_obj_ids
                ]
            )
            out_binary_masks = torch.cat(
                [obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0
            )

            assert out_binary_masks.dtype == torch.bool

            keep = out_binary_masks.any(dim=(1, 2)).cpu()  # remove masks with 0 areas
            # hide outputs for those object IDs in `obj_ids_to_hide`
            obj_ids_to_hide = []
            if suppressed_obj_ids is not None:
                obj_ids_to_hide.extend(suppressed_obj_ids)
            if removed_obj_ids is not None:
                obj_ids_to_hide.extend(removed_obj_ids)
            if unconfirmed_obj_ids is not None:
                obj_ids_to_hide.extend(unconfirmed_obj_ids)
            if len(obj_ids_to_hide) > 0:
                obj_ids_to_hide_t = torch.tensor(obj_ids_to_hide, dtype=torch.int64)
                keep &= ~torch.isin(out_obj_ids, obj_ids_to_hide_t)

            # slice those valid entries from the original outputs
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]

            # Only pin_memory if device is CUDA (requires accelerator)
            if out_binary_masks.device.type == "cuda":
                keep_idx_gpu = keep_idx.pin_memory().to(
                    device=out_binary_masks.device, non_blocking=True
                )
            else:
                keep_idx_gpu = keep_idx.to(device=out_binary_masks.device)

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_tracker_probs = torch.index_select(out_tracker_probs, 0, keep_idx)
            out_binary_masks = torch.index_select(out_binary_masks, 0, keep_idx_gpu)

            if perflib.is_enabled:
                out_boxes_xyxy = perf_masks_to_boxes(
                    out_binary_masks, out_obj_ids.tolist()
                )
            else:
                out_boxes_xyxy = masks_to_boxes(out_binary_masks)

            out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)  # convert to xywh format
            # normalize boxes
            out_boxes_xywh[..., 0] /= W_video
            out_boxes_xywh[..., 1] /= H_video
            out_boxes_xywh[..., 2] /= W_video
            out_boxes_xywh[..., 3] /= H_video

        # apply non-overlapping constraints on the existing masklets
        if out_binary_masks.shape[0] > 1:
            assert len(out_binary_masks) == len(out_tracker_probs)
            out_binary_masks = (
                self.tracker._apply_object_wise_non_overlapping_constraints(
                    out_binary_masks.unsqueeze(1),
                    out_tracker_probs.unsqueeze(1).to(out_binary_masks.device),
                    background_value=0,
                ).squeeze(1)
            ) > 0

        outputs = {
            "obj_ids": out_obj_ids.cpu().numpy(),
            "out_probs": out_probs.cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
            "frame_stats": out.get("frame_stats", None),
        }
        return outputs

    def _cache_frame_outputs(
        self,
        inference_state,
        frame_idx,
        obj_id_to_mask,
        suppressed_obj_ids=None,
        removed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        # Filter out suppressed, removed, and unconfirmed objects from the cache
        filtered_obj_id_to_mask = obj_id_to_mask.copy()

        objects_to_exclude = set()
        if suppressed_obj_ids is not None:
            objects_to_exclude.update(suppressed_obj_ids)
        if removed_obj_ids is not None:
            objects_to_exclude.update(removed_obj_ids)
        if unconfirmed_obj_ids is not None:
            objects_to_exclude.update(unconfirmed_obj_ids)

        if objects_to_exclude:
            for obj_id in objects_to_exclude:
                if obj_id in filtered_obj_id_to_mask:
                    del filtered_obj_id_to_mask[obj_id]

        inference_state["cached_frame_outputs"][frame_idx] = filtered_obj_id_to_mask

    def _build_tracker_output(
        self, inference_state, frame_idx, refined_obj_id_to_mask=None
    ):
        assert (
            "cached_frame_outputs" in inference_state
            and frame_idx in inference_state["cached_frame_outputs"]
        ), "No cached outputs found. Ensure normal propagation has run first to populate the cache."
        cached_outputs = inference_state["cached_frame_outputs"][frame_idx]

        obj_id_to_mask = cached_outputs.copy()

        # Update with refined masks if provided
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                assert (
                    refined_mask is not None
                ), f"Refined mask data must be provided for obj_id {obj_id}"
                obj_id_to_mask[obj_id] = refined_mask

        return obj_id_to_mask

    def _compile_model(self):
        """No-op: ComfyUI handles optimization through its execution engine."""
        pass

    def _warm_up_vg_propagation(self, inference_state, start_frame_idx=0):
        # use different tracking score thresholds for each round to simulate different number of output objects
        num_objects_list = range(self.num_obj_for_compile + 1)
        new_det_score_thresh_list = [0.3, 0.5, 0.7]
        num_rounds = len(new_det_score_thresh_list)
        orig_new_det_thresh = self.new_det_thresh

        for i, thresh in enumerate(new_det_score_thresh_list):
            self.new_det_thresh = thresh
            for num_objects in num_objects_list:
                logger.info(f"{i+1}/{num_rounds} warming up model compilation")
                self.add_prompt(
                    inference_state, frame_idx=start_frame_idx, text_str="cat"
                )
                logger.info(
                    f"{i+1}/{num_rounds} warming up model compilation -- simulating {num_objects}/{self.num_obj_for_compile} objects"
                )
                inference_state = self.add_fake_objects_to_inference_state(
                    inference_state, num_objects, frame_idx=start_frame_idx
                )
                inference_state["tracker_metadata"]["rank0_metadata"].update(
                    {
                        "masklet_confirmation": {
                            "status": np.zeros(num_objects, dtype=np.int64),
                            "consecutive_det_num": np.zeros(
                                num_objects, dtype=np.int64
                            ),
                        }
                    }
                )
                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=False
                ):
                    pass
                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=True
                ):
                    pass
                self.reset_state(inference_state)
                logger.info(
                    f"{i+1}/{num_rounds} warming up model compilation -- completed round {i+1} out of {num_rounds}"
                )

        # Warm up Tracker memory encoder with varying input shapes
        num_iters = 3
        feat_size = self.tracker.sam_image_embedding_size**2  # 72 * 72 = 5184
        hidden_dim = self.tracker.hidden_dim  # 256
        mem_dim = self.tracker.mem_dim  # 64
        for _ in tqdm(range(num_iters)):
            for b in range(1, self.num_obj_for_compile + 1):
                for i in range(
                    1,
                    self.tracker.max_cond_frames_in_attn + self.tracker.num_maskmem,
                ):
                    for j in range(
                        self.tracker.max_cond_frames_in_attn
                        + self.tracker.max_obj_ptrs_in_encoder
                    ):
                        num_obj_ptr_tokens = (hidden_dim // mem_dim) * j
                        src = torch.randn(feat_size, b, hidden_dim, device=self.device)
                        src_pos = torch.randn(
                            feat_size, b, hidden_dim, device=self.device
                        )
                        prompt = torch.randn(
                            feat_size * i + num_obj_ptr_tokens,
                            b,
                            mem_dim,
                            device=self.device,
                        )
                        prompt_pos = torch.randn(
                            feat_size * i + num_obj_ptr_tokens,
                            b,
                            mem_dim,
                            device=self.device,
                        )

                        self.tracker.transformer.encoder.forward(
                            src=src,
                            src_pos=src_pos,
                            prompt=prompt,
                            prompt_pos=prompt_pos,
                            num_obj_ptr_tokens=num_obj_ptr_tokens,
                        )

        self.new_det_thresh = orig_new_det_thresh
        return inference_state

    def add_fake_objects_to_inference_state(
        self, inference_state, num_objects, frame_idx
    ):
        new_det_obj_ids_local = np.arange(num_objects)
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        new_det_masks = torch.ones(
            len(new_det_obj_ids_local), high_res_H, high_res_W
        ).to(self.device)

        inference_state["tracker_inference_states"] = self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            new_obj_ids=new_det_obj_ids_local,
            new_obj_masks=new_det_masks,
            tracker_states_local=inference_state["tracker_inference_states"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            feature_cache=inference_state["feature_cache"],
        )

        # Synthesize obj_id_to_mask data for cached_frame_outputs to support _build_tracker_output during warmup
        obj_id_to_mask = {}
        if num_objects > 0:
            H_video = inference_state["orig_height"]
            W_video = inference_state["orig_width"]

            video_res_masks = F.interpolate(
                new_det_masks.unsqueeze(1),  # Add channel dimension for interpolation
                size=(H_video, W_video),
                mode="bilinear",
                align_corners=False,
            )  # (num_objects, 1, H_video, W_video)
            for i, obj_id in enumerate(new_det_obj_ids_local):
                obj_id_to_mask[obj_id] = (video_res_masks[i] > 0.0).to(torch.bool)
        if self.rank == 0:
            for fidx in range(inference_state["num_frames"]):
                self._cache_frame_outputs(inference_state, fidx, obj_id_to_mask)

        inference_state["tracker_metadata"].update(
            {
                "obj_ids_per_gpu": [np.arange(num_objects)],
                "obj_ids_all_gpu": np.arange(num_objects),  # Same as 1 GPU
                "num_obj_per_gpu": [num_objects],
                "obj_id_to_score": {i: 1.0 for i in range(num_objects)},
                "max_obj_id": num_objects,
                "rank0_metadata": {
                    "masklet_confirmation": {
                        "status": np.zeros(num_objects, dtype=np.int64),
                        "consecutive_det_num": np.zeros(num_objects, dtype=np.int64),
                    },
                    "removed_obj_ids": set(),
                    "suppressed_obj_ids": defaultdict(set),
                },
            }
        )
        return inference_state

    @torch.inference_mode()
    def warm_up_compilation(self):
        """
        Warm up the model by running a dummy inference to compile the model. This is
        useful to avoid the compilation overhead in the first inference call.
        """
        if not self.compile_model:
            return
        self._warm_up_complete = False
        if self.device.type != "cuda":
            return

        # temporally set to single GPU temporarily for warm-up compilation
        orig_rank = self.rank
        orig_world_size = self.world_size
        self.rank = self.detector.rank = 0
        self.world_size = self.detector.world_size = 1
        orig_recondition_every_nth_frame = self.recondition_every_nth_frame

        # Get a random video
        inference_state = self.init_state(resource_path="<load-dummy-video-30>")
        start_frame_idx = 0

        # Run basic propagation warm-up
        inference_state = self._warm_up_vg_propagation(inference_state, start_frame_idx)

        logger.info("Warm-up compilation completed.")

        # revert to the original GPU and rank
        self.rank = self.detector.rank = orig_rank
        self.world_size = self.detector.world_size = orig_world_size
        self.recondition_every_nth_frame = orig_recondition_every_nth_frame
        self._warm_up_complete = True
        self.tracker.transformer.encoder.forward.set_logging(True)

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        boxes_xywh=None,
        box_labels=None,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
        logger.debug("Running add_prompt on frame %d", frame_idx)

        num_frames = inference_state["num_frames"]
        assert (
            text_str is not None or boxes_xywh is not None
        ), "at least one type of prompt (text, boxes) must be provided"
        assert (
            0 <= frame_idx < num_frames
        ), f"{frame_idx=} is out of range for a total of {num_frames} frames"

        # since it's a semantic prompt, we start over
        self.reset_state(inference_state)

        # 1) add text prompt
        if text_str is not None and text_str != "visual":
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            text_id = self.TEXT_ID_FOR_TEXT
        else:
            inference_state["text_prompt"] = None
            inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
            text_id = self.TEXT_ID_FOR_VISUAL
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) handle box prompt
        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            # input boxes are expected to be [xmin, ymin, width, height] format
            # in normalized coordinates of range 0~1, similar to FA
            assert boxes_xywh.dim() == 2
            assert boxes_xywh.size(0) > 0 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
            assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            assert (boxes_cxcywh >= 0).all().item() and (boxes_cxcywh <= 1).all().item()

            new_box_input = boxes_cxcywh, box_labels
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, geometric_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )

            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        out = self._run_single_frame_inference(
            inference_state, frame_idx, reverse=False
        )
        return frame_idx, self._postprocess_output(inference_state, out)



# ---------------------------------------------------------------------------
# Sam3VideoInferenceWithInstanceInteractivity
# ---------------------------------------------------------------------------

class Sam3VideoInferenceWithInstanceInteractivity(Sam3VideoInference):
    def __init__(
        self,
        use_prev_mem_frame=False,
        use_stateless_refinement=False,
        refinement_detector_cond_frame_removal_window=16,
        **kwargs,
    ):
        """
        use_prev_mem_frame: bool, whether to condition on previous memory frames for adding points
        use_stateless_refinement: bool, whether to enable stateless refinement behavior
        refinement_detector_cond_frame_removal_window: int, we remove a detector conditioning frame if it
            is within this many frames of a user refined frame. Set to a large value (e.g. 10000) to
            always remove detector conditioning frames if there is any user refinement in the video.
        """
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.use_stateless_refinement = use_stateless_refinement
        self.refinement_detector_cond_frame_removal_window = (
            refinement_detector_cond_frame_removal_window
        )

    def _init_new_tracker_state(self, inference_state):
        return self.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
        )

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        # step 1: check which type of propagation to run, should be the same for all GPUs.
        propagation_type, obj_ids = self.parse_action_history_for_propagation(
            inference_state
        )
        self.add_action_history(
            inference_state,
            action_type=propagation_type,
            obj_ids=obj_ids,
            frame_idx=start_frame_idx,
        )

        # step 2: run full VG propagation
        if propagation_type == "propagation_full":
            logger.debug(f"Running full VG propagation (reverse={reverse}).")
            yield from super().propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
            )
            return

        # step 3: run Tracker partial propagation or direct fetch existing predictions
        assert propagation_type in ["propagation_partial", "propagation_fetch"]
        logger.debug(
            f"Running Tracker propagation for objects {obj_ids} and merging it with existing VG predictions (reverse={reverse})."
            if propagation_type == "propagation_partial"
            else f"Fetching existing VG predictions without running any propagation (reverse={reverse})."
        )
        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

        tracker_metadata = inference_state["tracker_metadata"]

        # if fetch just return from output
        if propagation_type == "propagation_fetch":
            for frame_idx in tqdm(processing_order):
                if self.rank == 0:
                    obj_id_to_mask = inference_state["cached_frame_outputs"].get(
                        frame_idx, {}
                    )
                    # post processing - remove suppressed obj_ids
                    obj_id_to_score = tracker_metadata["obj_id_to_score"]
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    obj_id_to_tracker_score = tracker_metadata[
                        "obj_id_to_tracker_score_frame_wise"
                    ][frame_idx]

                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": obj_id_to_score,
                        "obj_id_to_tracker_score": obj_id_to_tracker_score,
                    }
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, None

            return

        # get Tracker inference states containing selected obj_ids
        if propagation_type == "propagation_partial":
            # can be empty for GPUs where objects are not in their inference states
            tracker_states_local = self._get_tracker_inference_states_by_obj_ids(
                inference_state, obj_ids
            )
            for tracker_state in tracker_states_local:
                self.tracker.propagate_in_video_preflight(
                    tracker_state, run_mem_encoder=True
                )

        for frame_idx in tqdm(processing_order):
            # run Tracker propagation
            if propagation_type == "propagation_partial":
                self._prepare_backbone_feats(inference_state, frame_idx, reverse)
                obj_ids_local, low_res_masks_local, tracker_scores_local = (
                    self._propogate_tracker_one_frame_local_gpu(
                        tracker_states_local,
                        frame_idx=frame_idx,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                )

                # broadcast refined object tracker scores and masks to all GPUs
                # handle multiple objects that can be located on different GPUs
                refined_obj_data = {}  # obj_id -> (score, mask_video_res)

                # Collect data for objects on this GPU
                local_obj_data = {}
                for obj_id in obj_ids:
                    obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                    if self.rank == obj_rank and obj_id in obj_ids_local:
                        refined_obj_idx = obj_ids_local.index(obj_id)
                        refined_mask_low_res = low_res_masks_local[
                            refined_obj_idx
                        ]  # (H_low_res, W_low_res)
                        refined_score = tracker_scores_local[refined_obj_idx]

                        # Keep low resolution for broadcasting to reduce communication cost
                        local_obj_data[obj_id] = (refined_score, refined_mask_low_res)

                # Broadcast data from each GPU that has refined objects
                if self.world_size > 1:
                    for obj_id in obj_ids:
                        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                        if self.rank == obj_rank:
                            # This GPU has the object, broadcast its data
                            data_to_broadcast = local_obj_data.get(obj_id, None)
                            data_list = [
                                (data_to_broadcast[0].cpu(), data_to_broadcast[1].cpu())
                            ]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            if data_to_broadcast is not None:
                                refined_obj_data[obj_id] = data_to_broadcast
                        elif self.rank != obj_rank:
                            # This GPU doesn't have the object, receive data
                            data_list = [None]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            refined_obj_data[obj_id] = (
                                data_list[0][0].to(self.device),
                                data_list[0][1].to(self.device),
                            )
                else:
                    # Single GPU case
                    refined_obj_data = local_obj_data

                # Update Tracker scores for all refined objects
                for obj_id, (refined_score, _) in refined_obj_data.items():
                    tracker_metadata["obj_id_to_tracker_score_frame_wise"][
                        frame_idx
                    ].update({obj_id: refined_score.item()})

                if self.rank == 0:
                    # get predictions from Tracker inference states, it includes the original
                    # VG predictions and the refined predictions from interactivity.

                    # Prepare refined masks dictionary - upscale to video resolution after broadcast
                    refined_obj_id_to_mask = {}
                    for obj_id, (_, refined_mask_low_res) in refined_obj_data.items():
                        refined_mask_video_res = (
                            self._convert_low_res_mask_to_video_res(
                                refined_mask_low_res, inference_state
                            )
                        )  # (1, H_video, W_video) bool
                        refined_obj_id_to_mask[obj_id] = refined_mask_video_res

                    # Initialize cache if not present (needed for point prompts during propagation)
                    if "cached_frame_outputs" not in inference_state:
                        inference_state["cached_frame_outputs"] = {}
                    if frame_idx not in inference_state["cached_frame_outputs"]:
                        inference_state["cached_frame_outputs"][frame_idx] = {}

                    obj_id_to_mask = self._build_tracker_output(
                        inference_state, frame_idx, refined_obj_id_to_mask
                    )
                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": tracker_metadata["obj_id_to_score"],
                        "obj_id_to_tracker_score": tracker_metadata[
                            "obj_id_to_tracker_score_frame_wise"
                        ][frame_idx],
                    }
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    self._cache_frame_outputs(
                        inference_state,
                        frame_idx,
                        obj_id_to_mask,
                        suppressed_obj_ids=suppressed_obj_ids,
                    )
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, None

    def add_action_history(
        self, inference_state, action_type, frame_idx=None, obj_ids=None
    ):
        """
        action_history is used to automatically decide what to do during propagation.
        action_type: one of ["add", "remove", "refine"] + ["propagation_full", "propagation_partial", "propagation_fetch"]
        """
        instance_actions = ["add", "remove", "refine"]
        propagation_actions = [
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
        ]
        assert (
            action_type in instance_actions + propagation_actions
        ), f"Invalid action type: {action_type}, must be one of {instance_actions + propagation_actions}"
        action = {
            "type": action_type,
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
        }
        inference_state["action_history"].append(action)

    def _has_object_been_refined(self, inference_state, obj_id):
        action_history = inference_state["action_history"]
        for action in action_history:
            if action["type"] in ["add", "refine"] and action.get("obj_ids"):
                if obj_id in action["obj_ids"]:
                    return True
        return False

    def parse_action_history_for_propagation(self, inference_state):
        """
        Parse the actions in history before the last propagation and prepare for the next propagation.
        We support multiple actions (add/remove/refine) between two propagations. If we had an action
        history similar to this ["propagate", "add", "refine", "remove", "add"], the next propagation
        would remove the removed object, and also propagate the two added/refined objects.

        Returns:
            propagation_type: one of ["propagation_full", "propagation_partial", "propagation_fetch"]
                - "propagation_full": run VG propagation for all objects
                - "propagation_partial": run Tracker propagation for selected objects, useful for add/refine actions
                - "propagation_fetch": fetch existing VG predictions without running any propagation
            obj_ids: list of object ids to run Tracker propagation on if propagation_type is "propagation_partial".
        """
        action_history = inference_state["action_history"]
        if len(action_history) == 0:
            # we run propagation for the first time
            return "propagation_full", None

        if "propagation" in action_history[-1]["type"]:
            if action_history[-1]["type"] in ["propagation_fetch"]:
                # last propagation is direct fetch, we fetch existing predictions
                return "propagation_fetch", None
            elif action_history[-1]["type"] in [
                "propagation_partial",
                "propagation_full",
            ]:
                # we do fetch prediction if we have already run propagation twice or we have run
                # propagation once and it is from the first frame or last frame.
                if (
                    len(action_history) > 1
                    and action_history[-2]["type"]
                    in ["propagation_partial", "propagation_full"]
                ) or action_history[-1]["frame_idx"] in [
                    0,
                    inference_state["num_frames"] - 1,
                ]:
                    # we have run both forward and backward partial/full propagation
                    return "propagation_fetch", None
                else:
                    # we have run partial/full forward or backward propagation once, need run it for the rest of the frames
                    return action_history[-1]["type"], action_history[-1]["obj_ids"]

        # parse actions since last propagation
        obj_ids = []
        for action in action_history[::-1]:
            if "propagation" in action["type"]:
                # we reached the last propagation action, stop parsing
                break
            if action["type"] in ["add", "refine"]:
                obj_ids.extend(action["obj_ids"])
            # else action["type"] == "remove": noop
        obj_ids = list(set(obj_ids)) if len(obj_ids) > 0 else None
        propagation_type = (
            "propagation_partial" if obj_ids is not None else "propagation_fetch"
        )
        return propagation_type, obj_ids

    def remove_object(self, inference_state, obj_id, is_user_action=False):
        """
        We try to remove object from tracker states on every GPU, it will do nothing
        for states without this object.
        """
        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        assert obj_rank is not None, f"Object {obj_id} not found in any GPU."

        tracker_states_local = inference_state["tracker_inference_states"]
        if self.rank == obj_rank:
            self._tracker_remove_object(tracker_states_local, obj_id)

        if is_user_action:
            self.add_action_history(
                inference_state, action_type="remove", obj_ids=[obj_id]
            )

        # update metadata
        tracker_metadata = inference_state["tracker_metadata"]
        _obj_ids = tracker_metadata["obj_ids_per_gpu"][obj_rank]
        tracker_metadata["obj_ids_per_gpu"][obj_rank] = _obj_ids[_obj_ids != obj_id]
        tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
            tracker_metadata["obj_ids_per_gpu"][obj_rank]
        )
        tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata["obj_ids_per_gpu"]
        )
        tracker_metadata["obj_id_to_score"].pop(obj_id, None)

        # Clean up cached frame outputs to remove references to the deleted object
        if "cached_frame_outputs" in inference_state:
            for frame_idx in inference_state["cached_frame_outputs"]:
                frame_cache = inference_state["cached_frame_outputs"][frame_idx]
                if obj_id in frame_cache:
                    del frame_cache[obj_id]

    def _get_gpu_id_by_obj_id(self, inference_state, obj_id):
        """
        Locate GPU ID for a given object.
        """
        obj_ids_per_gpu = inference_state["tracker_metadata"]["obj_ids_per_gpu"]
        for rank, obj_ids in enumerate(obj_ids_per_gpu):
            if obj_id in obj_ids:
                return rank
        return None  # object not found in any GPU

    def _get_tracker_inference_states_by_obj_ids(self, inference_state, obj_ids):
        """
        Get the Tracker inference states that contain the given object ids.
        This is used to run partial Tracker propagation on a single object/bucket.
        Possibly multiple or zero states can be returned.
        """
        states = [
            state
            for state in inference_state["tracker_inference_states"]
            if set(obj_ids) & set(state["obj_ids"])
        ]
        return states

    def _prepare_backbone_feats(self, inference_state, frame_idx, reverse):
        input_batch = inference_state["input_batch"]
        feature_cache = inference_state["feature_cache"]
        num_frames = inference_state["num_frames"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        _ = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            reverse=reverse,
            allow_new_detections=True,
        )

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        boxes_xywh=None,
        box_labels=None,
        points=None,
        point_labels=None,
        obj_id=None,
        rel_coordinates=True,
    ):
        if points is not None:
            # Tracker instance prompts
            assert (
                text_str is None and boxes_xywh is None
            ), "When points are provided, text_str and boxes_xywh must be None."
            assert (
                obj_id is not None
            ), "When points are provided, obj_id must be provided."
            return self.add_tracker_new_points(
                inference_state,
                frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=self.use_prev_mem_frame,
            )
        else:
            # SAM3 prompts
            return super().add_prompt(
                inference_state,
                frame_idx,
                text_str=text_str,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
            )

    @torch.inference_mode()
    def add_tracker_new_points(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        rel_coordinates=True,
        use_prev_mem_frame=False,
    ):
        """Add a new point prompt to Tracker. Suppporting instance refinement to existing
        objects by passing existing obj_id or adding a new object by passing a new obj_id.
        use_prev_mem_frame=False to disable cross attention to previous memory frames.
        Every GPU returns the same results, and results should contain all masks including
        these masks not refined or not added by the current user points.
        """
        assert obj_id is not None, "obj_id must be provided to add new points"
        tracker_metadata = inference_state["tracker_metadata"]
        if tracker_metadata == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            tracker_metadata.update(self._initialize_metadata())

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)

        # prepare feature
        self._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

        object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
        if (
            obj_rank is not None
            and self.use_stateless_refinement
            and not object_has_been_refined
        ):
            # The first time we start refinement on the object, we remove it.
            logger.debug(
                f"[rank={self.rank}] Removing object {obj_id} before refinement."
            )
            self.remove_object(inference_state, obj_id, is_user_action=False)
            obj_rank = None

        if obj_rank is None:
            # new object, we assign it a GPU and create a new inference state if limit allows
            num_prev_obj = np.sum(tracker_metadata["num_obj_per_gpu"])
            if num_prev_obj >= self.max_num_objects:
                logger.warning(
                    f"add_tracker_new_points: cannot add a new object as we are already tracking {num_prev_obj=} "
                    f"masklets (under {self.max_num_objects=})"
                )
                obj_ids = []
                H_low_res = W_low_res = self.tracker.low_res_mask_size
                H_video_res = inference_state["orig_height"]
                W_video_res = inference_state["orig_width"]
                low_res_masks = torch.zeros(0, 1, H_low_res, W_low_res)
                video_res_masks = torch.zeros(0, 1, H_video_res, W_video_res)
                return frame_idx, obj_ids, low_res_masks, video_res_masks

            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=1,
                prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
            )
            obj_rank = new_det_gpu_ids[0]

            # get tracker inference state for the new object
            if self.rank == obj_rank:
                # for batched inference, we create a new inference state
                tracker_state = self._init_new_tracker_state(inference_state)
                inference_state["tracker_inference_states"].append(tracker_state)

            # update metadata
            tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
                [
                    tracker_metadata["obj_ids_per_gpu"][obj_rank],
                    np.array([obj_id], dtype=np.int64),
                ]
            )
            tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
                tracker_metadata["obj_ids_per_gpu"][obj_rank]
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
                tracker_metadata["obj_ids_per_gpu"]
            )
            tracker_metadata["max_obj_id"] = max(tracker_metadata["max_obj_id"], obj_id)

            logger.debug(
                f"[rank={self.rank}] Adding new object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
            )
        else:
            # existing object, for refinement
            if self.rank == obj_rank:
                tracker_states = self._get_tracker_inference_states_by_obj_ids(
                    inference_state, [obj_id]
                )
                assert (
                    len(tracker_states) == 1
                ), f"[rank={self.rank}] Multiple Tracker inference states found for the same object id."
                tracker_state = tracker_states[0]

            # log
            logger.debug(
                f"[rank={self.rank}] Refining existing object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
            )

        # assign higher score to added/refined object
        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0

        if self.rank == 0:
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})

            if "removed_obj_ids" in rank0_metadata:
                rank0_metadata["removed_obj_ids"].discard(obj_id)

            if "suppressed_obj_ids" in rank0_metadata:
                for frame_id in rank0_metadata["suppressed_obj_ids"]:
                    rank0_metadata["suppressed_obj_ids"][frame_id].discard(obj_id)

            if "masklet_confirmation" in rank0_metadata:
                obj_ids_all_gpu = tracker_metadata["obj_ids_all_gpu"]
                obj_indices = np.where(obj_ids_all_gpu == obj_id)[0]
                if len(obj_indices) > 0:
                    obj_idx = obj_indices[0]
                    if obj_idx < len(rank0_metadata["masklet_confirmation"]["status"]):
                        rank0_metadata["masklet_confirmation"]["status"][obj_idx] = 1
                        rank0_metadata["masklet_confirmation"]["consecutive_det_num"][
                            obj_idx
                        ] = self.masklet_confirmation_consecutive_det_thresh

        if self.rank == obj_rank:
            frame_idx, obj_ids, low_res_masks, video_res_masks = (
                self.tracker.add_new_points(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                    rel_coordinates=rel_coordinates,
                    use_prev_mem_frame=use_prev_mem_frame,
                )
            )

            if video_res_masks is not None and len(video_res_masks) > 0:
                video_res_masks = fill_holes_in_mask_scores(
                    video_res_masks,  # shape (N, 1, H_video, W_video)
                    max_area=self.fill_hole_area,
                    fill_holes=True,
                    remove_sprinkles=True,
                )

            # Since the mem encoder has already run for the current input points?
            self.tracker.propagate_in_video_preflight(
                tracker_state, run_mem_encoder=True
            )
            # Clear detector conditioning frames when user clicks are received to allow
            # model updating masks on these frames. It is a noop if user is refining on the
            # detector conditioning frames or adding new objects.
            self.clear_detector_added_cond_frame_in_tracker(
                tracker_state, obj_id, frame_idx
            )

        # fetch results from states and gather across GPUs
        # Use optimized caching approach to avoid reprocessing unmodified objects
        if self.rank == obj_rank and len(obj_ids) > 0:
            new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(
                torch.bool
            )
        else:
            new_mask_data = None
        # Broadcast the new mask data across all ranks for consistency
        if self.world_size > 1:
            data_list = [new_mask_data.cpu() if new_mask_data is not None else None]
            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
            new_mask_data = data_list[0].to(self.device)

        if self.rank == 0:
            # Initialize cache if not present (needed for point prompts without prior propagation)
            if "cached_frame_outputs" not in inference_state:
                inference_state["cached_frame_outputs"] = {}
            if frame_idx not in inference_state["cached_frame_outputs"]:
                inference_state["cached_frame_outputs"][frame_idx] = {}

            obj_id_to_mask = self._build_tracker_output(
                inference_state,
                frame_idx,
                {obj_id: new_mask_data} if new_mask_data is not None else None,
            )
            # post processing - remove suppressed obj_ids
            obj_id_to_score = tracker_metadata["obj_id_to_score"]
            suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                "suppressed_obj_ids"
            ][frame_idx]
            obj_id_to_tracker_score = tracker_metadata[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx]

            out = {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": obj_id_to_score,
                "obj_id_to_tracker_score": obj_id_to_tracker_score,
            }
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            return frame_idx, self._postprocess_output(
                inference_state, out, suppressed_obj_ids=suppressed_obj_ids
            )
        else:
            return frame_idx, None  # no output on other GPUs

    def _gather_obj_id_to_mask_across_gpus(self, inference_state, obj_id_to_mask_local):
        """Gather obj_id_to_mask from all GPUs. Optionally resize the masks to the video resolution."""
        tracker_metadata = inference_state["tracker_metadata"]

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.tracker.low_res_mask_size
        obj_ids_local = tracker_metadata["obj_ids_per_gpu"][self.rank]
        low_res_masks_local = []
        for obj_id in obj_ids_local:
            if obj_id in obj_id_to_mask_local:
                low_res_masks_local.append(obj_id_to_mask_local[obj_id])
            else:
                low_res_masks_local.append(
                    torch.full((H_mask, W_mask), -1024.0, device=self.device)
                )
        if len(low_res_masks_local) > 0:
            low_res_masks_local = torch.stack(low_res_masks_local, dim=0)  # (N, H, W)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)

        # all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        if self.world_size > 1:
            low_res_masks_local = low_res_masks_local.float().contiguous()
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
        return low_res_masks_global

    def _convert_low_res_mask_to_video_res(self, low_res_mask, inference_state):
        """
        Convert a low-res mask to video resolution, matching the format expected by _build_tracker_output.

        Args:
            low_res_mask: Tensor of shape (H_low_res, W_low_res)
            inference_state: Contains video dimensions

        Returns:
            video_res_mask: Tensor of shape (1, H_video, W_video) bool
        """
        if low_res_mask is None:
            return None

        # Convert to 3D for interpolation: (H_low_res, W_low_res) -> (1, H_low_res, W_low_res)
        low_res_mask_3d = low_res_mask.unsqueeze(0).unsqueeze(0)

        # Get video dimensions
        H_video = inference_state["orig_height"]
        W_video = inference_state["orig_width"]

        video_res_mask = F.interpolate(
            low_res_mask_3d.float(),
            size=(H_video, W_video),
            mode="bilinear",
            align_corners=False,
        )  # (1, H_video, W_video)

        # Convert to boolean - already in the right shape!
        return (video_res_mask.squeeze(0) > 0.0).to(torch.bool)

    def clear_detector_added_cond_frame_in_tracker(
        self, tracker_state, obj_id, refined_frame_idx
    ):
        """Clear detector added conditioning frame if it is within a predefined window
        of the refined frame. This allow model to update masks on these frames."""
        obj_idx = self.tracker._obj_id_to_idx(tracker_state, obj_id)

        mask_only_cond_frame_indices = []
        window = self.refinement_detector_cond_frame_removal_window
        for frame_idx in tracker_state["mask_inputs_per_obj"][obj_idx]:
            if frame_idx not in tracker_state["point_inputs_per_obj"][obj_idx]:
                # clear conditioning frames within a window of the refined frame
                if abs(frame_idx - refined_frame_idx) <= window:
                    mask_only_cond_frame_indices.append(frame_idx)

        # clear
        if len(mask_only_cond_frame_indices) > 0:
            for frame_idx in mask_only_cond_frame_indices:
                # obj_ids_on_this_frame is essentially all obj_ids in the state
                # since they are bucket batched
                obj_ids_on_this_frame = tracker_state["obj_id_to_idx"].keys()
                for obj_id2 in obj_ids_on_this_frame:
                    self.tracker.clear_all_points_in_frame(
                        tracker_state, frame_idx, obj_id2, need_output=False
                    )
            logger.debug(
                f"Cleared detector mask only conditioning frames ({mask_only_cond_frame_indices}) in Tracker."
            )
        return


# ---------------------------------------------------------------------------
# SAM3InteractiveImagePredictor (from model/sam1_task_predictor.py)
# ---------------------------------------------------------------------------

class SAM3InteractiveImagePredictor(nn.Module):
    def __init__(
        self,
        sam_model: Sam3TrackerBase,
        mask_threshold=0.0,
        max_hole_area=256.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-3 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (288, 288),
            (144, 144),
            (72, 72),
        ]

    @torch.no_grad()
    def set_image(self, image):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.
        """
        from PIL.Image import Image

        self.reset_predictor()
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(self, image_list):
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape HWC"
            self._orig_hw.append(image.shape[:2])
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch=None,
        point_labels_batch=None,
        box_batch=None,
        mask_input_batch=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    ):
        """Batched prediction mode."""
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            # Preserve dtype if mask_logits is already a tensor (e.g. bf16
            # low_res_masks fed back from a previous predict call).  Forcing
            # dtype=float32 here causes dense_prompt_embeddings to be fp32,
            # which promotes src to fp32 and crashes the matmul against bf16
            # hyper_in in predict_masks.
            if isinstance(mask_logits, torch.Tensor):
                mask_input = mask_logits.to(device=self.device)
            else:
                mask_input = torch.as_tensor(
                    mask_logits, dtype=torch.float, device=self.device
                )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords,
        point_labels,
        boxes=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        img_idx=-1,
    ):
        """Predict masks using currently set image features."""
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points_val = (point_coords, point_labels)
        else:
            concat_points_val = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            if concat_points_val is not None:
                concat_coords = torch.cat([box_coords, concat_points_val[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points_val[1]], dim=1)
                concat_points_val = (concat_coords, concat_labels)
            else:
                concat_points_val = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points_val,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points_val is not None and concat_points_val[0].shape[0] > 1
        )
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self):
        """Returns the image embeddings for the currently set image."""
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self):
        return self.model.device

    def reset_predictor(self):
        """Resets the image embeddings and other state variables."""
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
