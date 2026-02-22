# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# ComfyUI-native attention, RoPE, and SAM attention classes.
# Consolidated from: sam/rope.py, sam/transformer.py, sam/common.py

import logging
import math
from functools import partial
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention, get_attention_function

log = logging.getLogger("sam3")

ops = comfy.ops.manual_cast


# ---------------------------------------------------------------------------
# ComfyUI attention dispatch — per-model backend override
# ---------------------------------------------------------------------------

_sam3_attn_fn = None  # None = not yet configured, falls back to optimized_attention
_sam3_attn_printed = False
_sam3_target_dtype = None  # None = no forced dtype; set to torch.bfloat16/float16 to normalize

_BACKEND_MAP = {
    "flash_attn": "flash",
    "sage": "sage",
    "sage3": "sage3",
    "xformers": "xformers",
    "sdpa": "pytorch",
    "sub_quad": "sub_quad",
    "split": "split",
}


def set_sam3_backend(name: str = "auto"):
    """Configure SAM3's attention backend.

    ``auto`` uses the global ``optimized_attention``, but if SageAttention is
    active globally, falls back to flash_attn — SAM3 is incompatible with sage
    due to relative position bias concatenation modifying Q/K dimensions.
    Explicit user choices are respected (including ``sage`` at user's risk).
    """
    global _sam3_attn_fn, _sam3_attn_printed
    pytorch = get_attention_function("pytorch")
    if name == "auto":
        if optimized_attention.__name__ in ("attention_sage", "attention3_sage"):
            _sam3_attn_fn = get_attention_function("flash", default=pytorch)
            log.warning(
                "SageAttention is not compatible with SAM3 "
                "(relative position bias modifies Q/K dimensions). "
                "Using %s instead.", _sam3_attn_fn.__name__
            )
        else:
            _sam3_attn_fn = optimized_attention
    else:
        reg_name = _BACKEND_MAP.get(name, "pytorch")
        _sam3_attn_fn = get_attention_function(reg_name, default=pytorch)
    _sam3_attn_printed = False


def set_sam3_dtype(dtype):
    """Set the model's target dtype for attention normalization.

    When set to bfloat16 or float16, sam3_attention() will cast all Q/K/V
    tensors to this dtype before the attention call. This covers cases where
    fp32 activations arise from type promotion (e.g. bf16_tensor + fp32_pe),
    ensuring flash attention always receives half-precision inputs.
    """
    global _sam3_target_dtype, _sam3_attn_printed
    _sam3_target_dtype = dtype
    _sam3_attn_printed = False  # reset so the new dtype shows in the log


def sam3_attention(q, k, v, num_heads):
    """ComfyUI-native attention dispatch.

    Drop-in replacement for comfy_attn.dispatch_attention.
    Expects q, k, v in shape [B, H, L, D]. Returns [B, H, L, D].

    Normalizes Q/K/V to ``_sam3_target_dtype`` (set by set_sam3_dtype) when
    the model is configured for half precision.  This covers all sources of
    fp32 activations -- positional-encoding additions that promote bf16->fp32,
    self-attention on fp32 token embeddings from the prompt encoder, etc. —
    so flash attention always receives the expected dtype regardless of call
    site.
    """
    global _sam3_attn_printed
    fn = _sam3_attn_fn if _sam3_attn_fn is not None else optimized_attention

    # Normalize dtype centrally:
    # 1. If _sam3_target_dtype is set (half precision model), cast everything.
    # 2. Otherwise fall back to the per-call mixed-dtype check so we at least
    #    avoid the crash from mismatched Q/K/V dtypes.
    target = _sam3_target_dtype
    if target is not None and target in (torch.bfloat16, torch.float16):
        if q.dtype != target:
            q = q.to(target)
        if k.dtype != target:
            k = k.to(target)
        if v.dtype != target:
            v = v.to(target)
    elif not (q.dtype == k.dtype == v.dtype):
        all_dtypes = (q.dtype, k.dtype, v.dtype)
        if torch.bfloat16 in all_dtypes:
            target = torch.bfloat16
        elif torch.float16 in all_dtypes:
            target = torch.float16
        else:
            target = q.dtype
        q, k, v = q.to(target), k.to(target), v.to(target)

    if not _sam3_attn_printed:
        log.info("attention backend: %s | dtype: %s", fn.__name__, q.dtype)
        _sam3_attn_printed = True
    return fn(q, k, v, heads=num_heads, skip_reshape=True, skip_output_reshape=True)


# ---------------------------------------------------------------------------
# SplitMultiheadAttention — drop-in replacement for nn.MultiheadAttention
# ---------------------------------------------------------------------------

class SplitMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention using separate Linear projections.

    Uses operations.Linear for ComfyUI weight management and
    optimized_attention for computation (supports masks).
    Returns (output, None) to match nn.MultiheadAttention signature.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        batch_first=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.to_q = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        # Named out_proj to match nn.MultiheadAttention key names in state dicts
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                need_weights=False, attn_mask=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        # Normalize dtypes: prefer half precision so flash attention can run.
        # Cross-attention between fp32 point embeddings and bf16 image features
        # would otherwise promote everything to fp32 (query-dominant).
        if not (query.dtype == key.dtype == value.dtype):
            all_dtypes = (query.dtype, key.dtype, value.dtype)
            if torch.bfloat16 in all_dtypes:
                target = torch.bfloat16
            elif torch.float16 in all_dtypes:
                target = torch.float16
            else:
                target = query.dtype
            query = query.to(target)
            key = key.to(target)
            value = value.to(target)

        if not self.batch_first:
            query = query.transpose(0, 1)   # [L, B, D] -> [B, L, D]
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, L_q, _ = query.shape
        L_k = key.shape[1]

        q = self.to_q(query).reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(key).reshape(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(value).reshape(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare mask for attention
        sdpa_mask = self._prepare_mask(attn_mask, key_padding_mask, B, L_q, L_k, q.dtype, q.device)

        # When mask is needed, use SDPA (pytorch) which supports masks natively.
        # Flash/sage/xformers don't support arbitrary attention masks.
        if sdpa_mask is not None:
            pytorch_attn = get_attention_function("pytorch")
            out = pytorch_attn(q, k, v, heads=self.num_heads, mask=sdpa_mask, skip_reshape=True)
        else:
            fn = _sam3_attn_fn if _sam3_attn_fn is not None else optimized_attention
            out = fn(q, k, v, heads=self.num_heads, skip_reshape=True)

        out = out.reshape(B, L_q, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)   # [B, L, D] -> [L, B, D]

        return out, None

    def _prepare_mask(self, attn_mask, key_padding_mask, B, L_q, L_k, dtype, device):
        if attn_mask is None and key_padding_mask is None:
            return None

        mask = torch.zeros(B, self.num_heads, L_q, L_k, dtype=dtype, device=device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=dtype)
                am.masked_fill_(attn_mask, float("-inf"))
                attn_mask = am
            if attn_mask.dim() == 2:
                # [L_q, L_k] — broadcast over B and H
                mask = mask + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B*H, L_q, L_k] — reshape to [B, H, L_q, L_k]
                mask = mask + attn_mask.view(B, self.num_heads, L_q, L_k)
            else:
                mask = mask + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: [B, L_k], True = padded
            pad = torch.zeros(B, 1, 1, L_k, dtype=dtype, device=device)
            pad.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            mask = mask + pad

        return mask


# ---------------------------------------------------------------------------
# RoPE functions (from sam/rope.py)
# ---------------------------------------------------------------------------

def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device=None,
):
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def complex_mult(xq_real, xq_imag, freqs_cis_real, freqs_cis_imag):
    real_part = xq_real * freqs_cis_real - xq_imag * freqs_cis_imag
    imag_part = xq_real * freqs_cis_imag + xq_imag * freqs_cis_real
    return torch.stack([real_part, imag_part], dim=-1)


def apply_rotary_enc_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_real: torch.Tensor,
    freqs_cis_imag: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    assert xk is not None
    assert xk.shape[-2] != 0

    xq_real = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 0]
    xq_imag = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 1]
    xk_real = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 0]
    xk_imag = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 1]
    freqs_cis_real = reshape_for_broadcast(freqs_cis_real, xq_real)
    freqs_cis_imag = reshape_for_broadcast(freqs_cis_imag, xq_imag)
    xq_out = complex_mult(xq_real, xq_imag, freqs_cis_real, freqs_cis_imag).flatten(3)
    if repeat_freqs_k:
        r = xk_real.shape[-2] // xq_real.shape[-2]
        freqs_cis_real = freqs_cis_real.repeat(*([1] * (freqs_cis_real.ndim - 2)), r, 1)
        freqs_cis_imag = freqs_cis_imag.repeat(*([1] * (freqs_cis_imag.ndim - 2)), r, 1)
    xk_out = complex_mult(xk_real, xk_imag, freqs_cis_real, freqs_cis_imag).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# ---------------------------------------------------------------------------
# MLPBlock (from sam/common.py)
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.lin1 = operations.Linear(embedding_dim, mlp_dim, dtype=dtype, device=device)
        self.lin2 = operations.Linear(mlp_dim, embedding_dim, dtype=dtype, device=device)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# ---------------------------------------------------------------------------
# LayerNorm2d (from sam/common.py) — used by memory/mask modules
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# SAM Attention classes (from sam/transformer.py)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with optional downscaling, using ComfyUI dispatch."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = operations.Linear(embedding_dim, self.internal_dim, dtype=dtype, device=device)
        self.k_proj = operations.Linear(self.kv_in_dim, self.internal_dim, dtype=dtype, device=device)
        self.v_proj = operations.Linear(self.kv_in_dim, self.internal_dim, dtype=dtype, device=device)
        self.out_proj = operations.Linear(self.internal_dim, embedding_dim, dtype=dtype, device=device)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Normalize dtype: cross-attention between fp32 point embeddings and
        # bf16 image features produces mixed Q/K/V dtypes (e.g. K=fp32 from
        # keys+key_pe type promotion, V=bf16 from direct keys reference).
        # Flash attention requires all three to match — prefer half precision.
        if not (q.dtype == k.dtype == v.dtype):
            dtypes = (q.dtype, k.dtype, v.dtype)
            if torch.bfloat16 in dtypes:
                target = torch.bfloat16
            elif torch.float16 in dtypes:
                target = torch.float16
            else:
                target = q.dtype
            q, k, v = q.to(target), k.to(target), v.to(target)

        # ComfyUI optimized attention — q, k, v are [B, H, L, D]
        out = sam3_attention(q, k, v, self.num_heads)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        use_rope_real=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_rope_real = use_rope_real
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        self._feat_sizes = feat_sizes
        # Lazily computed on first forward using input tensor device
        self.freqs_cis = None
        self.freqs_cis_real = None
        self.freqs_cis_imag = None
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        if self.freqs_cis is None or self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h, device=q.device)
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        if self.use_rope_real:
            q, k[:, :, :num_k_rope] = apply_rotary_enc_real(
                q,
                k[:, :, :num_k_rope],
                freqs_cis_real=self.freqs_cis_real,
                freqs_cis_imag=self.freqs_cis_imag,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            q, k[:, :, :num_k_rope] = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        # ComfyUI optimized attention
        out = sam3_attention(q, k, v, self.num_heads)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------------
# TwoWayAttentionBlock and TwoWayTransformer (from sam/transformer.py)
# ---------------------------------------------------------------------------

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.norm1 = operations.LayerNorm(embedding_dim, dtype=dtype, device=device)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
            dtype=dtype, device=device, operations=operations,
        )
        self.norm2 = operations.LayerNorm(embedding_dim, dtype=dtype, device=device)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation, dtype=dtype, device=device, operations=operations)
        self.norm3 = operations.LayerNorm(embedding_dim, dtype=dtype, device=device)

        self.norm4 = operations.LayerNorm(embedding_dim, dtype=dtype, device=device)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
            dtype=dtype, device=device, operations=operations,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
            dtype=dtype, device=device, operations=operations,
        )
        self.norm_final_attn = operations.LayerNorm(embedding_dim, dtype=dtype, device=device)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
