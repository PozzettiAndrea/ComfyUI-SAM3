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
    from timm.layers import trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import trunc_normal_

from .model_components import (
    _dtype_debug,
    _DeviceCacheMixin,
    VitMlp,
    PatchEmbed,
    concat_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    ops,
)
from .attention import sam3_attention

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

