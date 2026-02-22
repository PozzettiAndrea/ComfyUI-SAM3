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

import numpy as np

from torch import Tensor
from torchvision.ops.roi_align import RoIAlign

from .model_components import (
    _dtype_debug,
    _DeviceCacheMixin,
    _build_linear_stack,
    MLP,
    SamMLP,
    PositionEmbeddingRandom,
    ops,
)
from .attention import LayerNorm2d
from .utils import (
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    gen_sineembed_for_position,
)

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
        src = src + dense_prompt_embeddings
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
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

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

class SegmentationHead(_DeviceCacheMixin, nn.Module):
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

