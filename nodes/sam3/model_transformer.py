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

from .model_components import (
    _dtype_debug,
    MLP,
    ops,
)
from .attention import SplitMultiheadAttention

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

