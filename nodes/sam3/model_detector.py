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

from collections import OrderedDict
from copy import deepcopy

import numpy as np
from torch import Tensor
from torchvision.ops import masks_to_boxes

from .model_components import (
    _dtype_debug,
    _DeviceCacheMixin,
    ops,
)
from .model_backbone import SAM3VLBackbone
from .utils import (
    box_cxcywh_to_xyxy,
    box_xywh_to_cxcywh,
    box_xyxy_to_xywh,
    inverse_sigmoid,
    get_valid_ratio,
    SAM3Output,
    Prompt,
    concat_padded_sequences,
    BatchedDatapoint,
    FindStage,
    convert_my_tensors,
)

log = logging.getLogger("sam3")

def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value

class Sam3Image(_DeviceCacheMixin, torch.nn.Module):
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
        vision_feats = self._get_inst_interactive_vision_feats(inference_state)

        vision_feats[-1] = (
            vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
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
        vision_feats = self._get_inst_interactive_vision_feats(inference_state)
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

    def _get_inst_interactive_vision_feats(self, inference_state):
        backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.inst_interactive_predictor.model._prepare_backbone_features(
            backbone_out
        )
        return vision_feats


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

