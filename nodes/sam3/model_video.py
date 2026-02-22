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

import datetime
from collections import OrderedDict, defaultdict
from copy import copy, deepcopy
from enum import Enum

import numpy as np
import torch.distributed as dist

from .model_components import (
    _DeviceCacheMixin,
    ops,
)
from .model_tracker import NO_OBJ_SCORE, Sam3TrackerBase
from .utils import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_xywh,
    fast_diag_box_iou,
    SAM3Output,
    Prompt,
    BatchedDatapoint,
    convert_my_tensors,
    copy_data_to_device,
    load_resource_as_video_frames,
    IMAGE_EXTS,
    rle_encode,
    load_video_frames,
)
from . import perflib
from .perflib import mask_iou
from .perflib import masks_to_boxes as perf_masks_to_boxes
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1
    CONFIRMED = 2


class Sam3VideoBase(_DeviceCacheMixin, nn.Module):
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

    def _build_tracker_frame_output(self, inference_state, frame_idx, obj_id_to_mask):
        tracker_metadata = inference_state["tracker_metadata"]
        obj_id_to_score = tracker_metadata["obj_id_to_score"]
        suppressed_obj_ids = tracker_metadata["rank0_metadata"]["suppressed_obj_ids"][
            frame_idx
        ]
        obj_id_to_tracker_score = tracker_metadata[
            "obj_id_to_tracker_score_frame_wise"
        ][frame_idx]

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_tracker_score": obj_id_to_tracker_score,
        }
        return out, suppressed_obj_ids

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
                    out, suppressed_obj_ids = self._build_tracker_frame_output(
                        inference_state, frame_idx, obj_id_to_mask
                    )
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
            out, suppressed_obj_ids = self._build_tracker_frame_output(
                inference_state, frame_idx, obj_id_to_mask
            )
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

