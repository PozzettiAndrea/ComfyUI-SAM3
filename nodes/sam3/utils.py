# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Consolidated utility module — pure functions, data classes, and non-model helpers.
# Merged from: model/box_ops.py, model/act_ckpt_utils.py, model/data_misc.py,
#   model/masks_ops.py, model/edt.py, model/io_utils.py, model/sam3_image_processor.py,
#   model/sam3_tracker_utils.py, model/utils/sam2_utils.py, model/utils/misc.py,
#   model/utils/sam1_utils.py, model/model_misc.py (non-nn.Module parts)

import contextlib
import copy
import inspect
import logging
import math
import os
import queue
import re
import time
import warnings
import weakref
from collections import defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from functools import wraps
from threading import Condition, get_ident, Lock, Thread
from typing import (
    Any, Callable, Dict, get_args, get_origin, List,
    Mapping, Optional, Protocol, Sequence, Tuple, TypeVar, Union,
    runtime_checkable,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils._pytree import tree_map_only
from typing_extensions import override

log = logging.getLogger("sam3")


# ===========================================================================
# Box operations (from model/box_ops.py)
# ===========================================================================

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x_, y, w, h = x.unbind(-1)
    b = [(x_), (y), (x_ + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    x_, y, w, h = x.unbind(-1)
    b = [(x_ + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x_, y, X, Y = x.unbind(-1)
    b = [(x_), (y), (X - x_), (Y - y)]
    return torch.stack(b, dim=-1)


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)
    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0] + 1
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0] + 1
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    boxes = torch.stack([x_min, y_min, x_max, y_max], 1)
    boxes = boxes * masks.flatten(-2).any(-1)
    return boxes


@torch.jit.script
def fast_diag_generalized_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    area1 = (box1_xy - box1_XY).prod(-1)
    area2 = (box2_xy - box2_XY).prod(-1)
    lt = torch.max(box1_XY, box2_XY)
    lt2 = torch.min(box1_XY, box2_XY)
    rb = torch.min(box1_xy, box2_xy)
    rb2 = torch.max(box1_xy, box2_xy)
    inter = (rb - lt).clamp(min=0).prod(-1)
    tot_area = (rb2 - lt2).clamp(min=0).prod(-1)
    union = area1 + area2 - inter
    iou = inter / union
    return iou - (tot_area - union) / tot_area


@torch.jit.script
def fast_diag_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    area1 = (box1_xy - box1_XY).prod(-1)
    area2 = (box2_xy - box2_XY).prod(-1)
    lt = torch.max(box1_XY, box2_XY)
    rb = torch.min(box1_xy, box2_xy)
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area1 + area2 - inter
    iou = inter / union
    return iou


def box_xywh_inter_union(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert boxes1.size(-1) == 4 and boxes2.size(-1) == 4
    boxes1 = box_xywh_to_xyxy(boxes1)
    boxes2 = box_xywh_to_xyxy(boxes2)
    box1_tl_xy = boxes1[..., :2]
    box1_br_xy = boxes1[..., 2:]
    box2_tl_xy = boxes2[..., :2]
    box2_br_xy = boxes2[..., 2:]
    area1 = (box1_br_xy - box1_tl_xy).prod(-1)
    area2 = (box2_br_xy - box2_tl_xy).prod(-1)
    assert (area1 >= 0).all() and (area2 >= 0).all()
    tl = torch.max(box1_tl_xy, box2_tl_xy)
    br = torch.min(box1_br_xy, box2_br_xy)
    inter = (br - tl).clamp(min=0).prod(-1)
    union = area1 + area2 - inter
    return inter, union


# ===========================================================================
# Activation checkpoint utils (from model/act_ckpt_utils.py)
# ===========================================================================

T = TypeVar("T")
Module = TypeVar("Module", bound=nn.Module)


def activation_ckpt_wrapper(module: Union[nn.Module, Callable]) -> Callable:
    @wraps(module)
    def act_ckpt_wrapper(
        *args, act_ckpt_enable: bool = True, use_reentrant: bool = False, **kwargs
    ):
        if act_ckpt_enable:
            if len(args) > 0:
                raise ValueError(
                    "This wrapper expects keyword arguments only when `act_ckpt_enable=True`"
                )
            callable_fn = module.forward if isinstance(module, nn.Module) else module
            sig = inspect.signature(callable_fn)
            param_defaults = {
                name: param.default for name, param in sig.parameters.items()
            }
            args = []
            for p_name in param_defaults.keys():
                if p_name in kwargs:
                    args.append(kwargs.pop(p_name))
                elif param_defaults[p_name] is not inspect.Parameter.empty:
                    args.append(param_defaults[p_name])
                elif (
                    sig.parameters[p_name].kind is not inspect.Parameter.VAR_KEYWORD
                ):
                    raise ValueError(f"Missing positional argument: {p_name}")

            remaining_keys = list(kwargs.keys())
            for key in remaining_keys:
                if isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = "_REMOVED_BY_ACT_CKPT_WRAPPER_"

            ret = torch.utils.checkpoint.checkpoint(
                module, *args, use_reentrant=use_reentrant, **kwargs
            )
        else:
            ret = module(*args, **kwargs)
        return ret

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )
    return wrapped


# ===========================================================================
# Model miscellaneous utilities (from model/model_misc.py — non-nn.Module parts)
# ===========================================================================

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_clones_seq(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_module(activation):
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


# ===========================================================================
# SAM3Output (from model/model_misc.py)
# ===========================================================================

class SAM3Output(list):
    class IterMode(Enum):
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()

    def __init__(
        self,
        output: List[List[Dict]] = None,
        iter_mode: "SAM3Output.IterMode" = None,
        loss_stages: Optional[List[int]] = None,
    ):
        if iter_mode is None:
            iter_mode = SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        if output is not None:
            assert (
                isinstance(output, list)
                and len(output) > 0
                and isinstance(output[0], list)
            ), "Expected output to be a list of lists"
            self.output = output
        else:
            self.output = []
        assert isinstance(
            iter_mode, SAM3Output.IterMode
        ), f"iter_mode should be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"

        self.iter_mode = iter_mode
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (
                inner_list[-1] for inner_list in self_ref().output
            ),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }
        self.loss_stages = loss_stages

    @override
    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

    def __getitem__(self, index):
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        elif self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            if index == -1:
                return self.output[-1][-1]
            else:
                flattened_output = sum(self.output, [])
                return flattened_output[index]

    class _IterationMode(AbstractContextManager):
        def __init__(
            self, model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"
        ):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        @override
        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        @override
        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(
        model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"
    ) -> "_IterationMode":
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(item, list), f"Only list items are supported. Got {type(item)}"
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in [
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ]:
            return len(self.output)
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            flattened_output = sum(self.output, [])
            return len(flattened_output)


# ===========================================================================
# Data misc — dataclasses and helpers (from model/data_misc.py)
# ===========================================================================

MyTensor = Union[torch.Tensor, List[Any]]


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )
    assert (
        input.shape[0] != 0 or input.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"
    if input.shape[1] == 0:
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = torch.long
    text_ids: MyTensor
    text_ids__type = torch.long
    input_boxes: MyTensor
    input_boxes__type = torch.float
    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool
    input_boxes_label: MyTensor
    input_boxes_label__type = torch.long
    input_points: MyTensor
    input_points__type = torch.float
    input_points_mask: MyTensor
    input_points_mask__type = torch.bool
    object_ids: Optional[List[List]] = None


@dataclass
class BatchedFindTarget:
    num_boxes: MyTensor
    num_boxes__type = torch.long
    boxes: MyTensor
    boxes__type = torch.float
    boxes_padded: MyTensor
    boxes_padded__type = torch.float
    repeated_boxes: MyTensor
    repeated_boxes__type = torch.float
    segments: Optional[MyTensor]
    segments__type = torch.bool
    semantic_segments: Optional[MyTensor]
    semantic_segments__type = torch.bool
    is_valid_segment: Optional[MyTensor]
    is_valid_segment__type = torch.bool
    is_exhaustive: MyTensor
    is_exhaustive__type = torch.bool
    object_ids: MyTensor
    object_ids__type = torch.long
    object_ids_padded: MyTensor
    object_ids_padded__type = torch.long


@dataclass
class BatchedInferenceMetadata:
    coco_image_id: MyTensor
    coco_image_id__type = torch.long
    original_image_id: MyTensor
    original_image_id__type = torch.long
    original_category_id: MyTensor
    original_category_id__type = torch.int
    original_size: MyTensor
    original_size__type = torch.long
    object_id: MyTensor
    object_id__type = torch.long
    frame_index: MyTensor
    frame_index__type = torch.long
    is_conditioning_only: List[Optional[bool]]


@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]
    raw_images: Optional[List[Any]] = None


def convert_my_tensors(obj):
    def is_optional_field(field) -> bool:
        return get_origin(field) is Union and type(None) in get_args(field)

    for field in fields(obj):
        if is_dataclass(getattr(obj, field.name)):
            convert_my_tensors(getattr(obj, field.name))
            continue
        field_type = field.type
        if is_optional_field(field.type):
            field_type = Union[get_args(field.type)[:-1]]
        if field_type != MyTensor or getattr(obj, field.name) is None:
            continue
        elif len(getattr(obj, field.name)) and isinstance(
            getattr(obj, field.name)[0], torch.Tensor
        ):
            stack_dim = 0
            if field.name in ["input_boxes", "input_boxes_label"]:
                stack_dim = 1
            setattr(
                obj,
                field.name,
                torch.stack(getattr(obj, field.name), dim=stack_dim).to(
                    getattr(obj, field.name + "__type")
                ),
            )
        else:
            setattr(
                obj,
                field.name,
                torch.as_tensor(
                    getattr(obj, field.name), dtype=getattr(obj, field.name + "__type")
                ),
            )
    return obj


# ===========================================================================
# Mask operations (from model/masks_ops.py)
# ===========================================================================

def instance_masks_to_semantic_masks(
    instance_masks: torch.Tensor, num_instances: torch.Tensor
) -> torch.Tensor:
    masks_per_query = torch.split(instance_masks, num_instances.tolist())
    return torch.stack([torch.any(masks, dim=0) for masks in masks_per_query], dim=0)


def mask_intersection(masks1, masks2, block_size=16):
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool
    result = torch.zeros(
        masks1.shape[0], masks2.shape[0], device=masks1.device, dtype=torch.long
    )
    for i in range(0, masks1.shape[0], block_size):
        for j in range(0, masks2.shape[0], block_size):
            intersection = (
                (masks1[i : i + block_size, None] * masks2[None, j : j + block_size])
                .flatten(-2)
                .sum(-1)
            )
            result[i : i + block_size, j : j + block_size] = intersection
    return result


def mask_iom(masks1, masks2):
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool
    intersection = mask_intersection(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)
    area2 = masks2.flatten(-2).sum(-1)
    min_area = torch.min(area1[:, None], area2[None, :])
    return intersection / (min_area + 1e-8)


def compute_boundary(seg):
    assert seg.ndim >= 2
    e = torch.zeros_like(seg)
    s = torch.zeros_like(seg)
    se = torch.zeros_like(seg)
    e[..., :, :-1] = seg[..., :, 1:]
    s[..., :-1, :] = seg[..., 1:, :]
    se[..., :-1, :-1] = seg[..., 1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[..., -1, :] = seg[..., -1, :] ^ e[..., -1, :]
    b[..., :, -1] = seg[..., :, -1] ^ s[..., :, -1]
    b[..., -1, -1] = 0
    return b


@torch.no_grad()
def rle_encode(orig_mask, return_areas=False):
    from pycocotools import mask as mask_util
    assert orig_mask.ndim == 3, "Mask must be of shape (N, H, W)"
    assert orig_mask.dtype == torch.bool, "Mask must have dtype=torch.bool"
    if orig_mask.numel() == 0:
        return []
    mask = orig_mask.transpose(1, 2)
    flat_mask = mask.reshape(mask.shape[0], -1)
    if return_areas:
        mask_areas = flat_mask.sum(-1).tolist()
    differences = torch.ones(
        mask.shape[0], flat_mask.shape[1] + 1, device=mask.device, dtype=torch.bool
    )
    differences[:, 1:-1] = flat_mask[:, :-1] != flat_mask[:, 1:]
    differences[:, 0] = flat_mask[:, 0]
    _, change_indices = torch.where(differences)
    try:
        boundaries = torch.cumsum(differences.sum(-1), 0).cpu()
    except RuntimeError:
        boundaries = torch.cumsum(differences.cpu().sum(-1), 0)
    change_indices_clone = change_indices.clone()
    for i in range(mask.shape[0]):
        beg = 0 if i == 0 else boundaries[i - 1].item()
        end = boundaries[i].item()
        change_indices[beg + 1 : end] -= change_indices_clone[beg : end - 1]
    change_indices = change_indices.tolist()
    batch_rles = []
    for i in range(mask.shape[0]):
        beg = 0 if i == 0 else boundaries[i - 1].item()
        end = boundaries[i].item()
        run_lengths = change_indices[beg:end]
        uncompressed_rle = {"counts": run_lengths, "size": list(orig_mask.shape[1:])}
        h, w = uncompressed_rle["size"]
        rle = mask_util.frPyObjects(uncompressed_rle, h, w)
        rle["counts"] = rle["counts"].decode("utf-8")
        if return_areas:
            rle["area"] = mask_areas[i]
        batch_rles.append(rle)
    return batch_rles


# ===========================================================================
# Copy data to device (from model/utils/misc.py)
# ===========================================================================

def _is_named_tuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


@runtime_checkable
class _CopyableData(Protocol):
    def to(self, device: torch.device, *args: Any, **kwargs: Any):
        ...


def copy_data_to_device(data, device: torch.device, *args: Any, **kwargs: Any):
    if _is_named_tuple(data):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {k: copy_data_to_device(v, device, *args, **kwargs) for k, v in data.items()},
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {k: copy_data_to_device(v, device, *args, **kwargs) for k, v in data.items()}
        )
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data


# ===========================================================================
# I/O utilities — video/image loading (from model/io_utils.py)
# ===========================================================================

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """A command line formatter with different colors for each level."""

    def __init__(self):
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",
            logging.INFO: f"{reset}\033[32m",
            logging.WARNING: f"{reset}\033[33m",
            logging.ERROR: f"{reset}\033[31m",
            logging.CRITICAL: f"{reset}\033[35m",
        }
        fmt_str = "{color}%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)4d:{reset} %(message)s"
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset))
            for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)


def get_logger(name, level=logging.INFO):
    """A command line logger."""
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        assert level in _LOG_LEVELS, f"Invalid LOG_LEVEL: {level}, must be one of {list(_LOG_LEVELS.keys())}"
        level = _LOG_LEVELS[level]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)
    return logger
from tqdm import tqdm
import comfy.model_management
import torchvision.transforms.functional as TF
from PIL import Image

logger = get_logger(__name__)

IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
RANK = int(os.getenv("RANK", "0"))

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def _get_float_dtype(device):
    if device.type == 'cpu':
        return torch.float32
    return torch.float16


def load_resource_as_video_frames(
    resource_path,
    image_size,
    offload_video_to_cpu,
    device=None,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    video_loader_type="cv2",
):
    if device is None:
        device = comfy.model_management.get_torch_device()
    float_dtype = _get_float_dtype(device)
    if isinstance(resource_path, list):
        img_mean = torch.tensor(img_mean, dtype=float_dtype)[:, None, None]
        img_std = torch.tensor(img_std, dtype=float_dtype)[:, None, None]
        assert all(isinstance(img_pil, Image.Image) for img_pil in resource_path)
        assert len(resource_path) is not None
        orig_height, orig_width = resource_path[0].size
        orig_height, orig_width = orig_width, orig_height
        images = []
        for img_pil in resource_path:
            img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
            assert img_np.dtype == np.uint8, "np.uint8 is expected for JPEG images"
            img_np = img_np / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            img = img.to(dtype=float_dtype)
            img -= img_mean
            img /= img_std
            images.append(img)
        images = torch.stack(images)
        if not offload_video_to_cpu:
            images = images.to(device)
        return images, orig_height, orig_width

    is_image = (
        isinstance(resource_path, str)
        and os.path.splitext(resource_path)[-1].lower() in IMAGE_EXTS
    )
    if is_image:
        return load_image_as_single_frame_video(
            image_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            device=device,
            img_mean=img_mean,
            img_std=img_std,
        )
    else:
        return load_video_frames(
            video_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            device=device,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )


def load_image_as_single_frame_video(
    image_path, image_size, offload_video_to_cpu, device=None,
    img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
):
    if device is None:
        device = comfy.model_management.get_torch_device()
    float_dtype = _get_float_dtype(device)
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).to(float_dtype)
    img_mean = torch.tensor(img_mean, dtype=float_dtype)[:, None, None]
    img_std = torch.tensor(img_std, dtype=float_dtype)[:, None, None]
    if not offload_video_to_cpu:
        images = images.to(device)
        img_mean = img_mean.to(device)
        img_std = img_std.to(device)
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


def load_video_frames(
    video_path, image_size, offload_video_to_cpu, device=None,
    img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False, video_loader_type="cv2",
):
    if device is None:
        device = comfy.model_management.get_torch_device()
    assert isinstance(video_path, str)
    if video_path.startswith("<load-dummy-video"):
        match = re.match(r"<load-dummy-video-(\d+)>", video_path)
        num_frames = int(match.group(1)) if match else 60
        return load_dummy_video(image_size, offload_video_to_cpu, device, num_frames=num_frames)
    elif os.path.isdir(video_path):
        return load_video_frames_from_image_folder(
            image_folder=video_path, image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu, device=device,
            img_mean=img_mean, img_std=img_std,
            async_loading_frames=async_loading_frames,
        )
    elif os.path.splitext(video_path)[-1].lower() in VIDEO_EXTS:
        return load_video_frames_from_video_file(
            video_path=video_path, image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu, device=device,
            img_mean=img_mean, img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
    else:
        raise NotImplementedError("Only video files and image folders are supported")


def load_video_frames_from_image_folder(
    image_folder, image_size, offload_video_to_cpu, device,
    img_mean, img_std, async_loading_frames,
):
    float_dtype = _get_float_dtype(device)
    frame_names = [
        p for p in os.listdir(image_folder)
        if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
    ]
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        logger.warning(
            f'frame names are not in "<frame_index>.<img_ext>" format: {frame_names[:5]=}, '
            f"falling back to lexicographic sort."
        )
        frame_names.sort()
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {image_folder}")
    img_paths = [os.path.join(image_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=float_dtype)[:, None, None]
    img_std = torch.tensor(img_std, dtype=float_dtype)[:, None, None]
    if async_loading_frames:
        lazy_images = LazyImageFrameLoader(
            img_paths, image_size, offload_video_to_cpu, device, img_mean, img_std
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=float_dtype)
    video_height, video_width = None, None
    for n, img_path in enumerate(
        tqdm(img_paths, desc=f"frame loading (image folder) [rank={RANK}]")
    ):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.to(device)
        img_mean = img_mean.to(device)
        img_std = img_std.to(device)
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path, image_size, offload_video_to_cpu, device,
    img_mean, img_std, async_loading_frames,
    gpu_acceleration=False, gpu_device=None, video_loader_type="cv2",
):
    if video_loader_type == "cv2":
        return load_video_frames_from_video_file_using_cv2(
            video_path=video_path, image_size=image_size,
            img_mean=img_mean, img_std=img_std,
            offload_video_to_cpu=offload_video_to_cpu, device=device,
        )
    elif video_loader_type == "torchcodec":
        logger.info("Using torchcodec to load video file")
        lazy_images = AsyncVideoFileLoaderWithTorchCodec(
            video_path=video_path, image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean, img_std=img_std,
            gpu_acceleration=gpu_acceleration, gpu_device=gpu_device, device=device,
        )
        if not async_loading_frames:
            async_thread = lazy_images.thread
            if async_thread is not None:
                async_thread.join()
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    else:
        raise RuntimeError("video_loader_type must be either 'cv2' or 'torchcodec'")


def load_video_frames_from_video_file_using_cv2(
    video_path: str, image_size: int,
    img_mean: tuple = (0.5, 0.5, 0.5), img_std: tuple = (0.5, 0.5, 0.5),
    offload_video_to_cpu: bool = False, device: torch.device = None,
) -> torch.Tensor:
    import cv2
    if device is None:
        device = comfy.model_management.get_torch_device()
    float_dtype = _get_float_dtype(device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = num_frames if num_frames > 0 else None
    frames = []
    pbar = tqdm(desc=f"frame loading (OpenCV) [rank={RANK}]", total=num_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        frames.append(frame_resized)
        pbar.update(1)
    cap.release()
    pbar.close()
    frames_np = np.stack(frames, axis=0).astype(np.float32)
    video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
    img_mean = torch.tensor(img_mean, dtype=float_dtype).view(1, 3, 1, 1)
    img_std = torch.tensor(img_std, dtype=float_dtype).view(1, 3, 1, 1)
    if not offload_video_to_cpu:
        video_tensor = video_tensor.to(device)
        img_mean = img_mean.to(device)
        img_std = img_std.to(device)
    video_tensor -= img_mean
    video_tensor /= img_std
    return video_tensor, original_height, original_width


def load_dummy_video(image_size, offload_video_to_cpu, device, num_frames=60):
    float_dtype = _get_float_dtype(device)
    video_height, video_width = 480, 640
    images = torch.randn(num_frames, 3, image_size, image_size, dtype=float_dtype)
    if not offload_video_to_cpu:
        images = images.to(device)
    return images, video_height, video_width


def _load_img_as_tensor(img_path, image_size):
    img = Image.open(img_path).convert("RGB")
    orig_width, orig_height = img.width, img.height
    img = TF.resize(img, size=(image_size, image_size))
    img = TF.to_tensor(img)
    return img, orig_height, orig_width


class LazyImageFrameLoader:
    def __init__(self, img_paths, image_size, offload_video_to_cpu, device,
                 img_mean, img_std, max_cached_frames=64):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.device = device
        self.img_mean = img_mean
        self.img_std = img_std
        self.images = {}
        self._access_order = []
        self.max_cached_frames = max(4, max_cached_frames)
        self.video_height = None
        self.video_width = None
        self._num_frames = len(img_paths)
        self._load_count = 0
        log.info(f"LazyLoader initialized: {self._num_frames} frames, "
                 f"max_cached={self.max_cached_frames}, offload_to_cpu={offload_video_to_cpu}")
        self.__getitem__(0)

    def _evict_if_needed(self):
        while len(self.images) > self.max_cached_frames and self._access_order:
            evict_idx = self._access_order.pop(0)
            self.images.pop(evict_idx, None)

    def __getitem__(self, index):
        img = self.images.get(index)
        if img is not None:
            if index in self._access_order:
                self._access_order.remove(index)
            self._access_order.append(index)
            return img
        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        img = img.to(dtype=_get_float_dtype(self.device))
        img -= self.img_mean
        img /= self.img_std
        self.images[index] = img
        self._access_order.append(index)
        self._evict_if_needed()
        return img

    def __len__(self):
        return self._num_frames


class TorchCodecDecoder:
    def __init__(self, source, dimension_order="NCHW", device="cpu", num_threads=1):
        from torchcodec import _core as core
        self._source = source
        if isinstance(source, str):
            self._decoder = core.create_from_file(source, "exact")
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source, "exact")
        else:
            raise TypeError(f"Unknown source type: {type(source)}.")
        assert dimension_order in ("NCHW", "NHWC")
        device_string = str(device)
        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(
            self._decoder, dimension_order=dimension_order,
            device=device_string,
            num_threads=(1 if "cuda" in device_string else num_threads),
        )
        video_metadata = core.get_container_metadata(self._decoder)
        best_stream_index = video_metadata.best_video_stream_index
        assert best_stream_index is not None
        self.metadata = video_metadata.streams[best_stream_index]
        assert self.metadata.num_frames_from_content is not None
        self._num_frames = self.metadata.num_frames_from_content

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key: int):
        from torchcodec import _core as core
        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(f"Index {key} is out of bounds; length is {self._num_frames}")
        frame_data, *_ = core.get_frame_at_index(self._decoder, frame_index=key)
        return frame_data


class FIFOLock:
    def __init__(self):
        self._lock = Lock()
        self._waiters = queue.Queue()
        self._condition = Condition()

    def acquire(self):
        ident = get_ident()
        with self._condition:
            self._waiters.put(ident)
            while self._waiters.queue[0] != ident or not self._lock.acquire(blocking=False):
                self._condition.wait()

    def release(self):
        with self._condition:
            self._lock.release()
            self._waiters.get()
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()

    def __exit__(self, t, v, tb):
        self.release()


class AsyncVideoFileLoaderWithTorchCodec:
    def __init__(
        self, video_path, image_size, offload_video_to_cpu,
        img_mean, img_std, gpu_acceleration=True, gpu_device=None,
        device=None, use_rand_seek_in_loading=False,
    ):
        gpu_id = (
            gpu_device.index
            if gpu_device is not None and gpu_device.index is not None
            else (torch.cuda.current_device() if comfy.model_management.get_torch_device().type == "cuda" else None)
        )
        if device is not None:
            out_device = device
        elif offload_video_to_cpu:
            out_device = torch.device("cpu")
        else:
            out_device = comfy.model_management.get_torch_device() if gpu_device is None else gpu_device
        self.out_device = out_device
        float_dtype = _get_float_dtype(out_device)
        self.gpu_acceleration = gpu_acceleration and out_device.type == "cuda"
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        if not isinstance(img_mean, torch.Tensor):
            img_mean = torch.tensor(img_mean, dtype=float_dtype)[:, None, None]
        self.img_mean = img_mean
        if not isinstance(img_std, torch.Tensor):
            img_std = torch.tensor(img_std, dtype=float_dtype)[:, None, None]
        self.img_std = img_std
        if self.gpu_acceleration:
            self.img_mean = self.img_mean.to(f"cuda:{self.gpu_id}")
            self.img_std = self.img_std.to(f"cuda:{self.gpu_id}")
            decoder_option = {"device": f"cuda:{self.gpu_id}"}
        else:
            self.img_mean = self.img_mean.to(out_device)
            self.img_std = self.img_std.to(out_device)
            decoder_option = {"num_threads": 1}
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.async_reader = TorchCodecDecoder(video_path, **decoder_option)
        self.num_frames = self.async_reader.metadata.num_frames_from_content
        self.video_height = self.async_reader.metadata.height
        self.video_width = self.async_reader.metadata.width
        self.images_loaded = [False] * self.num_frames
        self.images = torch.zeros(
            self.num_frames, 3, self.image_size, self.image_size,
            dtype=float_dtype, device=self.out_device,
        )
        self.exception = None
        self.use_rand_seek_in_loading = use_rand_seek_in_loading
        self.rand_seek_idx_queue = queue.Queue()
        self.torchcodec_access_lock = FIFOLock()
        self._start_video_loading()

    def _load_one_frame(self, idx):
        frame_resized = self._transform_frame(self.async_reader[idx])
        return frame_resized

    @torch.inference_mode()
    def _start_video_loading(self):
        desc = f"frame loading (TorchCodec w/ {'GPU' if self.gpu_acceleration else 'CPU'}) [rank={RANK}]"
        pbar = tqdm(desc=desc, total=self.num_frames)
        self.num_loaded_frames = 0
        idx = self.num_loaded_frames
        self.images[idx] = self._load_one_frame(idx)
        self.images_loaded[idx] = True
        self.num_loaded_frames += 1
        pbar.update(n=1)
        self.all_frames_loaded = self.num_loaded_frames == self.num_frames

        def _load_frames():
            finished = self.all_frames_loaded
            chunk_size = 16
            while not finished:
                with self.torchcodec_access_lock, torch.inference_mode():
                    for _ in range(chunk_size):
                        try:
                            idx = self.num_loaded_frames
                            self.images[idx] = self._load_one_frame(idx)
                            self.images_loaded[idx] = True
                            self.num_loaded_frames += 1
                            pbar.update(n=1)
                            if self.num_loaded_frames >= self.num_frames:
                                finished = True
                                break
                        except Exception as e:
                            self.exception = e
                            raise
                    while True:
                        try:
                            idx = self.rand_seek_idx_queue.get_nowait()
                            if not self.images_loaded[idx]:
                                self.images[idx] = self._load_one_frame(idx)
                                self.images_loaded[idx] = True
                        except queue.Empty:
                            break
                        except Exception as e:
                            self.exception = e
                            raise
            if self.num_loaded_frames != self.num_frames:
                raise RuntimeError(
                    f"There are {self.num_frames} frames in the video, but only "
                    f"{self.num_loaded_frames} frames can be loaded successfully."
                )
            else:
                self.all_frames_loaded = True
                pbar.close()
                with self.torchcodec_access_lock:
                    import gc
                    reader = self.async_reader
                    if reader is not None:
                        reader._source = None
                    self.async_reader = None
                    self.pbar = None
                    self.thread = None
                    self.rand_seek_idx_queue = None
                    gc.collect()
                self.torchcodec_access_lock = contextlib.nullcontext()

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def _transform_frame(self, frame):
        frame = frame.clone()
        frame = frame.float()
        frame_resized = F.interpolate(
            frame[None, :], size=(self.image_size, self.image_size),
            mode="bicubic", align_corners=False,
        )[0]
        frame_resized = frame_resized.to(dtype=_get_float_dtype(self.out_device))
        frame_resized /= 255
        frame_resized -= self.img_mean
        frame_resized /= self.img_std
        if self.offload_video_to_cpu:
            frame_resized = frame_resized.cpu()
        elif frame_resized.device != self.out_device:
            frame_resized = frame_resized.to(device=self.out_device, non_blocking=torch.cuda.is_available())
        return frame_resized

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception
        max_tries = 1200
        for _ in range(max_tries):
            with self.torchcodec_access_lock:
                if self.images_loaded[index]:
                    return self.images[index]
                if self.use_rand_seek_in_loading:
                    self.rand_seek_idx_queue.put(index)
            time.sleep(0.1)
        raise RuntimeError(f"Failed to load frame {index} after {max_tries} tries")

    def __len__(self):
        return len(self.images)

    def __getstate__(self):
        async_thread = self.thread
        if async_thread is not None:
            async_thread.join()
        reader = self.async_reader
        if reader is not None:
            reader._source = None
        self.async_reader = None
        self.pbar = None
        self.thread = None
        self.rand_seek_idx_queue = None
        self.torchcodec_access_lock = contextlib.nullcontext()
        return self.__dict__.copy()


# ===========================================================================
# SAM3 Image Processor (from model/sam3_image_processor.py)
# ===========================================================================

class Sam3Processor:
    def __init__(self, model, resolution=1008, device=None, confidence_threshold=0.2):
        from torchvision.transforms import v2
        self.model = model
        self.resolution = resolution
        if device is None:
            device = str(comfy.model_management.get_torch_device())
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        from torchvision.transforms import v2
        if state is None:
            state = {}
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)
        image = v2.functional.to_image(image).to(model_device)
        image = self.transform(image).unsqueeze(0)
        # Native ComfyUI pattern: cast input to target dtype at the pipeline
        # boundary. manual_cast layers will cast their fp32 weights to match.
        inference_dtype = getattr(self, '_inference_dtype', None)
        if inference_dtype is not None and inference_dtype in (torch.float16, torch.bfloat16):
            image = image.to(dtype=inference_dtype)
        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        from torchvision.transforms import v2
        if state is None:
            state = {}
        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], Image.Image), "Images must be a list of PIL images"
        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)
        images = [
            self.transform(v2.functional.to_image(image).to(model_device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        inference_dtype = getattr(self, '_inference_dtype', None)
        if inference_dtype is not None and inference_dtype in (torch.float16, torch.bfloat16):
            images = images.to(dtype=inference_dtype)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_multiple_box_prompts(self, boxes: List[List], labels: List[bool], state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_multiple_box_prompts")
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        boxes_tensor = torch.tensor(boxes, device=self.device, dtype=torch.float32).view(len(boxes), 1, 4)
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.bool).view(len(labels), 1)
        state["geometric_prompt"].append_boxes(boxes_tensor, labels_tensor)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_point_prompt(self, points: List[List], labels: List[int], state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        points_tensor = torch.tensor(points, device=self.device, dtype=torch.float32).view(len(points), 1, 2)
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.long).view(len(labels), 1)
        state["geometric_prompt"].append_points(points_tensor, labels_tensor)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_mask_prompt(self, mask: torch.Tensor, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_mask_prompt")
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        if mask.device != self.device:
            mask = mask.to(self.device)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        state["geometric_prompt"].append_masks(mask)
        return self._forward_grounding(state)

    def sync_device_with_model(self):
        try:
            model_device = next(self.model.parameters()).device
            self.device = str(model_device)
            if self.find_stage is not None:
                if self.find_stage.img_ids is not None:
                    self.find_stage.img_ids = self.find_stage.img_ids.to(model_device)
                if self.find_stage.text_ids is not None:
                    self.find_stage.text_ids = self.find_stage.text_ids.to(model_device)
        except StopIteration:
            pass

    def reset_all_prompts(self, state: Dict):
        if "backbone_out" in state:
            backbone_keys_to_del = ["language_features", "language_mask", "language_embeds"]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]
        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid().squeeze(-1)
        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]
        out_masks = interpolate(
            out_masks.unsqueeze(1), (img_h, img_w), mode="bilinear", align_corners=False,
        ).sigmoid()
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state


# ===========================================================================
# Tracker utilities (from model/sam3_tracker_utils.py)
# ===========================================================================

def sample_box_points(
    masks: torch.Tensor, noise: float = 0.1, noise_bound: int = 20,
    top_left_label: int = 2, bottom_right_label: int = 3,
):
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)
        box_coords = box_coords + box_noise
        img_bounds = torch.tensor([W, H, W, H], device=device) - 1
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)
    box_coords = box_coords.reshape(-1, 2, 2)
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def mask_to_box(masks: torch.Tensor):
    B, _, h, w = masks.shape
    device = masks.device
    mask_area = masks.sum(dim=(-1, -2))
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    bbox_coords = torch.where(mask_area[..., None] > 0, bbox_coords, torch.zeros_like(bbox_coords))
    return bbox_coords


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0
    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device
    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels



def select_closest_cond_frames(
    frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False
):
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min(
                (t for t in cond_frame_outputs if t < frame_idx), default=None
            )
            if idx_first is None:
                idx_first = max(
                    (t for t in cond_frame_outputs if t > frame_idx), default=None
                )
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }
    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_best_gt_match_from_multimasks(pred_multimasks, gt_masks, pred_scores=None):
    assert pred_multimasks.ndim == 4 and gt_masks.ndim == 4
    if pred_multimasks.size(1) == 1:
        return pred_multimasks
    pred_multimasks_binary = pred_multimasks > 0
    area_i = torch.sum(pred_multimasks_binary & gt_masks, dim=(2, 3)).float()
    area_u = torch.sum(pred_multimasks_binary | gt_masks, dim=(2, 3)).float()
    ious = area_i / torch.clamp(area_u, min=1.0)
    if pred_scores is not None:
        has_nonzero_ious = torch.any(ious > 0).expand_as(ious)
        scores = torch.where(has_nonzero_ious, ious, pred_scores)
    else:
        scores = ious
    best_scores_inds = torch.argmax(scores, dim=-1)
    batch_inds = torch.arange(scores.size(0), device=scores.device)
    best_pred_mask = pred_multimasks[batch_inds, best_scores_inds].unsqueeze(1)
    return best_pred_mask


def fill_holes_in_mask_scores(mask, max_area, fill_holes=True, remove_sprinkles=True):
    if max_area <= 0:
        return mask
    if fill_holes:
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)
    if remove_sprinkles:
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)
    return mask


def _get_connected_components_with_padding(mask):
    from .perflib import connected_components
    mask = mask.to(torch.uint8)
    _, _, H, W = mask.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h == 0 and pad_w == 0:
        labels, counts = connected_components(mask)
    else:
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
        labels, counts = connected_components(mask_pad)
        labels = labels[:, :, :H, :W]
        counts = counts[:, :, :H, :W]
    return labels, counts


# ===========================================================================
# SAM2Transforms (from model/utils/sam2_utils.py)
# ===========================================================================

from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
            Resize((self.resolution, self.resolution)),
            Normalize(self.mean, self.std),
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        coords = coords * self.resolution
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)
        try:
            from .perflib import connected_components
            if self.max_hole_area > 0:
                labels, areas = connected_components(
                    (mask_flat <= self.mask_threshold).to(torch.uint8)
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)
            if self.max_sprinkle_area > 0:
                labels, areas = connected_components(
                    (mask_flat > self.mask_threshold).to(torch.uint8)
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 3 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam3/blob/main/INSTALL.md).",
                category=UserWarning, stacklevel=2,
            )
            masks = input_masks
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks




# ---------------------------------------------------------------------------
# Prompt utilities (from geometry_encoders.py)
# ---------------------------------------------------------------------------

def is_right_padded(mask):
    """Given a padding mask (following pytorch convention, 1s for padded values),
    returns whether the padding is on the right or not."""
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Following pytorch's convention, tensors are sequence first, and the mask are
    batch first, with 1s for padded values.

    :param seq1: A tensor of shape (seq1_length, batch_size, hidden_size).
    :param mask1: A tensor of shape (batch_size, seq1_length).
    :param seq2: A tensor of shape (seq2_length, batch_size,  hidden_size).
    :param mask2: A tensor of shape (batch_size, seq2_length).
    :param return_index: If True, also returns the index of the ids of the element of seq2
        in the concatenated sequence. This can be used to retrieve the elements of seq2
    :return: A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    torch._assert_async(is_right_padded(mask1))
    torch._assert_async(is_right_padded(mask2))

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
        >= final_lengths[:, None]
    )

    # (max_len, batch_size, hidden_size)
    concatenated_sequence = torch.zeros(
        (max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype
    )
    concatenated_sequence[:seq1_length, :, :] = seq1

    # At this point, the element of seq1 are in the right place
    # We just need to shift the elements of seq2

    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]

    concatenated_sequence = concatenated_sequence.scatter(
        0, index[:, :, None].expand(-1, -1, hidden_size), seq2
    )

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class Prompt:
    """Utility class to manipulate geometric prompts.

    We expect the sequences in pytorch convention, that is sequence first, batch second
    The dimensions are expected as follows:
    box_embeddings shape: N_boxes x B x C_box
    box_mask shape: B x N_boxes. Can be None if nothing is masked out
    point_embeddings shape: N_points x B x C_point
    point_mask shape: B x N_points. Can be None if nothing is masked out
    mask_embeddings shape: N_masks x B x 1 x H_mask x W_mask
    mask_mask shape: B x N_masks. Can be None if nothing is masked out

    We also store positive/negative labels. These tensors are also stored batch-first
    If they are None, we'll assume positive labels everywhere
    box_labels: long tensor of shape N_boxes x B
    point_labels: long tensor of shape N_points x B
    mask_labels: long tensor of shape N_masks x B
    """

    def __init__(
        self,
        box_embeddings=None,
        box_mask=None,
        point_embeddings=None,
        point_mask=None,
        box_labels=None,
        point_labels=None,
        mask_embeddings=None,
        mask_mask=None,
        mask_labels=None,
    ):
        # Check for null prompt
        if (
            box_embeddings is None
            and point_embeddings is None
            and mask_embeddings is None
        ):
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            self.point_embeddings = None
            self.point_labels = None
            self.point_mask = None
            self.mask_embeddings = None
            self.mask_mask = None
            self.mask_labels = None
            return
        # Get sequence lengths and device
        box_seq_len, point_seq_len, mask_seq_len, bs, device = (
            self._init_seq_len_and_device(
                box_embeddings, point_embeddings, mask_embeddings
            )
        )

        # Initialize embeds, labels, attention masks.
        box_embeddings, box_labels, box_mask = self._init_box(
            box_embeddings, box_labels, box_mask, box_seq_len, bs, device
        )
        point_embeddings, point_labels, point_mask = self._init_point(
            point_embeddings, point_labels, point_mask, point_seq_len, bs, device
        )
        mask_embeddings, mask_labels, mask_mask = self._init_mask(
            mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
        )

        # Dimension checks
        assert (
            box_embeddings is not None
            and list(box_embeddings.shape[:2]) == [box_seq_len, bs]
        ), f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        assert (
            box_mask is not None
            and list(box_mask.shape) == [bs, box_seq_len]
        ), f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        assert (
            point_embeddings is not None
            and list(point_embeddings.shape[:2]) == [point_seq_len, bs]
        ), f"Wrong dimension for point embeddings. Expected [{point_seq_len}, {bs}, *] got {point_embeddings.shape}"
        assert (
            point_mask is not None
            and list(point_mask.shape) == [bs, point_seq_len]
        ), f"Wrong dimension for point mask. Expected [{bs}, {point_seq_len}] got {point_mask.shape}"
        assert (
            box_labels is not None
            and list(box_labels.shape) == [box_seq_len, bs]
        ), f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        assert (
            point_labels is not None
            and list(point_labels.shape) == [point_seq_len, bs]
        ), f"Wrong dimension for point labels. Expected [{point_seq_len}, {bs}] got {point_labels.shape}"
        assert (
            mask_embeddings is None
            or list(mask_embeddings.shape[:2]) == [mask_seq_len, bs]
        ), f"Wrong dimension for mask embeddings. Expected [{mask_seq_len}, {bs}, *] got {mask_embeddings.shape}"
        assert (
            mask_mask is None
            or list(mask_mask.shape) == [bs, mask_seq_len]
        ), f"Wrong dimension for mask attn. mask. Expected [{bs}, {mask_seq_len}] got {mask_mask.shape}"

        # Device checks
        assert (
            box_embeddings is not None and box_embeddings.device == device
        ), f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        assert (
            box_mask is not None and box_mask.device == device
        ), f"Expected box mask to be on device {device}, got {box_mask.device}"
        assert (
            box_labels is not None and box_labels.device == device
        ), f"Expected box labels to be on device {device}, got {box_labels.device}"
        assert (
            point_embeddings is not None and point_embeddings.device == device
        ), f"Expected point embeddings to be on device {device}, got {point_embeddings.device}"
        assert (
            point_mask is not None and point_mask.device == device
        ), f"Expected point mask to be on device {device}, got {point_mask.device}"
        assert (
            point_labels is not None and point_labels.device == device
        ), f"Expected point labels to be on device {device}, got {point_labels.device}"
        assert (
            mask_embeddings is None or mask_embeddings.device == device
        ), f"Expected mask embeddings to be on device {device}, got {mask_embeddings.device}"
        assert (
            mask_mask is None or mask_mask.device == device
        ), f"Expected mask attn. mask to be on device {device}, got {mask_mask.device}"

        self.box_embeddings = box_embeddings
        self.point_embeddings = point_embeddings
        self.box_mask = box_mask
        self.point_mask = point_mask
        self.box_labels = box_labels
        self.point_labels = point_labels
        self.mask_embeddings = mask_embeddings
        self.mask_labels = mask_labels
        self.mask_mask = mask_mask

    def _init_seq_len_and_device(
        self, box_embeddings, point_embeddings, mask_embeddings
    ):
        box_seq_len = point_seq_len = mask_seq_len = 0
        bs = None
        device = None
        if box_embeddings is not None:
            bs = box_embeddings.shape[1]
            box_seq_len = box_embeddings.shape[0]
            device = box_embeddings.device

        if point_embeddings is not None:
            point_seq_len = point_embeddings.shape[0]
            if bs is not None:
                assert (
                    bs == point_embeddings.shape[1]
                ), f"Batch size mismatch between box and point embeddings. Got {bs} and {point_embeddings.shape[1]}."
            else:
                bs = point_embeddings.shape[1]
            if device is not None:
                assert (
                    device == point_embeddings.device
                ), "Device mismatch between box and point embeddings"
            else:
                device = point_embeddings.device

        if mask_embeddings is not None:
            mask_seq_len = mask_embeddings.shape[0]
            if bs is not None:
                assert (
                    bs == mask_embeddings.shape[1]
                ), f"Batch size mismatch between box/point and mask embedding. Got {bs} and {mask_embeddings.shape[1]}"
            else:
                bs = mask_embeddings.shape[1]
            if device is not None:
                assert (
                    device == mask_embeddings.device
                ), "Device mismatch between box/point and mask embeddings."
            else:
                device = mask_embeddings.device

        return box_seq_len, point_seq_len, mask_seq_len, bs, device

    def _init_box(self, box_embeddings, box_labels, box_mask, box_seq_len, bs, device):
        if box_embeddings is None:
            box_embeddings = torch.zeros(box_seq_len, bs, 4, device=device)
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)
        return box_embeddings, box_labels, box_mask

    def _init_point(
        self, point_embeddings, point_labels, point_mask, point_seq_len, bs, device
    ):
        if point_embeddings is None:
            point_embeddings = torch.zeros(point_seq_len, bs, 2, device=device)
        if point_labels is None:
            point_labels = torch.ones(
                point_seq_len, bs, device=device, dtype=torch.long
            )
        if point_mask is None:
            point_mask = torch.zeros(bs, point_seq_len, device=device, dtype=torch.bool)
        return point_embeddings, point_labels, point_mask

    def _init_mask(
        self, mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
    ):
        if mask_labels is None:
            mask_labels = torch.ones(mask_seq_len, bs, device=device, dtype=torch.long)
        if mask_mask is None:
            mask_mask = torch.zeros(bs, mask_seq_len, device=device, dtype=torch.bool)
        return mask_embeddings, mask_labels, mask_mask

    def append_boxes(self, boxes, labels, mask=None):
        if self.box_embeddings is None:
            self.box_embeddings = boxes
            self.box_labels = labels
            self.box_mask = mask
            return

        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == labels.shape[1] == bs
        assert list(boxes.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(
                bs, boxes.shape[0], dtype=torch.bool, device=boxes.device
            )

        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(
            self.box_embeddings, self.box_mask, boxes, mask
        )

    def append_points(self, points, labels, mask=None):
        if self.point_embeddings is None:
            self.point_embeddings = points
            self.point_labels = labels
            self.point_mask = mask
            return

        bs = self.point_embeddings.shape[1]
        assert points.shape[1] == labels.shape[1] == bs
        assert list(points.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(
                bs, points.shape[0], dtype=torch.bool, device=points.device
            )

        self.point_labels, _ = concat_padded_sequences(
            self.point_labels.unsqueeze(-1), self.point_mask, labels.unsqueeze(-1), mask
        )
        self.point_labels = self.point_labels.squeeze(-1)
        self.point_embeddings, self.point_mask = concat_padded_sequences(
            self.point_embeddings, self.point_mask, points, mask
        )

    def append_masks(self, masks, labels=None, attn_mask=None):
        if labels is not None:
            assert list(masks.shape[:2]) == list(labels.shape[:2])
        if self.mask_embeddings is None:
            self.mask_embeddings = masks
            mask_seq_len, bs = masks.shape[:2]
            if labels is None:
                self.mask_labels = torch.ones(
                    mask_seq_len, bs, device=masks.device, dtype=torch.long
                )
            else:
                self.mask_labels = labels
            if attn_mask is None:
                self.mask_mask = torch.zeros(
                    bs, mask_seq_len, device=masks.device, dtype=torch.bool
                )
            else:
                self.mask_mask = attn_mask
        else:
            raise NotImplementedError("Only one mask per prompt is supported.")

    def clone(self):
        return Prompt(
            box_embeddings=(
                None if self.box_embeddings is None else self.box_embeddings.clone()
            ),
            box_mask=None if self.box_mask is None else self.box_mask.clone(),
            point_embeddings=(
                None if self.point_embeddings is None else self.point_embeddings.clone()
            ),
            point_mask=None if self.point_mask is None else self.point_mask.clone(),
            box_labels=None if self.box_labels is None else self.box_labels.clone(),
            point_labels=(
                None if self.point_labels is None else self.point_labels.clone()
            ),
        )
