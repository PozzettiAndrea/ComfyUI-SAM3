# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Performance utilities — flattened from perflib/ subdirectory.

import logging
import os

import numpy as np
import torch

log = logging.getLogger("sam3")

# ---------------------------------------------------------------------------
# Feature flag (from perflib/__init__.py)
# ---------------------------------------------------------------------------

is_enabled = os.getenv("USE_PERFLIB", "1") == "1"


# ---------------------------------------------------------------------------
# Mask operations (from perflib/masks_ops.py)
# ---------------------------------------------------------------------------

def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        N, H, W = masks.shape
        device = masks.device
        y = torch.arange(H, device=device).view(1, H)
        x = torch.arange(W, device=device).view(1, W)

        masks_with_obj = masks != 0
        masks_with_obj_x = masks_with_obj.amax(dim=1)
        masks_with_obj_y = masks_with_obj.amax(dim=2)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        bounding_boxes_0 = torch.amin(
            (masks_without_obj_x * W) + (masks_with_obj_x * x), dim=1
        )
        bounding_boxes_1 = torch.amin(
            (masks_without_obj_y * H) + (masks_with_obj_y * y), dim=1
        )
        bounding_boxes_2 = torch.amax(masks_with_obj_x * x, dim=1)
        bounding_boxes_3 = torch.amax(masks_with_obj_y * y, dim=1)

        bounding_boxes = torch.stack(
            [bounding_boxes_0, bounding_boxes_1, bounding_boxes_2, bounding_boxes_3],
            dim=1,
        ).to(dtype=torch.float)
        assert bounding_boxes.shape == (N, 4)
        assert bounding_boxes.device == masks.device
        assert bounding_boxes.dtype == torch.float
        return bounding_boxes


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute IoU between predicted masks and ground truth masks."""
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape

    pred_flat = pred_masks.view(N, 1, H * W)
    gt_flat = gt_masks.view(1, M, H * W)

    intersection = (pred_flat & gt_flat).sum(dim=2).float()
    union = (pred_flat | gt_flat).sum(dim=2).float()
    ious = intersection / union.clamp(min=1)
    return ious


# ---------------------------------------------------------------------------
# NMS (from perflib/nms.py) — depends on mask_iou above
# ---------------------------------------------------------------------------

try:
    from torch_generic_nms import generic_nms as generic_nms_cuda
    GENERIC_NMS_AVAILABLE = True
except ImportError:
    logging.debug(
        "torch_generic_nms not available, falling back to CPU mask NMS implementation."
    )
    GENERIC_NMS_AVAILABLE = False

_SHOWN_NMS_WARNING = False


def nms_masks(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
) -> torch.Tensor:
    is_valid = pred_probs > prob_threshold
    probs = pred_probs[is_valid]
    masks_binary = pred_masks[is_valid] > 0
    if probs.numel() == 0:
        return is_valid

    ious = mask_iou(masks_binary, masks_binary)
    kept_inds = generic_nms(ious, probs, iou_threshold)

    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)
    keep = torch.isin(valid_inds, kept_inds)
    return keep


def generic_nms(
    ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    """A generic version of torchvision.ops.nms that takes a pairwise IoU matrix."""
    assert ious.dim() == 2 and ious.size(0) == ious.size(1)
    assert scores.dim() == 1 and scores.size(0) == ious.size(0)

    if ious.is_cuda:
        if GENERIC_NMS_AVAILABLE:
            return generic_nms_cuda(ious, scores, iou_threshold, use_iou_matrix=True)
        else:
            global _SHOWN_NMS_WARNING
            if not _SHOWN_NMS_WARNING:
                log.warning(
                    "GPU-accelerated NMS not available - video tracking is 5-10x slower. "
                    "To enable GPU acceleration, run: cd custom_nodes/ComfyUI-SAM3 && python install.py"
                )
                _SHOWN_NMS_WARNING = True
            return generic_nms_cpu(ious, scores, iou_threshold)

    return generic_nms_cpu(ious, scores, iou_threshold)


def generic_nms_cpu(
    ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept_inds = []
    while order.size > 0:
        i = order.item(0)
        kept_inds.append(i)
        inds = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[inds + 1]
    return torch.tensor(kept_inds, dtype=torch.int64, device=scores.device)


# ---------------------------------------------------------------------------
# Connected components (from perflib/connected_components.py)
# ---------------------------------------------------------------------------

try:
    from cc_torch import get_connected_components
    HAS_CC_TORCH = True
except ImportError:
    logging.debug("cc_torch not found. Consider installing for better performance.")
    HAS_CC_TORCH = False


def connected_components_cpu_single(values: torch.Tensor):
    assert values.dim() == 2
    from skimage.measure import label

    labels, num = label(values.cpu().numpy(), return_num=True)
    labels = torch.from_numpy(labels)
    counts = torch.zeros_like(labels)
    for i in range(1, num + 1):
        cur_mask = labels == i
        cur_count = cur_mask.sum()
        counts[cur_mask] = cur_count
    return labels, counts


def connected_components_cpu(input_tensor: torch.Tensor):
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert input_tensor.dim() == 3, "Input tensor must be (B, H, W) or (B, 1, H, W)."

    batch_size = input_tensor.shape[0]
    if batch_size == 0:
        return torch.zeros_like(input_tensor), torch.zeros_like(input_tensor)
    labels_list = []
    counts_list = []
    for b in range(batch_size):
        labels, counts = connected_components_cpu_single(input_tensor[b])
        labels_list.append(labels)
        counts_list.append(counts)
    labels_tensor = torch.stack(labels_list, dim=0).to(input_tensor.device)
    counts_tensor = torch.stack(counts_list, dim=0).to(input_tensor.device)
    return labels_tensor.view(out_shape), counts_tensor.view(out_shape)


def connected_components(input_tensor: torch.Tensor):
    """Computes connected components labeling on a batch of 2D tensors."""
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)

    assert (
        input_tensor.dim() == 4 and input_tensor.shape[1] == 1
    ), "Input tensor must be (B, H, W) or (B, 1, H, W)."

    if input_tensor.is_cuda:
        if HAS_CC_TORCH:
            return get_connected_components(input_tensor.to(torch.uint8))
        else:
            logging.debug("GPU connected components not available, using CPU fallback")
            return connected_components_cpu(input_tensor)

    return connected_components_cpu(input_tensor)


# ---------------------------------------------------------------------------
# Compile utilities (from perflib/compile.py)
# ---------------------------------------------------------------------------

def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        if b is None:
            return b
        trivial_types = [bool, int]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")
    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    seen_shapes = set()

    def get_shape(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 1:
                return tuple(get_shape(x) for x in obj)
            return get_shape(obj[0])
        elif isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        else:
            return type(obj).__name__

    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v))
            for k, v in kwargs.items()
            if isinstance(v, (torch.Tensor, list))
            and (len(keep_kwargs) > 0 and k in keep_kwargs)
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                log.info(f"New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    wrapper.enable_logging = enable_logging

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enable_logging

    wrapper.set_logging = set_logging
    return wrapper
