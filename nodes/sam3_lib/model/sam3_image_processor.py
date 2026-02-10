# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import PIL
import torch

from . import box_ops

from .data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.2):
        self.model = model
        self.resolution = resolution
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
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        # Get model device/dtype and ensure image matches
        try:
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_device = torch.device(self.device)
            model_dtype = torch.float32

        image = v2.functional.to_image(image).to(model_device)
        image = self.transform(image).unsqueeze(0)

        # Ensure image dtype matches model for mixed precision scenarios
        if image.dtype != model_dtype and model_dtype in (torch.float16, torch.bfloat16):
            image = image.to(dtype=model_dtype)

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
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        # Get model device/dtype and ensure images match
        try:
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_device = torch.device(self.device)
            model_dtype = torch.float32

        images = [
            self.transform(v2.functional.to_image(image).to(model_device))
            for image in images
        ]
        images = torch.stack(images, dim=0)

        # Ensure images dtype matches model for mixed precision scenarios
        if images.dtype != model_dtype and model_dtype in (torch.float16, torch.bfloat16):
            images = images.to(dtype=model_dtype)

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
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_multiple_box_prompts(self, boxes: List[List], labels: List[bool], state: Dict):
        """Adds multiple box prompts and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        Each box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        Each label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_multiple_box_prompts")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Convert to [seq_len, batch_size, 4] format
        boxes_tensor = torch.tensor(boxes, device=self.device, dtype=torch.float32).view(len(boxes), 1, 4)
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.bool).view(len(labels), 1)
        state["geometric_prompt"].append_boxes(boxes_tensor, labels_tensor)

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_point_prompt(self, points: List[List], labels: List[int], state: Dict):
        """Adds point prompts and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        Points should be in [x, y] format, normalized in [0, 1] range.
        Labels should be 1 for foreground points, 0 for background points.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Convert to [seq_len, batch_size, 2] format
        points_tensor = torch.tensor(points, device=self.device, dtype=torch.float32).view(len(points), 1, 2)
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.long).view(len(labels), 1)
        state["geometric_prompt"].append_points(points_tensor, labels_tensor)

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_mask_prompt(self, mask: torch.Tensor, state: Dict):
        """Adds a mask prompt and run the inference.
        The mask should be a binary tensor with shape matching the model's expected input.
        This is typically used for iterative refinement.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_mask_prompt")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Ensure mask is on correct device and has batch dimension
        if mask.device != self.device:
            mask = mask.to(self.device)

        # Add sequence and batch dimensions if needed: [seq_len, batch_size, H, W]
        if len(mask.shape) == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:  # [1, H, W] or [batch, H, W]
            mask = mask.unsqueeze(0)

        state["geometric_prompt"].append_masks(mask)

        return self._forward_grounding(state)

    def sync_device_with_model(self):
        """Synchronize processor device with model's actual device."""
        try:
            model_device = next(self.model.parameters()).device
            self.device = str(model_device)

            # Also sync find_stage tensors
            if self.find_stage is not None:
                if self.find_stage.img_ids is not None:
                    self.find_stage.img_ids = self.find_stage.img_ids.to(model_device)
                if self.find_stage.text_ids is not None:
                    self.find_stage.text_ids = self.find_stage.text_ids.to(model_device)
        except StopIteration:
            pass  # Model has no parameters

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
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
        # Note: presence_score is already baked into pred_logits during training
        # (via supervise_joint_box_scores=True), so we don't multiply again
        out_probs = out_logits.sigmoid().squeeze(-1)

        keep = out_probs > self.confidence_threshold

        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
