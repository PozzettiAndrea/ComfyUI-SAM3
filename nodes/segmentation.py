"""
SAM3Segmentation node - Performs segmentation using text prompts
"""
import torch
import numpy as np
from .utils import (
    comfy_image_to_pil,
    pil_to_comfy_image,
    masks_to_comfy_mask,
    visualize_masks_on_image,
    tensor_to_list,
    ensure_model_on_device,
    offload_model_if_needed
)


class SAM3Segmentation:
    """
    Node to perform SAM3 segmentation with text prompts

    Takes an image and text prompt, returns segmentation masks,
    bounding boxes, confidence scores, and visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Lower threshold (0.2) works better with SAM3's presence scoring"
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., 'cat', 'person in red', 'car'"
                }),
                "positive_boxes": ("SAM3_BOXES_PROMPT",),
                "negative_boxes": ("SAM3_BOXES_PROMPT",),
                "positive_points": ("SAM3_POINTS_PROMPT",),
                "negative_points": ("SAM3_POINTS_PROMPT",),
                "mask_prompt": ("MASK",),
                "max_detections": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum number of detections to return (-1 for all)"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("masks", "visualization", "boxes", "scores")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, sam3_model, image, confidence_threshold=0.2, text_prompt="",
                positive_boxes=None, negative_boxes=None, positive_points=None, negative_points=None,
                mask_prompt=None, max_detections=-1):
        """
        Perform SAM3 segmentation with optional geometric prompts

        Args:
            sam3_model: Model dict from LoadSAM3Model node
            image: ComfyUI image tensor [B, H, W, C]
            confidence_threshold: Minimum confidence score for detections
            text_prompt: Optional text description of objects to segment
            positive_boxes: Optional positive box prompts
            negative_boxes: Optional negative box prompts
            positive_points: Optional positive point prompts
            negative_points: Optional negative point prompts
            mask_prompt: Optional mask prompt
            max_detections: Maximum number of detections to return

        Returns:
            Tuple of (masks, visualization, boxes_json, scores_json)
        """
        # Extract processor from model dict
        processor = sam3_model["processor"]
        device = sam3_model["device"]

        # Ensure model is on GPU if use_gpu_cache is False (model may have been offloaded)
        ensure_model_on_device(sam3_model)
        device = sam3_model["device"]  # Update device in case it changed

        print(f"[SAM3] Running segmentation")
        if text_prompt:
            print(f"[SAM3]   Text prompt: '{text_prompt}'")
        if positive_boxes:
            print(f"[SAM3]   Positive boxes: {len(positive_boxes['boxes'])}")
        if negative_boxes:
            print(f"[SAM3]   Negative boxes: {len(negative_boxes['boxes'])}")
        if positive_points:
            print(f"[SAM3]   Positive points: {len(positive_points['points'])}")
        if negative_points:
            print(f"[SAM3]   Negative points: {len(negative_points['points'])}")
        if mask_prompt is not None:
            print(f"[SAM3]   Mask prompt provided")
        print(f"[SAM3] Confidence threshold: {confidence_threshold}")

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)
        print(f"[SAM3] Image size: {pil_image.size}")

        # Update confidence threshold
        processor.set_confidence_threshold(confidence_threshold)

        # Set image (extracts features)
        state = processor.set_image(pil_image)

        # Add text prompt if provided
        if text_prompt and text_prompt.strip():
            print(f"[SAM3] Adding text prompt...")
            state = processor.set_text_prompt(text_prompt.strip(), state)

        # Add geometric prompts - combine positive and negative
        all_boxes = []
        all_box_labels = []

        if positive_boxes is not None and len(positive_boxes['boxes']) > 0:
            all_boxes.extend(positive_boxes['boxes'])
            all_box_labels.extend(positive_boxes['labels'])

        if negative_boxes is not None and len(negative_boxes['boxes']) > 0:
            all_boxes.extend(negative_boxes['boxes'])
            all_box_labels.extend(negative_boxes['labels'])

        if len(all_boxes) > 0:
            print(f"[SAM3] Adding {len(all_boxes)} box prompts...")
            state = processor.add_multiple_box_prompts(
                all_boxes,
                all_box_labels,
                state
            )

        all_points = []
        all_point_labels = []

        if positive_points is not None and len(positive_points['points']) > 0:
            all_points.extend(positive_points['points'])
            all_point_labels.extend(positive_points['labels'])

        if negative_points is not None and len(negative_points['points']) > 0:
            all_points.extend(negative_points['points'])
            all_point_labels.extend(negative_points['labels'])

        if len(all_points) > 0:
            print(f"[SAM3] Adding {len(all_points)} point prompts...")
            state = processor.add_point_prompt(
                all_points,
                all_point_labels,
                state
            )

        if mask_prompt is not None:
            print(f"[SAM3] Adding mask prompt...")
            if not isinstance(mask_prompt, torch.Tensor):
                mask_prompt = torch.from_numpy(mask_prompt)
            mask_prompt = mask_prompt.to(device)
            state = processor.add_mask_prompt(mask_prompt, state)

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        # DEBUG: Show ALL predictions before threshold filtering
        print(f"[SAM3 DEBUG] ========== RAW PREDICTIONS ==========")
        print(f"[SAM3 DEBUG] State keys: {list(state.keys())}")
        if scores is not None and len(scores) > 0:
            print(f"[SAM3 DEBUG] Total predictions: {len(scores)}")
            print(f"[SAM3 DEBUG] Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"[SAM3 DEBUG] Score mean: {scores.mean():.4f}")
            # Show top 10 scores regardless of threshold
            top_10_scores = torch.topk(scores, min(10, len(scores)))
            print(f"[SAM3 DEBUG] Top 10 scores: {top_10_scores.values.tolist()}")
            # Show score distribution at different thresholds
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                count = (scores > thresh).sum().item()
                print(f"[SAM3 DEBUG]   Detections > {thresh}: {count}")
        elif scores is not None:
            print(f"[SAM3 DEBUG] Total predictions: 0 (empty tensor!)")
            print(f"[SAM3 DEBUG] ⚠️ Model returned ZERO predictions - this is unusual!")
        else:
            print(f"[SAM3 DEBUG] No scores in state!")

        if masks is not None:
            print(f"[SAM3 DEBUG] Masks shape: {masks.shape}")
        if boxes is not None:
            print(f"[SAM3 DEBUG] Boxes shape: {boxes.shape}")
            print(f"[SAM3 DEBUG] Sample boxes: {boxes[:3].tolist() if len(boxes) > 0 else 'empty'}")

        # Check masks_logits for raw model output
        if 'masks_logits' in state:
            masks_logits = state['masks_logits']
            print(f"[SAM3 DEBUG] Masks logits shape: {masks_logits.shape if masks_logits is not None else 'None'}")

        print(f"[SAM3 DEBUG] ====================================")

        # Check if we got any results AFTER threshold
        if masks is None or len(masks) == 0:
            print(f"[SAM3] No detections found for prompt: '{text_prompt}' at threshold {confidence_threshold}")
            print(f"[SAM3] TIP: Try lowering the confidence_threshold or check if the object is in the image")
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            # Offload model to CPU if use_gpu_cache is False
            offload_model_if_needed(sam3_model)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        print(f"[SAM3] Found {len(masks)} detections above threshold {confidence_threshold}")

        # Limit number of detections if specified
        if max_detections > 0 and len(masks) > max_detections:
            print(f"[SAM3] Limiting to top {max_detections} detections")
            # Sort by score and take top k
            if scores is not None:
                top_indices = torch.argsort(scores, descending=True)[:max_detections]
                masks = masks[top_indices]
                boxes = boxes[top_indices] if boxes is not None else None
                scores = scores[top_indices] if scores is not None else None

        # Convert masks to ComfyUI format
        comfy_masks = masks_to_comfy_mask(masks)

        # Create visualization
        print(f"[SAM3] Creating visualization...")
        vis_image = visualize_masks_on_image(
            pil_image,
            masks,
            boxes,
            scores,
            alpha=0.5
        )
        vis_tensor = pil_to_comfy_image(vis_image)

        # Convert boxes and scores to JSON strings for output
        boxes_list = tensor_to_list(boxes) if boxes is not None else []
        scores_list = tensor_to_list(scores) if scores is not None else []

        # Format as JSON strings
        import json
        boxes_json = json.dumps(boxes_list, indent=2)
        scores_json = json.dumps(scores_list, indent=2)

        print(f"[SAM3] Segmentation complete")
        print(f"[SAM3] Output: {len(comfy_masks)} masks")
        print(f"[SAM3 DEBUG] Final scores: {scores_list}")
        print(f"[SAM3 DEBUG] Mask output shape: {comfy_masks.shape}")

        # Offload model to CPU if use_gpu_cache is False
        offload_model_if_needed(sam3_model)

        return (comfy_masks, vis_tensor, boxes_json, scores_json)


class SAM3CreateBox:
    """
    Helper node to create a box prompt visually

    Use sliders to define a bounding box for refinement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box center X (normalized 0-1)"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box center Y (normalized 0-1)"
                }),
                "width": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box width (normalized 0-1)"
                }),
                "height": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box height (normalized 0-1)"
                }),
                "is_positive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True for positive (include), False for negative (exclude)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_BOX_PROMPT",)
    RETURN_NAMES = ("box_prompt",)
    FUNCTION = "create_box"
    CATEGORY = "SAM3/prompts"

    def create_box(self, center_x, center_y, width, height, is_positive):
        """Create a box prompt"""
        box_prompt = {
            "box": [center_x, center_y, width, height],
            "label": is_positive
        }
        return (box_prompt,)


class SAM3CreatePoint:
    """
    Helper node to create a point prompt visually

    Use sliders to define a point for refinement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Point X (normalized 0-1)"
                }),
                "y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Point Y (normalized 0-1)"
                }),
                "is_foreground": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True for foreground, False for background"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_POINT_PROMPT",)
    RETURN_NAMES = ("point_prompt",)
    FUNCTION = "create_point"
    CATEGORY = "SAM3/prompts"

    def create_point(self, x, y, is_foreground):
        """Create a point prompt"""
        point_prompt = {
            "point": [x, y],
            "label": 1 if is_foreground else 0
        }
        return (point_prompt,)


class SAM3CombineBoxes:
    """
    Combine multiple box prompts into a single input

    Connect multiple SAM3CreateBox nodes to combine them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "box_1": ("SAM3_BOX_PROMPT",),
                "box_2": ("SAM3_BOX_PROMPT",),
                "box_3": ("SAM3_BOX_PROMPT",),
                "box_4": ("SAM3_BOX_PROMPT",),
                "box_5": ("SAM3_BOX_PROMPT",),
            }
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT",)
    RETURN_NAMES = ("boxes_prompt",)
    FUNCTION = "combine_boxes"
    CATEGORY = "SAM3/prompts"

    def combine_boxes(self, **kwargs):
        """Combine multiple box prompts"""
        boxes = []
        labels = []

        for i in range(1, 6):
            box_key = f"box_{i}"
            if box_key in kwargs and kwargs[box_key] is not None:
                box_data = kwargs[box_key]
                boxes.append(box_data["box"])
                labels.append(box_data["label"])

        combined = {
            "boxes": boxes,
            "labels": labels
        }
        return (combined,)


class SAM3CombinePoints:
    """
    Combine multiple point prompts into a single input

    Connect multiple SAM3CreatePoint nodes to combine them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "point_1": ("SAM3_POINT_PROMPT",),
                "point_2": ("SAM3_POINT_PROMPT",),
                "point_3": ("SAM3_POINT_PROMPT",),
                "point_4": ("SAM3_POINT_PROMPT",),
                "point_5": ("SAM3_POINT_PROMPT",),
                "point_6": ("SAM3_POINT_PROMPT",),
                "point_7": ("SAM3_POINT_PROMPT",),
                "point_8": ("SAM3_POINT_PROMPT",),
                "point_9": ("SAM3_POINT_PROMPT",),
                "point_10": ("SAM3_POINT_PROMPT",),
            }
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT",)
    RETURN_NAMES = ("points_prompt",)
    FUNCTION = "combine_points"
    CATEGORY = "SAM3/prompts"

    def combine_points(self, **kwargs):
        """Combine multiple point prompts"""
        points = []
        labels = []

        for i in range(1, 11):
            point_key = f"point_{i}"
            if point_key in kwargs and kwargs[point_key] is not None:
                point_data = kwargs[point_key]
                points.append(point_data["point"])
                labels.append(point_data["label"])

        combined = {
            "points": points,
            "labels": labels
        }
        return (combined,)


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3CreateBox": SAM3CreateBox,
    "SAM3CreatePoint": SAM3CreatePoint,
    "SAM3CombineBoxes": SAM3CombineBoxes,
    "SAM3CombinePoints": SAM3CombinePoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3CreateBox": "SAM3 Create Box",
    "SAM3CreatePoint": "SAM3 Create Point",
    "SAM3CombineBoxes": "SAM3 Combine Boxes",
    "SAM3CombinePoints": "SAM3 Combine Points",
}
