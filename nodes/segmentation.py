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
    tensor_to_list
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
                "text_prompt": ("STRING", {
                    "default": "person",
                    "multiline": False,
                    "placeholder": "e.g., 'cat', 'person in red', 'car'"
                }),
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

    def segment(self, sam3_model, image, text_prompt, confidence_threshold=0.5, max_detections=-1):
        """
        Perform SAM3 segmentation

        Args:
            sam3_model: Model dict from LoadSAM3Model node
            image: ComfyUI image tensor [B, H, W, C]
            text_prompt: Text description of objects to segment
            confidence_threshold: Minimum confidence score for detections
            max_detections: Maximum number of detections to return

        Returns:
            Tuple of (masks, visualization, boxes_json, scores_json)
        """
        # Extract processor from model dict
        processor = sam3_model["processor"]
        device = sam3_model["device"]

        print(f"[SAM3] Running segmentation with prompt: '{text_prompt}'")
        print(f"[SAM3] Confidence threshold: {confidence_threshold}")

        # Convert ComfyUI image to PIL
        print(f"[SAM3 DEBUG] Input image shape: {image.shape}, dtype: {image.dtype}")
        print(f"[SAM3 DEBUG] Input image range: [{image.min():.3f}, {image.max():.3f}]")
        pil_image = comfy_image_to_pil(image)
        print(f"[SAM3] Image size: {pil_image.size}, mode: {pil_image.mode}")

        # Update confidence threshold
        processor.set_confidence_threshold(confidence_threshold)

        # Set image (extracts features)
        print(f"[SAM3] Extracting image features...")
        state = processor.set_image(pil_image)
        print(f"[SAM3 DEBUG] State keys after set_image: {list(state.keys())}")

        # Check if text prompt is provided
        if not text_prompt or not text_prompt.strip():
            print(f"[SAM3] Warning: Empty text prompt, returning empty results")
            # Return empty results
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        # Set text prompt
        print(f"[SAM3] Running detection with text prompt...")
        state = processor.set_text_prompt(text_prompt.strip(), state)

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

        return (comfy_masks, vis_tensor, boxes_json, scores_json)


class SAM3GeometricRefine:
    """
    Node to refine SAM3 results with geometric prompts (boxes)

    This node is optional and can be used to add positive or negative
    box prompts to refine the segmentation results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "box_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box center X (normalized 0-1)"
                }),
                "box_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box center Y (normalized 0-1)"
                }),
                "box_w": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box width (normalized 0-1)"
                }),
                "box_h": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Box height (normalized 0-1)"
                }),
                "is_positive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True for positive prompt, False for negative"
                }),
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
                    "placeholder": "Optional: combine with text prompt"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("masks", "visualization", "boxes", "scores")
    FUNCTION = "refine"
    CATEGORY = "SAM3"

    def refine(self, sam3_model, image, box_x, box_y, box_w, box_h,
               is_positive=True, confidence_threshold=0.5, text_prompt=""):
        """
        Refine segmentation with geometric box prompt

        Args:
            sam3_model: Model dict from LoadSAM3Model node
            image: ComfyUI image tensor
            box_x, box_y, box_w, box_h: Box parameters (normalized 0-1)
            is_positive: Whether this is a positive or negative prompt
            confidence_threshold: Minimum confidence score
            text_prompt: Optional text prompt to combine with box

        Returns:
            Tuple of (masks, visualization, boxes_json, scores_json)
        """
        processor = sam3_model["processor"]

        print(f"[SAM3] Running geometric refinement")
        print(f"[SAM3] Box: ({box_x:.2f}, {box_y:.2f}, {box_w:.2f}, {box_h:.2f})")
        print(f"[SAM3] Positive prompt: {is_positive}")

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)

        # Update confidence threshold
        processor.set_confidence_threshold(confidence_threshold)

        # Set image
        state = processor.set_image(pil_image)

        # Add text prompt if provided
        if text_prompt and text_prompt.strip():
            print(f"[SAM3] Adding text prompt: '{text_prompt}'")
            state = processor.set_text_prompt(text_prompt.strip(), state)

        # Add geometric prompt
        box = [box_x, box_y, box_w, box_h]
        state = processor.add_geometric_prompt(box, is_positive, state)

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No detections found")
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        print(f"[SAM3] Found {len(masks)} detections")

        # Convert to ComfyUI formats
        comfy_masks = masks_to_comfy_mask(masks)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        # Convert to JSON
        import json
        boxes_list = tensor_to_list(boxes) if boxes is not None else []
        scores_list = tensor_to_list(scores) if scores is not None else []
        boxes_json = json.dumps(boxes_list, indent=2)
        scores_json = json.dumps(scores_list, indent=2)

        print(f"[SAM3] Refinement complete")

        return (comfy_masks, vis_tensor, boxes_json, scores_json)


class SAM3PointRefine:
    """
    Node for interactive point-based refinement with SAM3

    Use this to add foreground/background point prompts for precise segmentation.
    Points are specified in normalized coordinates (0-1).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "point_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Point X coordinate (normalized 0-1)"
                }),
                "point_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Point Y coordinate (normalized 0-1)"
                }),
                "is_foreground": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True for foreground point, False for background"
                }),
                "multimask_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return 3 masks for ambiguous prompts"
                }),
            },
            "optional": {
                "additional_points": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Additional points as JSON: [[x,y,label], ...]\nLabel: 1=foreground, 0=background"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("masks", "visualization", "scores")
    FUNCTION = "refine_with_points"
    CATEGORY = "SAM3"

    def refine_with_points(self, sam3_model, image, point_x, point_y,
                           is_foreground=True, multimask_output=False,
                           additional_points=""):
        """Point-based refinement using SAM3's grounding model"""
        import json
        processor = sam3_model["processor"]

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)

        # Set image
        state = processor.set_image(pil_image)

        # Build point list
        points = [[point_x, point_y]]
        labels = [1 if is_foreground else 0]

        # Parse additional points if provided
        if additional_points and additional_points.strip():
            try:
                extra_points = json.loads(additional_points.strip())
                for pt in extra_points:
                    if len(pt) == 3:  # [x, y, label]
                        points.append([pt[0], pt[1]])
                        labels.append(pt[2])
            except json.JSONDecodeError:
                print(f"[SAM3] Warning: Could not parse additional_points JSON")

        print(f"[SAM3] Running point refinement with {len(points)} points")

        # Use grounding approach for point-based refinement
        state = processor.add_point_prompt(points, labels, state)

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No masks generated")
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]")

        print(f"[SAM3] Generated {len(masks)} mask(s)")

        # Convert to ComfyUI formats
        comfy_masks = masks_to_comfy_mask(masks)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        scores_list = tensor_to_list(scores) if scores is not None else []
        scores_json = json.dumps(scores_list, indent=2)

        return (comfy_masks, vis_tensor, scores_json)


class SAM3MaskRefine:
    """
    Node for mask-to-mask refinement

    Takes an existing mask and refines it using SAM3's interactive predictor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "input_mask": ("MASK",),
                "multimask_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return 3 masks for ambiguous prompts"
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("masks", "visualization", "scores")
    FUNCTION = "refine_with_mask"
    CATEGORY = "SAM3"

    def refine_with_mask(self, sam3_model, image, input_mask, multimask_output=False):
        """Refine an existing mask using SAM3's grounding model"""
        import json
        processor = sam3_model["processor"]

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)

        # Set image
        state = processor.set_image(pil_image)

        # Convert mask to tensor for grounding approach
        if not isinstance(input_mask, torch.Tensor):
            input_mask = torch.from_numpy(input_mask)

        # Ensure mask is on correct device
        input_mask = input_mask.to(processor.device)

        print(f"[SAM3] Running mask refinement")
        print(f"[SAM3] Input mask shape: {input_mask.shape}")

        # Use grounding approach for mask-based refinement
        state = processor.add_mask_prompt(input_mask, state)

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No masks generated")
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]")

        print(f"[SAM3] Generated {len(masks)} mask(s)")

        # Convert to ComfyUI formats
        comfy_masks = masks_to_comfy_mask(masks)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        scores_list = tensor_to_list(scores) if scores is not None else []
        scores_json = json.dumps(scores_list, indent=2)

        return (comfy_masks, vis_tensor, scores_json)


class SAM3MultiPrompt:
    """
    Node for combining multiple geometric prompts

    Supports multiple boxes and/or points simultaneously for complex segmentation tasks.
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
                    "display": "slider"
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "placeholder": "Optional text prompt"
                }),
                "boxes_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Boxes as JSON: [[cx,cy,w,h,label], ...]\nLabel: true=positive, false=negative"
                }),
                "points_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Points as JSON: [[x,y,label], ...]\nLabel: 1=foreground, 0=background"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("masks", "visualization", "boxes", "scores")
    FUNCTION = "multi_prompt_segment"
    CATEGORY = "SAM3"

    def multi_prompt_segment(self, sam3_model, image, confidence_threshold=0.2,
                            text_prompt="", boxes_json="", points_json=""):
        """Segment with multiple geometric prompts"""
        import json
        processor = sam3_model["processor"]

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)

        # Update confidence threshold
        processor.set_confidence_threshold(confidence_threshold)

        # Set image
        state = processor.set_image(pil_image)

        # Add text prompt if provided
        if text_prompt and text_prompt.strip():
            print(f"[SAM3] Adding text prompt: '{text_prompt}'")
            state = processor.set_text_prompt(text_prompt.strip(), state)

        # Parse and add boxes
        if boxes_json and boxes_json.strip():
            try:
                boxes_data = json.loads(boxes_json.strip())
                boxes = []
                labels = []
                for box_entry in boxes_data:
                    if len(box_entry) == 5:  # [cx, cy, w, h, label]
                        boxes.append(box_entry[:4])
                        labels.append(box_entry[4])

                if boxes:
                    print(f"[SAM3] Adding {len(boxes)} box prompts")
                    state = processor.add_multiple_box_prompts(boxes, labels, state)
            except json.JSONDecodeError:
                print(f"[SAM3] Warning: Could not parse boxes_json")

        # Parse and add points
        if points_json and points_json.strip():
            try:
                points_data = json.loads(points_json.strip())
                points = []
                labels = []
                for pt_entry in points_data:
                    if len(pt_entry) == 3:  # [x, y, label]
                        points.append(pt_entry[:2])
                        labels.append(pt_entry[2])

                if points:
                    print(f"[SAM3] Adding {len(points)} point prompts")
                    state = processor.add_point_prompt(points, labels, state)
            except json.JSONDecodeError:
                print(f"[SAM3] Warning: Could not parse points_json")

        # Extract results
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No detections found")
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        print(f"[SAM3] Found {len(masks)} detections")

        # Convert to ComfyUI formats
        comfy_masks = masks_to_comfy_mask(masks)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        boxes_list = tensor_to_list(boxes) if boxes is not None else []
        scores_list = tensor_to_list(scores) if scores is not None else []
        boxes_json_out = json.dumps(boxes_list, indent=2)
        scores_json_out = json.dumps(scores_list, indent=2)

        return (comfy_masks, vis_tensor, boxes_json_out, scores_json_out)


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3GeometricRefine": SAM3GeometricRefine,
    "SAM3PointRefine": SAM3PointRefine,
    "SAM3MaskRefine": SAM3MaskRefine,
    "SAM3MultiPrompt": SAM3MultiPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3GeometricRefine": "SAM3 Geometric Refine",
    "SAM3PointRefine": "SAM3 Point Refine",
    "SAM3MaskRefine": "SAM3 Mask Refine",
    "SAM3MultiPrompt": "SAM3 Multi-Prompt",
}
