"""
SAM3Segmentation node - Performs segmentation using text prompts

This node uses ComfyUI's model_management for GPU/CPU handling.
"""
import torch
import numpy as np
import gc

import comfy.model_management

from .utils import (
    comfy_image_to_pil,
    pil_to_comfy_image,
    masks_to_comfy_mask,
    visualize_masks_on_image,
    tensor_to_list,
)
from .sam3_model_patcher import SAM3ModelPatcher


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
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model loaded from LoadSAM3Model node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to perform segmentation on"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum confidence score to keep detections. Lower threshold (0.2) works better with SAM3's presence scoring"
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., 'cat', 'person in red', 'car'",
                    "tooltip": "Describe what to segment using natural language (e.g., 'person', 'cat', 'red car', 'shoes')"
                }),
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Optional box prompts to include specific regions. Connect from SAM3CombineBoxes node."
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Optional box prompts to exclude specific regions. Connect from SAM3CombineBoxes node."
                }),
                "mask_prompt": ("MASK", {
                    "tooltip": "Optional mask to refine the segmentation. Useful for iterative refinement."
                }),
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

    def segment(self, sam3_model, image, confidence_threshold=0.2,
                text_prompt="", positive_boxes=None, negative_boxes=None,
                mask_prompt=None, max_detections=-1):
        """
        Perform SAM3 segmentation with text and box prompts

        Args:
            sam3_model: SAM3ModelPatcher from LoadSAM3Model node
            image: ComfyUI image tensor [B, H, W, C]
            confidence_threshold: Minimum confidence score for detections
            text_prompt: Optional text description of objects to segment
            positive_boxes: Optional positive box prompts
            negative_boxes: Optional negative box prompts
            mask_prompt: Optional mask prompt
            max_detections: Maximum number of detections to return

        Returns:
            Tuple of (masks, visualization, boxes_json, scores_json)
        """
        # Use ComfyUI's model management to load model to GPU
        comfy.model_management.load_models_gpu([sam3_model])

        # Access processor from the patcher
        processor = sam3_model.processor
        device = sam3_model.sam3_wrapper.device

        print(f"[SAM3] Running segmentation")
        if text_prompt:
            print(f"[SAM3]   Text prompt: '{text_prompt}'")
        if positive_boxes:
            print(f"[SAM3]   Positive boxes: {len(positive_boxes['boxes'])}")
        if negative_boxes:
            print(f"[SAM3]   Negative boxes: {len(negative_boxes['boxes'])}")
        if mask_prompt is not None:
            print(f"[SAM3]   Mask prompt provided")
        print(f"[SAM3] Confidence threshold: {confidence_threshold}")

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)
        img_w, img_h = pil_image.size
        print(f"[SAM3] Image size: {pil_image.size}")

        return self._segment_grounding(
            sam3_model, pil_image, img_w, img_h, confidence_threshold, text_prompt,
            positive_boxes, negative_boxes, mask_prompt, max_detections
        )

    def _segment_grounding(self, sam3_model, pil_image, img_w, img_h, confidence_threshold, text_prompt,
                           positive_boxes, negative_boxes, mask_prompt, max_detections):
        """
        Grounding mode - text-based detection with box prompts.
        """
        import json

        processor = sam3_model.processor
        device = sam3_model.sam3_wrapper.device

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
            print(f"[SAM3 DEBUG] WARNING: Model returned ZERO predictions - this is unusual!")
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
            empty_mask = torch.zeros(1, img_h, img_w)
            # Clean up state
            del state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        print(f"[SAM3] Found {len(masks)} detections above threshold {confidence_threshold}")

        # always sort by score
        if scores is not None and len(scores) > 0:
            print(f"[SAM3] Sorting {len(scores)} detections by score...")

            sorted_indices = torch.argsort(scores, descending=True)

            masks = masks[sorted_indices]
            boxes = boxes[sorted_indices] if boxes is not None else None
            scores = scores[sorted_indices] if scores is not None else None

        # Limit number of detections if specified
        if max_detections > 0 and len(masks) > max_detections:
            print(f"[SAM3] Limiting to top {max_detections} detections")
            # take top k since already sorted
            masks = masks[:max_detections]
            boxes = boxes[:max_detections] if boxes is not None else None
            scores = scores[:max_detections] if scores is not None else None

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
        boxes_json = json.dumps(boxes_list, indent=2)
        scores_json = json.dumps(scores_list, indent=2)

        print(f"[SAM3] Segmentation complete")
        print(f"[SAM3] Output: {len(comfy_masks)} masks")
        print(f"[SAM3 DEBUG] Final scores: {scores_list}")
        print(f"[SAM3 DEBUG] Mask output shape: {comfy_masks.shape}")

        # Clean up state to free GPU memory
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                "box_1": ("SAM3_BOX_PROMPT", {
                    "tooltip": "Connect box prompts from SAM3CreateBox nodes. Combines multiple boxes into a single prompt for SAM3Segmentation."
                }),
                "box_2": ("SAM3_BOX_PROMPT", {
                    "tooltip": "Connect box prompts from SAM3CreateBox nodes. Combines multiple boxes into a single prompt for SAM3Segmentation."
                }),
                "box_3": ("SAM3_BOX_PROMPT", {
                    "tooltip": "Connect box prompts from SAM3CreateBox nodes. Combines multiple boxes into a single prompt for SAM3Segmentation."
                }),
                "box_4": ("SAM3_BOX_PROMPT", {
                    "tooltip": "Connect box prompts from SAM3CreateBox nodes. Combines multiple boxes into a single prompt for SAM3Segmentation."
                }),
                "box_5": ("SAM3_BOX_PROMPT", {
                    "tooltip": "Connect box prompts from SAM3CreateBox nodes. Combines multiple boxes into a single prompt for SAM3Segmentation."
                }),
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
                "point_1": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_2": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_3": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_4": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_5": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_6": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_7": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_8": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_9": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
                "point_10": ("SAM3_POINT_PROMPT", {
                    "tooltip": "Connect point prompts from SAM3CreatePoint nodes. Combines multiple points into a single prompt for SAM3Segmentation."
                }),
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


class SAM3InteractiveSegmentation:
    """
    SAM2-style interactive segmentation using points and boxes.

    This node uses SAM3's inst_interactive_predictor for precise
    "segment exactly here" tasks. Unlike text-based grounding mode,
    this segments exactly at the provided point/box locations.

    NOTE: Currently experimental - may produce lower quality scores
    than expected. Use SAM3Segmentation for production workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model loaded from LoadSAM3Model node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to perform segmentation on"
                }),
            },
            "optional": {
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Foreground points - segment objects at these locations. Connect from SAM3CombinePoints or SAM3PointCollector."
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Background points - exclude these areas from segmentation. Connect from SAM3CombinePoints or SAM3PointCollector."
                }),
                "box": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Box prompt to constrain segmentation region. Only first box is used. Connect from SAM3CombineBoxes."
                }),
                "multimask_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, returns 3 mask candidates and selects the best one. If False, returns single mask."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("masks", "visualization", "boxes", "scores")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, sam3_model, image, positive_points=None, negative_points=None,
                box=None, multimask_output=True):
        """
        Perform SAM2-style interactive segmentation at point/box locations.

        Args:
            sam3_model: SAM3ModelPatcher from LoadSAM3Model node
            image: ComfyUI image tensor [B, H, W, C]
            positive_points: Foreground point prompts
            negative_points: Background point prompts
            box: Box prompt (first box used)
            multimask_output: Return multiple mask candidates

        Returns:
            Tuple of (masks, visualization, boxes_json, scores_json)
        """
        import json

        # Use ComfyUI's model management to load model to GPU
        comfy.model_management.load_models_gpu([sam3_model])

        processor = sam3_model.processor
        model = processor.model

        # Check if interactive predictor is available
        if model.inst_interactive_predictor is None:
            print("[SAM3 Interactive] ERROR: inst_interactive_predictor not available")
            print("[SAM3 Interactive] Make sure LoadSAM3Model was loaded with enable_inst_interactivity=True")
            pil_image = comfy_image_to_pil(image)
            img_w, img_h = pil_image.size
            empty_mask = torch.zeros(1, img_h, img_w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        print("[SAM3 Interactive] Using SAM2-style interactive segmentation")

        # Convert ComfyUI image to PIL
        pil_image = comfy_image_to_pil(image)
        img_w, img_h = pil_image.size
        print(f"[SAM3 Interactive] Image size: {pil_image.size}")

        # Set image and get backbone features
        state = processor.set_image(pil_image)

        # Debug: check if sam2_backbone_out exists
        backbone_out = state.get("backbone_out", {})
        if "sam2_backbone_out" in backbone_out:
            print("[SAM3 Interactive] sam2_backbone_out is available")
        else:
            print("[SAM3 Interactive] WARNING: sam2_backbone_out NOT available")
            print(f"[SAM3 Interactive]   backbone_out keys: {list(backbone_out.keys())}")

        # Collect all points
        all_points = []
        all_point_labels = []

        if positive_points is not None and len(positive_points.get('points', [])) > 0:
            for pt in positive_points['points']:
                # Convert normalized [0,1] to pixel coordinates
                px = pt[0] * img_w
                py = pt[1] * img_h
                all_points.append([px, py])
                all_point_labels.append(1)  # foreground
            print(f"[SAM3 Interactive] Added {len(positive_points['points'])} positive points")

        if negative_points is not None and len(negative_points.get('points', [])) > 0:
            for pt in negative_points['points']:
                px = pt[0] * img_w
                py = pt[1] * img_h
                all_points.append([px, py])
                all_point_labels.append(0)  # background
            print(f"[SAM3 Interactive] Added {len(negative_points['points'])} negative points")

        # Collect box (use first box if provided)
        box_array = None
        if box is not None and len(box.get('boxes', [])) > 0:
            b = box['boxes'][0]
            # Convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
            cx, cy, w, h = b
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            box_array = np.array([x1, y1, x2, y2])
            print(f"[SAM3 Interactive] Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # Prepare point arrays for predict_inst
        point_coords = np.array(all_points) if all_points else None
        point_labels = np.array(all_point_labels) if all_point_labels else None

        if point_coords is not None:
            print(f"[SAM3 Interactive] Points: {len(point_coords)}")
            print(f"[SAM3 Interactive]   Coords: {point_coords.tolist()}")
            print(f"[SAM3 Interactive]   Labels: {point_labels.tolist()}")

        if point_coords is None and box_array is None:
            print("[SAM3 Interactive] ERROR: No points or box provided. At least one prompt is required.")
            empty_mask = torch.zeros(1, img_h, img_w)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]")

        # Call predict_inst which uses inst_interactive_predictor
        masks_np, scores_np, low_res_masks = model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            multimask_output=multimask_output,
            normalize_coords=False,  # We already converted to pixel coords
        )

        print(f"[SAM3 Interactive] Prediction returned {masks_np.shape[0]} masks")
        print(f"[SAM3 Interactive]   Mask shape: {masks_np.shape}")
        print(f"[SAM3 Interactive]   Scores: {scores_np.tolist()}")

        # Select best mask (highest IoU score)
        best_idx = np.argmax(scores_np)
        best_mask = masks_np[best_idx]
        best_score = scores_np[best_idx]

        print(f"[SAM3 Interactive] Selected mask {best_idx} with score {best_score:.4f}")

        if best_score < 0.5:
            print(f"[SAM3 Interactive] WARNING: Low confidence score ({best_score:.4f})")
            print(f"[SAM3 Interactive] This is a known issue - interactive mode may not work correctly yet")

        # Convert to torch tensors
        masks = torch.from_numpy(best_mask).unsqueeze(0).float()  # [1, H, W]
        scores = torch.tensor([best_score])

        # Compute bounding box from mask
        mask_coords = torch.where(masks[0] > 0)
        if len(mask_coords[0]) > 0:
            y1 = mask_coords[0].min().item()
            y2 = mask_coords[0].max().item()
            x1 = mask_coords[1].min().item()
            x2 = mask_coords[1].max().item()
            boxes = torch.tensor([[x1, y1, x2, y2]]).float()
        else:
            boxes = torch.zeros(1, 4)

        # Convert to ComfyUI format
        comfy_masks = masks_to_comfy_mask(masks)

        # Create visualization
        vis_image = visualize_masks_on_image(
            pil_image, masks, boxes, scores, alpha=0.5
        )
        vis_tensor = pil_to_comfy_image(vis_image)

        # Format outputs
        boxes_list = tensor_to_list(boxes)
        scores_list = tensor_to_list(scores)
        boxes_json = json.dumps(boxes_list, indent=2)
        scores_json = json.dumps(scores_list, indent=2)

        print(f"[SAM3 Interactive] Segmentation complete")

        # Cleanup
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (comfy_masks, vis_tensor, boxes_json, scores_json)


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3InteractiveSegmentation": SAM3InteractiveSegmentation,
    "SAM3CreateBox": SAM3CreateBox,
    "SAM3CreatePoint": SAM3CreatePoint,
    "SAM3CombineBoxes": SAM3CombineBoxes,
    "SAM3CombinePoints": SAM3CombinePoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3InteractiveSegmentation": "SAM3 Interactive Segmentation (Experimental)",
    "SAM3CreateBox": "SAM3 Create Box",
    "SAM3CreatePoint": "SAM3 Create Point",
    "SAM3CombineBoxes": "SAM3 Combine Boxes",
    "SAM3CombinePoints": "SAM3 Combine Points",
}
