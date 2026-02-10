"""
SAM3 Point Collector - Interactive Point Selection

Point editor widget adapted from ComfyUI-KJNodes
Original: https://github.com/kijai/ComfyUI-KJNodes
Author: kijai
License: Apache 2.0

Modifications for SAM3:
- Removed bounding box functionality
- Simplified to positive/negative points only
- Outputs point arrays for use with SAM3Segmentation node
"""

import torch
import numpy as np
import json
import io
import base64
from PIL import Image


class SAM3PointCollector:
    """
    Interactive Point Collector for SAM3

    Displays image canvas in the node where users can click to add:
    - Positive points (Left-click) - green circles
    - Negative points (Shift+Left-click or Right-click) - red circles

    Outputs point arrays to feed into SAM3Segmentation node.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Left-click to add positive points (green), Shift+Left-click or Right-click to add negative points (red). Points are automatically normalized to image dimensions."
                }),
                "points_store": ("STRING", {"multiline": False, "default": "{}"}),
                "coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT")
    RETURN_NAMES = ("positive_points", "negative_points")
    FUNCTION = "collect_points"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, points_store, coordinates, neg_coordinates):
        # Return hash based on actual point content, not object identity
        # This ensures downstream nodes don't re-run when points haven't changed
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        result = h.hexdigest()
        print(f"[IS_CHANGED DEBUG] SAM3PointCollector: shape={image.shape}, coords={coordinates}, neg_coords={neg_coordinates}")
        print(f"[IS_CHANGED DEBUG] SAM3PointCollector: returning hash={result}")
        return result

    def collect_points(self, image, points_store, coordinates, neg_coordinates):
        """
        Collect points from interactive canvas

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            points_store: Combined JSON storage (hidden widget)
            coordinates: Positive points JSON (hidden widget)
            neg_coordinates: Negative points JSON (hidden widget)

        Returns:
            Tuple of (positive_points, negative_points) as separate SAM3_POINTS_PROMPT outputs
        """
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3PointCollector._cache:
            cached = SAM3PointCollector._cache[cache_key]
            print(f"[SAM3 Point Collector] CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached  # Return the SAME objects
            }

        print(f"[SAM3 Point Collector] CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse coordinates from JSON
        try:
            pos_coords = json.loads(coordinates) if coordinates and coordinates.strip() else []
            neg_coords = json.loads(neg_coordinates) if neg_coordinates and neg_coordinates.strip() else []
        except json.JSONDecodeError:
            pos_coords = []
            neg_coords = []

        print(f"[SAM3 Point Collector] Collected {len(pos_coords)} positive, {len(neg_coords)} negative points")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        print(f"[SAM3 Point Collector] Image dimensions: {img_width}x{img_height}")

        # Convert to SAM3 point format - separate positive and negative outputs
        # SAM3 expects normalized coordinates (0-1), so divide by image dimensions
        positive_points = {"points": [], "labels": []}
        negative_points = {"points": [], "labels": []}

        # Add positive points (label = 1) - normalize to 0-1
        for p in pos_coords:
            normalized_x = p['x'] / img_width
            normalized_y = p['y'] / img_height
            positive_points["points"].append([normalized_x, normalized_y])
            positive_points["labels"].append(1)
            print(f"[SAM3 Point Collector]   Positive point: ({p['x']:.1f}, {p['y']:.1f}) -> ({normalized_x:.3f}, {normalized_y:.3f})")

        # Add negative points (label = 0) - normalize to 0-1
        for n in neg_coords:
            normalized_x = n['x'] / img_width
            normalized_y = n['y'] / img_height
            negative_points["points"].append([normalized_x, normalized_y])
            negative_points["labels"].append(0)
            print(f"[SAM3 Point Collector]   Negative point: ({n['x']:.1f}, {n['y']:.1f}) -> ({normalized_x:.3f}, {normalized_y:.3f})")

        print(f"[SAM3 Point Collector] Output: {len(positive_points['points'])} positive, {len(negative_points['points'])} negative")

        # Cache the result
        result = (positive_points, negative_points)
        SAM3PointCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


class SAM3BBoxCollector:
    """
    Interactive BBox Collector for SAM3

    Displays image canvas in the node where users can click and drag to add:
    - Positive bounding boxes (Left-click and drag) - cyan rectangles
    - Negative bounding boxes (Shift+Left-click and drag or Right-click and drag) - red rectangles

    Outputs bbox arrays to feed into SAM3Segmentation node.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Click and drag to draw positive bboxes (cyan), Shift+Click/Right-click and drag to draw negative bboxes (red). Bounding boxes are automatically normalized to image dimensions."
                }),
                "bboxes": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_bboxes": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("positive_bboxes", "negative_bboxes")
    FUNCTION = "collect_bboxes"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, bboxes, neg_bboxes):
        # Return hash based on actual bbox content, not object identity
        # This ensures downstream nodes don't re-run when bboxes haven't changed
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        result = h.hexdigest()
        print(f"[IS_CHANGED DEBUG] SAM3BBoxCollector: shape={image.shape}, bboxes={bboxes}, neg_bboxes={neg_bboxes}")
        print(f"[IS_CHANGED DEBUG] SAM3BBoxCollector: returning hash={result}")
        return result

    def collect_bboxes(self, image, bboxes, neg_bboxes):
        """
        Collect bounding boxes from interactive canvas

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            bboxes: Positive BBoxes JSON array (hidden widget)
            neg_bboxes: Negative BBoxes JSON array (hidden widget)

        Returns:
            Tuple of (positive_bboxes, negative_bboxes) as separate SAM3_BOXES_PROMPT outputs
        """
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3BBoxCollector._cache:
            cached = SAM3BBoxCollector._cache[cache_key]
            print(f"[SAM3 BBox Collector] CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached  # Return the SAME objects
            }

        print(f"[SAM3 BBox Collector] CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse bboxes from JSON
        try:
            pos_bbox_list = json.loads(bboxes) if bboxes and bboxes.strip() else []
            neg_bbox_list = json.loads(neg_bboxes) if neg_bboxes and neg_bboxes.strip() else []
        except json.JSONDecodeError:
            pos_bbox_list = []
            neg_bbox_list = []

        print(f"[SAM3 BBox Collector] Collected {len(pos_bbox_list)} positive, {len(neg_bbox_list)} negative bboxes")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        print(f"[SAM3 BBox Collector] Image dimensions: {img_width}x{img_height}")

        # Convert to SAM3_BOXES_PROMPT format with boxes and labels
        positive_boxes = []
        positive_labels = []
        negative_boxes = []
        negative_labels = []

        # Add positive bboxes (label = True)
        for bbox in pos_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox['x1'] / img_width
            y1_norm = bbox['y1'] / img_height
            x2_norm = bbox['x2'] / img_width
            y2_norm = bbox['y2'] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            positive_boxes.append([center_x, center_y, width, height])
            positive_labels.append(True)  # Positive boxes
            print(f"[SAM3 BBox Collector]   Positive BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})")

        # Add negative bboxes (label = False)
        for bbox in neg_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox['x1'] / img_width
            y1_norm = bbox['y1'] / img_height
            x2_norm = bbox['x2'] / img_width
            y2_norm = bbox['y2'] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            negative_boxes.append([center_x, center_y, width, height])
            negative_labels.append(False)  # Negative boxes
            print(f"[SAM3 BBox Collector]   Negative BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})")

        print(f"[SAM3 BBox Collector] Output: {len(positive_boxes)} positive, {len(negative_boxes)} negative bboxes")

        # Format as SAM3_BOXES_PROMPT (dict with 'boxes' and 'labels' keys)
        positive_prompt = {
            "boxes": positive_boxes,
            "labels": positive_labels
        }
        negative_prompt = {
            "boxes": negative_boxes,
            "labels": negative_labels
        }

        # Cache the result
        result = (positive_prompt, negative_prompt)
        SAM3BBoxCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


class SAM3MultiRegionCollector:
    """
    Interactive Multi-Region Collector for SAM3

    Displays image canvas in the node where users can:
    - Click/Right-click: Add positive/negative POINTS
    - Shift + Click/Drag: Add positive/negative BOXES

    Supports multiple prompt regions via tab bar.
    Each prompt region has its own set of points and boxes.

    Outputs a list of prompts for multi-object segmentation.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Click to add points, Shift+drag to draw boxes. Use tab bar to manage multiple prompt regions."
                }),
                "multi_prompts_store": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_MULTI_PROMPTS",)
    RETURN_NAMES = ("multi_prompts",)
    FUNCTION = "collect_prompts"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, image, multi_prompts_store):
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        return h.hexdigest()

    def collect_prompts(self, image, multi_prompts_store):
        """
        Collect multiple prompt regions from interactive canvas.

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            multi_prompts_store: JSON string containing all prompt regions

        Returns:
            List of prompt dicts, each with positive/negative points/boxes
        """
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        cache_key = h.hexdigest()

        # Check cache
        if cache_key in SAM3MultiRegionCollector._cache:
            cached = SAM3MultiRegionCollector._cache[cache_key]
            print(f"[SAM3 Multi-Region] CACHE HIT - returning cached result for key={cache_key[:8]}")
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached
            }

        print(f"[SAM3 Multi-Region] CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse stored prompts
        try:
            raw_prompts = json.loads(multi_prompts_store) if multi_prompts_store.strip() else []
        except json.JSONDecodeError:
            raw_prompts = []

        img_height, img_width = image.shape[1], image.shape[2]
        print(f"[SAM3 Multi-Region] Image dimensions: {img_width}x{img_height}")
        print(f"[SAM3 Multi-Region] Processing {len(raw_prompts)} prompt regions")

        # Convert to normalized output format
        multi_prompts = []
        for idx, raw_prompt in enumerate(raw_prompts):
            prompt = {
                "id": idx,
                "positive_points": {"points": [], "labels": []},
                "negative_points": {"points": [], "labels": []},
                "positive_boxes": {"boxes": [], "labels": []},
                "negative_boxes": {"boxes": [], "labels": []},
            }

            # Normalize positive points
            for pt in raw_prompt.get("positive_points", []):
                norm_x = pt["x"] / img_width
                norm_y = pt["y"] / img_height
                prompt["positive_points"]["points"].append([norm_x, norm_y])
                prompt["positive_points"]["labels"].append(1)

            # Normalize negative points
            for pt in raw_prompt.get("negative_points", []):
                norm_x = pt["x"] / img_width
                norm_y = pt["y"] / img_height
                prompt["negative_points"]["points"].append([norm_x, norm_y])
                prompt["negative_points"]["labels"].append(0)

            # Normalize positive boxes (convert x1,y1,x2,y2 to center format)
            for box in raw_prompt.get("positive_boxes", []):
                x1_norm = box["x1"] / img_width
                y1_norm = box["y1"] / img_height
                x2_norm = box["x2"] / img_width
                y2_norm = box["y2"] / img_height
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                w = x2_norm - x1_norm
                h = y2_norm - y1_norm
                prompt["positive_boxes"]["boxes"].append([cx, cy, w, h])
                prompt["positive_boxes"]["labels"].append(True)

            # Normalize negative boxes
            for box in raw_prompt.get("negative_boxes", []):
                x1_norm = box["x1"] / img_width
                y1_norm = box["y1"] / img_height
                x2_norm = box["x2"] / img_width
                y2_norm = box["y2"] / img_height
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                w = x2_norm - x1_norm
                h = y2_norm - y1_norm
                prompt["negative_boxes"]["boxes"].append([cx, cy, w, h])
                prompt["negative_boxes"]["labels"].append(False)

            # Count items for logging
            pos_pts = len(prompt["positive_points"]["points"])
            neg_pts = len(prompt["negative_points"]["points"])
            pos_boxes = len(prompt["positive_boxes"]["boxes"])
            neg_boxes = len(prompt["negative_boxes"]["boxes"])
            print(f"[SAM3 Multi-Region]   Prompt {idx}: {pos_pts} pos pts, {neg_pts} neg pts, {pos_boxes} pos boxes, {neg_boxes} neg boxes")

            # Only include prompts with content
            if (prompt["positive_points"]["points"] or
                prompt["negative_points"]["points"] or
                prompt["positive_boxes"]["boxes"] or
                prompt["negative_boxes"]["boxes"]):
                multi_prompts.append(prompt)

        print(f"[SAM3 Multi-Region] Output: {len(multi_prompts)} non-empty prompts")

        # Cache and return
        result = (multi_prompts,)
        SAM3MultiRegionCollector._cache[cache_key] = result
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SAM3PointCollector": SAM3PointCollector,
    "SAM3BBoxCollector": SAM3BBoxCollector,
    "SAM3MultiRegionCollector": SAM3MultiRegionCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3PointCollector": "SAM3 Point Collector",
    "SAM3BBoxCollector": "SAM3 BBox Collector 📦",
    "SAM3MultiRegionCollector": "SAM3 Multi-Region Collector 🎯",
}
