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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": (positive_points, negative_points)
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


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SAM3PointCollector": SAM3PointCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3PointCollector": "SAM3 Point Collector",
}
