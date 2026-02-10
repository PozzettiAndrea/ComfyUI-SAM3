"""
SAM3 Server - API endpoints for interactive point selection

Provides endpoints for the SAM3 interactive editor:
- /sam3/prepare: Load model and image
- /sam3/detect: Run SAM3 with point prompts
"""

import io
import json
import os
import threading
import numpy as np
import torch
from PIL import Image
from aiohttp import web
from server import PromptServer
import folder_paths

# Global state for SAM3 predictor
sam3_predictor = None
sam3_lock = threading.Condition()


def load_image_from_path(filename, file_type, subfolder=""):
    """Load image from ComfyUI storage"""
    # Get the appropriate directory based on type
    if file_type == "input":
        input_dir = folder_paths.get_input_directory()
    elif file_type == "temp":
        input_dir = folder_paths.get_temp_directory()
    elif file_type == "output":
        input_dir = folder_paths.get_output_directory()
    else:
        input_dir = folder_paths.get_input_directory()

    # Build full path
    if subfolder:
        full_path = os.path.join(input_dir, subfolder, filename)
    else:
        full_path = os.path.join(input_dir, filename)

    print(f"[SAM3 Server] Loading image from: {full_path}")

    # Load image
    image = Image.open(full_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


@PromptServer.instance.routes.post("/sam3/prepare")
async def sam3_prepare(request):
    """
    Load SAM3 model and prepare image for interactive editing

    Request JSON:
    {
        "sam_model_name": "auto",
        "filename": "image.png",
        "type": "temp",
        "subfolder": ""
    }
    """
    global sam3_predictor

    try:
        data = await request.json()
        filename = data.get("filename", "")
        file_type = data.get("type", "temp")
        subfolder = data.get("subfolder", "")

        print(f"[SAM3 Server] Preparing SAM3 for image: {filename}")

        # Load image
        image = load_image_from_path(filename, file_type, subfolder)

        # Load SAM3 model if not already loaded
        # For now, we'll load the model on-demand when /sam3/detect is called
        # This is because the model is managed by LoadSAM3Model node

        with sam3_lock:
            sam3_predictor = {
                "image": image,
                "filename": filename,
                "type": file_type,
                "subfolder": subfolder,
            }
            print(f"[SAM3 Server] Image prepared: {image.size}")

        return web.Response(status=200)

    except Exception as e:
        print(f"[SAM3 Server] Error in /sam3/prepare: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))


@PromptServer.instance.routes.post("/sam3/detect")
async def sam3_detect(request):
    """
    Run SAM3 detection with point prompts

    Request JSON:
    {
        "positive_points": [[x1, y1], [x2, y2], ...],
        "negative_points": [[x3, y3], ...],
        "threshold": 0.7
    }

    Returns: PNG image of the mask
    """
    global sam3_predictor

    try:
        data = await request.json()
        positive_points = data.get("positive_points", [])
        negative_points = data.get("negative_points", [])
        threshold = data.get("threshold", 0.7)

        print(f"[SAM3 Server] Detecting with {len(positive_points)} positive and {len(negative_points)} negative points")

        if sam3_predictor is None:
            return web.Response(status=400, text="SAM3 not prepared. Call /sam3/prepare first.")

        # Get the stored image
        image = sam3_predictor.get("image")
        if image is None:
            return web.Response(status=400, text="No image loaded")

        # Relative import from package root (sam3_server.py is loaded via __init__.py)
        from .nodes.sam3_lib.model_builder import build_sam3_image_model
        from .nodes.sam3_lib.model.sam3_image_processor import Sam3Processor

        # Build model (cached globally if possible)
        if "model" not in sam3_predictor:
            print(f"[SAM3 Server] Loading SAM3 model...")
            import os
            import folder_paths
            ckpt_path = os.path.join(folder_paths.models_dir, "sam3", "sam3.pt")
            
            if os.path.exists(ckpt_path):
                print(f"[SAM3 Server] Found local model at {ckpt_path}")
                model = build_sam3_image_model(checkpoint_path=ckpt_path, load_from_HF=False)
            else:
                model = build_sam3_image_model()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()

            processor = Sam3Processor(model, device=device)

            sam3_predictor["model"] = model
            sam3_predictor["processor"] = processor
            sam3_predictor["device"] = device
            print(f"[SAM3 Server] SAM3 model loaded on {device}")

        processor = sam3_predictor["processor"]
        device = sam3_predictor["device"]

        # Set image and extract features
        print(f"[SAM3 Server] Setting image...")
        state = processor.set_image(image)

        # Prepare points and labels
        all_points = []
        all_labels = []

        # Add positive points (label=1)
        for point in positive_points:
            all_points.append(point)
            all_labels.append(1)

        # Add negative points (label=0)
        for point in negative_points:
            all_points.append(point)
            all_labels.append(0)

        print(f"[SAM3 Server] Total points: {len(all_points)}")

        # Run detection with points if we have any
        if len(all_points) > 0:
            # Normalize pixel coordinates to [0, 1] range
            img_width, img_height = image.size
            normalized_points = [[x / img_width, y / img_height] for x, y in all_points]

            print(f"[SAM3 Server] Running point prompt detection...")
            state = processor.add_point_prompt(normalized_points, all_labels, state)

            # Get masks from state
            masks = state.get("masks", None)
            scores = state.get("scores", None)

            if masks is None or len(masks) == 0:
                print(f"[SAM3 Server] No masks generated")
                # Return empty mask
                mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                print(f"[SAM3 Server] Got {len(masks)} masks, scores: {scores}")

                # Filter by threshold if scores available
                if scores is not None:
                    valid_indices = scores > threshold
                    if valid_indices.sum() > 0:
                        masks = masks[valid_indices]
                        scores = scores[valid_indices]
                        print(f"[SAM3 Server] Filtered to {len(masks)} masks above threshold {threshold}")
                    else:
                        # Take best mask if none above threshold
                        best_idx = torch.argmax(scores)
                        masks = masks[best_idx:best_idx+1]
                        scores = scores[best_idx:best_idx+1]
                        print(f"[SAM3 Server] No masks above threshold, using best mask with score {scores[0]:.3f}")

                # Combine multiple masks using bitwise OR
                mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
                for mask in masks:
                    # Convert tensor to numpy
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask

                    # Ensure correct shape
                    if mask_np.ndim == 3:
                        mask_np = mask_np[0]  # Remove batch dimension

                    # Resize if needed
                    if mask_np.shape != (image.height, image.width):
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray((mask_np > 0).astype(np.uint8) * 255)
                        mask_pil = mask_pil.resize((image.width, image.height), PILImage.NEAREST)
                        mask_np = np.array(mask_pil) > 0

                    # Combine with OR
                    mask_array = np.logical_or(mask_array, mask_np > 0).astype(np.uint8) * 255

        else:
            print(f"[SAM3 Server] No points provided, returning empty mask")
            mask_array = np.zeros((image.height, image.width), dtype=np.uint8)

        # Convert mask to PIL Image
        mask_image = Image.fromarray(mask_array, mode='L')

        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        mask_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        print(f"[SAM3 Server] Returning mask PNG")

        return web.Response(
            body=img_buffer.getvalue(),
            headers={'Content-Type': 'image/png'}
        )

    except Exception as e:
        print(f"[SAM3 Server] Error in /sam3/detect: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))


print("[SAM3 Server] Registered routes: /sam3/prepare, /sam3/detect")
