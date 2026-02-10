"""
Utility functions for ComfyUI-SAM3 nodes
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def get_comfy_models_dir():
    """Get the ComfyUI models directory"""
    # Try to find ComfyUI root by going up from custom_nodes
    current = Path(__file__).parent.parent.absolute()  # ComfyUI-SAM3
    comfy_custom_nodes = current.parent  # custom_nodes
    comfy_root = comfy_custom_nodes.parent  # ComfyUI root

    models_dir = comfy_root / "models" / "sam3"
    models_dir.mkdir(parents=True, exist_ok=True)

    return str(models_dir)


def comfy_image_to_pil(image):
    """
    Convert ComfyUI image tensor to PIL Image

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        PIL Image
    """
    # ComfyUI images are [B, H, W, C] in range [0, 1]
    if isinstance(image, torch.Tensor):
        # Take first image if batch
        if image.dim() == 4:
            image = image[0]

        # Convert to numpy
        img_np = image.cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        img_np = (img_np * 255).astype(np.uint8)

        # Convert to PIL
        pil_image = Image.fromarray(img_np)
        return pil_image

    return image


def pil_to_comfy_image(pil_image):
    """
    Convert PIL Image to ComfyUI image tensor

    Args:
        pil_image: PIL Image

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert to numpy array
    img_np = np.array(pil_image).astype(np.float32)

    # Normalize to [0, 1]
    img_np = img_np / 255.0

    # Convert to tensor [H, W, C]
    img_tensor = torch.from_numpy(img_np)

    # Add batch dimension [1, H, W, C]
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def masks_to_comfy_mask(masks):
    """
    Convert SAM3 masks to ComfyUI mask format

    Args:
        masks: torch.Tensor [N, H, W] or [N, 1, H, W] binary masks

    Returns:
        ComfyUI mask tensor [N, H, W] in range [0, 1]
    """
    if isinstance(masks, torch.Tensor):
        # Ensure float type and range [0, 1]
        masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        return masks
    elif isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        return masks

    return masks


def visualize_masks_on_image(image, masks, boxes=None, scores=None, alpha=0.5):
    """
    Create visualization of masks overlaid on image

    Args:
        image: PIL Image or numpy array
        masks: torch.Tensor [N, H, W] binary masks
        boxes: Optional torch.Tensor [N, 4] bounding boxes in [x0, y0, x1, y1]
        scores: Optional torch.Tensor [N] confidence scores
        alpha: Transparency of mask overlay

    Returns:
        PIL Image with visualization
    """
    if isinstance(image, torch.Tensor):
        image = comfy_image_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))

    # Convert to numpy for processing
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize masks to image size if needed
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    # Create colored overlay
    np.random.seed(42)  # Consistent colors
    overlay = img_np.copy()

    for i, mask in enumerate(masks_np):
        # Squeeze extra dimensions (masks may be [1, H, W] or [H, W])
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Resize mask to image size if needed
        if mask.shape != img_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        # Random color for this mask
        color = np.random.rand(3)

        # Apply colored mask
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0.5,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    # Convert back to PIL
    result = Image.fromarray((overlay * 255).astype(np.uint8))

    # Draw boxes if provided
    if boxes is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)

        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes

        for i, box in enumerate(boxes_np):
            x0, y0, x1, y1 = box

            # Random color for this box (same seed for consistency)
            np.random.seed(42 + i)
            color_int = tuple((np.random.rand(3) * 255).astype(int).tolist())

            # Draw box
            draw.rectangle([x0, y0, x1, y1], outline=color_int, width=3)

            # Draw score if provided
            if scores is not None:
                score = scores[i] if isinstance(scores, (list, np.ndarray)) else scores[i].item()
                text = f"{score:.2f}"
                draw.text((x0, y0 - 15), text, fill=color_int)

    return result


def tensor_to_list(tensor):
    """Convert torch tensor to python list"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().tolist()
    return tensor


def ensure_model_on_device(sam3_model, target_device=None):
    """
    Ensure model is on the target device before inference

    Args:
        sam3_model: Model dict from LoadSAM3Model
        target_device: Target device (uses original_device if None)

    Returns:
        None (modifies model dict in place)
    """
    model = sam3_model["model"]
    processor = sam3_model["processor"]

    if target_device is None:
        target_device = sam3_model["original_device"]

    # Check if model is already on target device
    current_device = next(model.parameters()).device
    if str(current_device) != target_device:
        print(f"[SAM3] Moving model from {current_device} to {target_device}")
        model.to(target_device)
        processor.device = target_device
        sam3_model["device"] = target_device


def offload_model_if_needed(sam3_model):
    """
    Offload model to CPU if use_gpu_cache is False

    Args:
        sam3_model: Model dict from LoadSAM3Model

    Returns:
        None (modifies model dict in place)
    """
    use_gpu_cache = sam3_model.get("use_gpu_cache", True)

    if not use_gpu_cache:
        model = sam3_model["model"]
        processor = sam3_model["processor"]
        current_device = next(model.parameters()).device

        # Only offload if currently on GPU
        if "cuda" in str(current_device):
            print(f"[SAM3] Offloading model to CPU to free VRAM")
            model.to("cpu")
            processor.device = "cpu"
            sam3_model["device"] = "cpu"
            # Force garbage collection to free VRAM
            torch.cuda.empty_cache()
            import gc
            gc.collect()
