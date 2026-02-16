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
        ComfyUI mask tensor [N, H, W] in range [0, 1] on CPU
    """
    if isinstance(masks, torch.Tensor):
        # Ensure float type and range [0, 1]
        masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Move to CPU to ensure compatibility with downstream nodes
        return masks.cpu()
    elif isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Already on CPU since from numpy
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

    # Use torch on GPU for fast mask overlay, fall back to CPU torch
    if isinstance(masks, torch.Tensor):
        masks_t = masks
    else:
        masks_t = torch.from_numpy(np.asarray(masks))

    device = masks_t.device if masks_t.is_cuda else torch.device('cpu')
    img_t = torch.from_numpy(np.array(image)).to(device=device, dtype=torch.float32) / 255.0  # [H, W, 3]
    H, W = img_t.shape[:2]
    overlay = img_t.clone()

    # Pre-generate consistent colors
    rng = torch.Generator(device='cpu').manual_seed(42)
    colors = torch.rand(masks_t.shape[0], 3, generator=rng, device=device)

    for i in range(masks_t.shape[0]):
        mask = masks_t[i]
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Resize mask to image size if needed
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = torch.nn.functional.interpolate(
                mask[None, None].float(), size=(H, W), mode='nearest'
            )[0, 0]

        # Vectorized: single where over all 3 channels via broadcasting
        mask_3d = (mask > 0.5).unsqueeze(-1)  # [H, W, 1]
        color = colors[i]  # [3]
        overlay = torch.where(mask_3d, overlay * (1 - alpha) + color * alpha, overlay)

    # Convert back to PIL
    result_np = (overlay.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    result = Image.fromarray(result_np)

    # Draw boxes if provided
    if boxes is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)

        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes

        colors_np = (colors.cpu().numpy() * 255).astype(int)
        for i, box in enumerate(boxes_np):
            x0, y0, x1, y1 = box
            color_int = tuple(colors_np[i].tolist())

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


from contextlib import contextmanager
import gc


@contextmanager
def inference_context():
    """
    Context manager ensuring cleanup after inference.

    Usage:
        with inference_context():
            # ... inference code ...

    This ensures gc.collect() and soft_empty_cache() are called
    after inference, even if an exception occurs.
    """
    import comfy.model_management
    try:
        yield
    finally:
        gc.collect()
        comfy.model_management.soft_empty_cache()


def cleanup_gpu_memory():
    """
    Force GPU memory cleanup.

    Call this after inference to ensure VRAM is freed.
    """
    import comfy.model_management
    gc.collect()
    comfy.model_management.soft_empty_cache()
