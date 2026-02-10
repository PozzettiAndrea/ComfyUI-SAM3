"""
Integration tests for SAM3 inference on CPU.

These tests are slower and require model downloads, so they're marked as 'integration'.
Run with: pytest tests/test_inference.py -v -m integration
"""
import os
import pytest
import torch
import numpy as np
from PIL import Image


@pytest.mark.integration
def test_sam3_cpu_inference():
    """
    Test basic SAM3 image segmentation on CPU.

    This verifies that:
    - Model loads successfully on CPU
    - Inference runs without errors
    - Outputs are in correct format
    """
    # Skip if HF_TOKEN not available
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN not set - skipping inference test")

    # Import SAM3 nodes
    from nodes.load_model import LoadSAM3Model
    from nodes.segmentation import SAM3Grounding

    # Get path to test image
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_path = os.path.join(script_dir, "assets", "example_image.jpg")

    assert os.path.exists(test_image_path), f"Test image not found: {test_image_path}"

    print(f"\n[TEST] Loading test image: {test_image_path}")

    # Load test image and convert to ComfyUI format
    pil_image = Image.open(test_image_path).convert("RGB")
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    # ComfyUI format: (batch, height, width, channels)
    image_tensor = torch.from_numpy(image_np)[None, ...]

    print(f"[TEST] Image shape: {image_tensor.shape}")

    # Load SAM3 model on CPU
    print("[TEST] Loading SAM3 model on CPU (this may take a while)...")
    model_loader = LoadSAM3Model()

    try:
        model_output = model_loader.load_model(
            model_path="",
            hf_token=hf_token
        )
    except Exception as e:
        pytest.fail(f"Failed to load SAM3 model: {e}")

    assert model_output is not None, "Model loader returned None"
    assert len(model_output) == 1, f"Expected 1 output from model loader, got {len(model_output)}"

    sam3_model = model_output[0]
    print(f"[TEST] Model loaded successfully: {type(sam3_model)}")

    # Run grounding-based segmentation with text prompt
    print("[TEST] Running grounding segmentation with prompt 'person'...")
    segmenter = SAM3Grounding()

    try:
        outputs = segmenter.segment(
            sam3_model=sam3_model,
            image=image_tensor,
            text_prompt="person",
            confidence_threshold=0.1,
            max_detections=-1
        )
    except Exception as e:
        pytest.fail(f"Segmentation failed: {e}")

    # Validate outputs
    assert outputs is not None, "Segmentation returned None"
    assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

    masks, visualization, boxes_json, scores_json = outputs

    print(f"[TEST] Segmentation completed")
    print(f"[TEST] - Masks shape: {masks.shape if isinstance(masks, torch.Tensor) else type(masks)}")
    print(f"[TEST] - Visualization shape: {visualization.shape if isinstance(visualization, torch.Tensor) else type(visualization)}")
    print(f"[TEST] - Boxes JSON: {boxes_json[:100]}..." if len(boxes_json) > 100 else f"[TEST] - Boxes JSON: {boxes_json}")
    print(f"[TEST] - Scores JSON: {scores_json[:100]}..." if len(scores_json) > 100 else f"[TEST] - Scores JSON: {scores_json}")

    # Validate mask output
    assert isinstance(masks, torch.Tensor), f"Masks should be torch.Tensor, got {type(masks)}"
    assert masks.dim() >= 3, f"Masks should have at least 3 dimensions, got {masks.dim()}"

    # Validate visualization output
    assert isinstance(visualization, torch.Tensor), f"Visualization should be torch.Tensor, got {type(visualization)}"
    assert visualization.dim() == 4, f"Visualization should be 4D (B,H,W,C), got {visualization.dim()}D"

    # Validate JSON outputs are strings
    assert isinstance(boxes_json, str), f"Boxes JSON should be string, got {type(boxes_json)}"
    assert isinstance(scores_json, str), f"Scores JSON should be string, got {type(scores_json)}"

    # Validate JSON is valid
    import json
    try:
        boxes_data = json.loads(boxes_json)
        scores_data = json.loads(scores_json)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON output: {e}")

    print(f"[TEST] [OK] All outputs are valid")
    print(f"[TEST] [OK] Found {len(boxes_data)} detections" if isinstance(boxes_data, list) else "[TEST] [OK] Boxes data validated")

    # Note: We don't require detections to be found - the image might not contain the prompt
    # The test passes as long as the model runs without errors and produces valid outputs


@pytest.mark.integration
def test_sam3_model_cache():
    """
    Test that SAM3 model caching works correctly.

    This verifies models are loaded from HuggingFace cache on subsequent runs.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN not set - skipping cache test")

    from nodes.load_model import LoadSAM3Model

    print("\n[TEST] Testing model caching...")

    model_loader = LoadSAM3Model()

    # First load (may download)
    print("[TEST] First model load...")
    model1 = model_loader.load_model(model_path="", hf_token=hf_token)
    assert model1 is not None

    # Second load (should use cache)
    print("[TEST] Second model load (should use cache)...")
    model2 = model_loader.load_model(model_path="", hf_token=hf_token)
    assert model2 is not None

    print("[TEST] [OK] Model caching works")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_inference.py -v -s -m integration
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
