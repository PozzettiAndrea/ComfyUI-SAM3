# ComfyUI-SAM3

ComfyUI integration for Meta's SAM3 (Segment Anything Model 3) - enabling open-vocabulary image and video segmentation using natural language text prompts.

## Installation

Install via ComfyUI Manager or clone to `ComfyUI/custom_nodes/`:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/PozzettiAndrea/ComfyUI-SAM3.git
cd ComfyUI-SAM3
python install.py
```

### Optional: GPU Acceleration for Video Tracking

For 5-10x faster video tracking, install GPU-accelerated CUDA extensions:
```bash
python speedup.py
```

This is **optional** and only benefits video tracking performance. Image segmentation works fine without it. The script will:
- Auto-install micromamba if needed
- Install CUDA toolkit via conda/micromamba
- Compile GPU-accelerated extensions (torch_generic_nms, cc_torch)

**Requirements:** NVIDIA GPU, conda/micromamba environment recommended.

### Examples

![bbox](docs/bbox.png)

![point](docs/point.png)

![text_prompt](docs/text_prompt.png)

![video](docs/video.png)

## Nodes

### Image Segmentation
- **LoadSAM3Model** - Load SAM3 model for image segmentation
- **SAM3Segmentation** - Segment objects using text prompts ("person", "cat in red", etc.)
- **SAM3CreateBox** - Create bounding box prompts (normalized coordinates)
- **SAM3CreatePoint** - Create point prompts with positive/negative labels
- **SAM3CombineBoxes** - Combine multiple box prompts
- **SAM3CombinePoints** - Combine multiple point prompts

### Video Tracking
- **SAM3VideoModelLoader** - Load SAM3 model for video tracking
- **SAM3InitVideoSession** - Initialize video tracking session
- **SAM3InitVideoSessionAdvanced** - Advanced session initialization with custom settings
- **SAM3AddVideoPrompt** - Add object prompts to track in video
- **SAM3PropagateVideo** - Propagate object tracking through video frames

### Interactive Tools
- **SAM3PointCollector** - Interactive UI for collecting point prompts
- **SAM3BBoxCollector** - Interactive UI for drawing bounding boxes

---

## Node Details

### LoadSAM3Model

Loads the SAM3 model and creates a processor for inference.

**Inputs:**
- `device` (auto/cuda/cpu): Device to run the model on (default: auto)
- `model_path` (optional): Path to custom checkpoint (leave empty to auto-download)

**Outputs:**
- `sam3_model`: Model object to be used by segmentation nodes

**Notes:**
- First run will download the model (~3.2GB) from HuggingFace
- Model is cached in `ComfyUI/models/sam3/sam3.pt`
- Subsequent loads are instant (uses in-memory cache)

### SAM3Segmentation

Performs segmentation using text prompts.

**Inputs:**
- `sam3_model`: Model from LoadSAM3Model node
- `image`: Input image (ComfyUI IMAGE format)
- `text_prompt`: Natural language description of objects to segment
  - Examples: "person", "cat", "person in red", "car on the left"
- `confidence_threshold` (0.0-1.0): Minimum confidence score (default: 0.5)
- `max_detections` (optional): Maximum number of detections to return (-1 for all)

**Outputs:**
- `masks`: Binary segmentation masks (ComfyUI MASK format)
- `visualization`: Image with colored mask overlays and bounding boxes
- `boxes`: JSON string containing bounding box coordinates [[x0, y0, x1, y1], ...]
- `scores`: JSON string containing confidence scores [0.95, 0.87, ...]

**Example Prompts:**
- Single object: `"shoe"`, `"cat"`, `"person"`
- With attributes: `"person in red"`, `"black car"`, `"wooden table"`
- Spatial relations: `"person on the left"`, `"car in the background"`

### SAM3GeometricRefine

Refines segmentation using geometric box prompts (advanced usage).

**Inputs:**
- `sam3_model`: Model from LoadSAM3Model node
- `image`: Input image
- `box_x`, `box_y`: Box center coordinates (normalized 0-1)
- `box_w`, `box_h`: Box width and height (normalized 0-1)
- `is_positive`: True for positive prompt, False for negative
- `confidence_threshold`: Minimum confidence score
- `text_prompt` (optional): Combine with text prompt

**Outputs:**
- Same as SAM3Segmentation

**Notes:**
- Box coordinates are normalized to [0, 1] range
- Positive prompts include the box region, negative prompts exclude it
- Can be combined with text prompts for more precise control

## Credits

- **SAM3**: Meta AI Research (https://github.com/facebookresearch/sam3)
- **ComfyUI Integration**: ComfyUI-SAM3
- **Interactive Points Editor**: Adapted from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai (Apache 2.0 License). The SAM3PointsEditor node is based on the PointsEditor implementation from KJNodes, simplified for SAM3-specific point-based segmentation.
