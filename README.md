# ComfyUI-SAM3

ComfyUI integration for Meta's SAM3 (Segment Anything Model 3) - enabling open-vocabulary image segmentation using natural language text prompts.

###Examples

![bbox](docs/bbox.png)

## Nodes

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
