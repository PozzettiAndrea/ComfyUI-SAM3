# ComfyUI-SAM3

ComfyUI integration for Meta's SAM3 (Segment Anything Model 3) - enabling open-vocabulary image segmentation using natural language text prompts.

## Features

- **Text-Based Segmentation**: Segment objects using natural language descriptions (e.g., "person in red", "cat", "car")
- **Automatic Model Download**: Downloads SAM3 model from HuggingFace automatically if not found locally
- **Geometric Refinement**: Optional box-based prompts for precise control
- **Visual Outputs**: Generates both masks and visualization overlays
- **Flexible Configuration**: Adjustable confidence thresholds and max detections

## Installation

### 1. Install ComfyUI-SAM3

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/ComfyUI-SAM3.git
```

### 2. Install Dependencies

```bash
cd ComfyUI-SAM3
pip install -r requirements.txt
```

This will install:
- SAM3 package from Facebook Research
- PyTorch (if not already installed)
- Other required dependencies

### 3. HuggingFace Authentication

SAM3 models are hosted on HuggingFace and require authentication:

1. Request access to the model at: https://huggingface.co/facebook/sam3
2. Login via HuggingFace CLI:
   ```bash
   huggingface-cli login
   ```
3. Enter your HuggingFace token when prompted

### 4. Restart ComfyUI

Restart ComfyUI to load the new nodes. The prestartup script will automatically:
- Copy example images to `ComfyUI/input/sam3_examples/`
- Copy example workflows to `ComfyUI/user/default/workflows/`

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

## Usage Examples

### Basic Text Segmentation

1. Load an image with `LoadImage`
2. Add `LoadSAM3Model` node (leave model_path empty for auto-download)
3. Add `SAM3Segmentation` node
4. Connect: Image → SAM3Segmentation, SAM3Model → SAM3Segmentation
5. Set text prompt (e.g., "person")
6. Connect `visualization` output to `PreviewImage`
7. Run the workflow!

### Multiple Object Detection

Use text prompts like:
- `"person"` - Detects all people
- `"shoe"` - Detects all shoes
- Adjust `confidence_threshold` to filter results
- Use `max_detections` to limit number of outputs

### Advanced: Geometric Refinement

1. Follow basic setup
2. Add `SAM3GeometricRefine` node
3. Set box parameters to specify region of interest
4. Set `is_positive` to True to segment objects in the box
5. Optionally add `text_prompt` for combined prompting

## Tips and Tricks

### Performance
- First inference is slow (~10-30 seconds) due to torch compilation
- Subsequent inferences are fast (~100-300ms on GPU)
- Model uses ~6-8GB GPU memory
- Keep the model loaded between runs for best performance

### Prompt Engineering
- Be specific: "person in red shirt" works better than just "red"
- Use spatial terms: "car on the left", "person in the background"
- Try variations if results aren't perfect
- Adjust confidence threshold if getting too many/few detections

### Troubleshooting

**"SAM3 package not found"**
- Run `pip install -r requirements.txt` in the ComfyUI-SAM3 directory

**"Failed to download model from HuggingFace"**
- Request access at https://huggingface.co/facebook/sam3
- Run `huggingface-cli login` and enter your token
- Check your internet connection

**"Out of memory"**
- Try device="cpu" if GPU memory is limited
- Reduce image size before processing
- Close other GPU-intensive applications

**No detections found**
- Lower confidence_threshold
- Try different text prompts
- Check if the object is actually in the image
- Ensure the image is properly loaded

## Model Information

- **Model**: SAM3 (Segment Anything Model 3) by Meta
- **Size**: ~848M parameters (~3.2GB checkpoint)
- **Input Resolution**: 1008x1008 pixels
- **Capabilities**: Open-vocabulary segmentation with 270K+ unique concepts
- **License**: Check Meta's license at https://huggingface.co/facebook/sam3

## File Structure

```
ComfyUI-SAM3/
├── __init__.py                # Main entry point
├── prestartup_script.py       # Auto-copy assets/workflows
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── nodes/
│   ├── __init__.py           # Node exports
│   ├── load_model.py         # LoadSAM3Model node
│   ├── segmentation.py       # Segmentation nodes
│   └── utils.py              # Helper functions
├── assets/
│   └── example_image.jpg     # Example image
└── workflows/
    └── sam3_example.json     # Example workflow
```

## Credits

- **SAM3**: Meta AI Research (https://github.com/facebookresearch/sam3)
- **ComfyUI Integration**: ComfyUI-SAM3
- **Interactive Points Editor**: Adapted from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai (Apache 2.0 License). The SAM3PointsEditor node is based on the PointsEditor implementation from KJNodes, simplified for SAM3-specific point-based segmentation.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Changelog

### v1.0.0 (2024-01-XX)
- Initial release
- LoadSAM3Model node with auto-download
- SAM3Segmentation node with text prompts
- SAM3GeometricRefine node for box prompts
- Example workflows and assets
