# ComfyUI-SAM3

ComfyUI integration for Meta's SAM3 (Segment Anything Model 3) - enabling open-vocabulary image and video segmentation using natural language text prompts.


https://github.com/user-attachments/assets/323df482-1f05-4c69-8681-9bfb4073f766

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
python speedup.py        # Auto-detects your GPU, ~3-5 min compilation
```

This is **optional** and only benefits video tracking performance. Image segmentation works fine without it. The script will:
- Auto-detect your GPU architecture and compile only for your specific GPU (75-80% faster than previous versions)
- Auto-install CUDA toolkit via conda/micromamba if needed
- Compile GPU-accelerated extensions (torch_generic_nms, cc_torch)

**Requirements:** NVIDIA GPU with compute capability 7.5+ (RTX 2000 series or newer), conda/micromamba environment recommended.

**RTX 50-series (Blackwell):** Experimental support available via `python speedup_blackwell.py` (~45-60 sec compilation). May compile successfully but runtime stability not guaranteed due to PyTorch lacking official sm_120 support. Falls back to CPU mode if compilation fails. Track PyTorch support at [pytorch/pytorch#159207](https://github.com/pytorch/pytorch/issues/159207).

## Troubleshooting

### SAM3 nodes not appearing in ComfyUI

If SAM3 doesn't load and you see "running in pytest mode - skipping initialization" in the logs, this is a false positive detection.

**Solution:** Set the environment variable before starting ComfyUI:
```bash
# Linux/Mac
export SAM3_FORCE_INIT=1

# Windows
set SAM3_FORCE_INIT=1
```

This forces SAM3 to initialize even if pytest is detected in your environment.

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

## Quick Start

1. Add **LoadSAM3Model** node (first run downloads ~3.2GB model from HuggingFace)
2. Add **SAM3Segmentation** node, connect model
3. Enter text prompt: `"person"`, `"cat in red"`, `"car on the left"`
4. Get masks, visualization, boxes, and confidence scores

**Text Prompt Examples:**
- `"shoe"`, `"cat"`, `"person"` - Single objects
- `"person in red"`, `"black car"` - With attributes
- `"person on the left"`, `"car in background"` - Spatial relations

**Video Tracking:**
1. Use **SAM3VideoModelLoader** instead of LoadSAM3Model
2. Initialize session with **SAM3InitVideoSession**
3. Add prompts with **SAM3AddVideoPrompt**
4. Propagate with **SAM3PropagateVideo**

## Credits

- **SAM3**: Meta AI Research (https://github.com/facebookresearch/sam3)
- **ComfyUI Integration**: ComfyUI-SAM3
- **Interactive Points Editor**: Adapted from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai (Apache 2.0 License). The SAM3PointsEditor node is based on the PointsEditor implementation from KJNodes, simplified for SAM3-specific point-based segmentation.
