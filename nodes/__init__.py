"""
ComfyUI-SAM3 Nodes - Stateless Architecture

Version 3.0.0 refactoring:
- ComfyUI model management integration (ModelPatcher, load_models_gpu)
- Immutable video state (no global mutable state)
- Automatic cleanup (no manual SAM3CloseVideoSession needed)
"""

# Patch torch.nn.init to skip wasteful random weight initialization.
# All models load checkpoint weights via load_state_dict() immediately after
# construction, so the default kaiming/xavier/etc. init is pure overhead.
# Safe: this file only runs inside the comfy-env isolated subprocess.
import torch.nn.init as _init

def _noop(tensor, *args, **kwargs):
    return tensor

for _fn in (
    "kaiming_uniform_", "kaiming_normal_",
    "xavier_uniform_", "xavier_normal_",
    "uniform_", "normal_", "trunc_normal_",
    "ones_", "zeros_", "constant_",
    "orthogonal_",
):
    if hasattr(_init, _fn):
        setattr(_init, _fn, _noop)

from .load_model import NODE_CLASS_MAPPINGS as LOAD_MODEL_MAPPINGS
from .load_model import NODE_DISPLAY_NAME_MAPPINGS as LOAD_MODEL_DISPLAY_MAPPINGS
from .segmentation import NODE_CLASS_MAPPINGS as SEGMENTATION_MAPPINGS
from .segmentation import NODE_DISPLAY_NAME_MAPPINGS as SEGMENTATION_DISPLAY_MAPPINGS
from .sam3_video_nodes import NODE_CLASS_MAPPINGS as VIDEO_MAPPINGS
from .sam3_video_nodes import NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DISPLAY_MAPPINGS
from .sam3_interactive import NODE_CLASS_MAPPINGS as INTERACTIVE_MAPPINGS
from .sam3_interactive import NODE_DISPLAY_NAME_MAPPINGS as INTERACTIVE_DISPLAY_MAPPINGS

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(LOAD_MODEL_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SEGMENTATION_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VIDEO_MAPPINGS)
NODE_CLASS_MAPPINGS.update(INTERACTIVE_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(LOAD_MODEL_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SEGMENTATION_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(INTERACTIVE_DISPLAY_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
