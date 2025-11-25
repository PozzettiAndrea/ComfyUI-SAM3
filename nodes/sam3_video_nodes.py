"""
SAM3 Video Tracking Nodes for ComfyUI - Stateless Architecture

These nodes provide video object tracking and segmentation using SAM3.
All state is encoded in immutable outputs - no global mutable state.

Key design principles:
1. All nodes are stateless - state flows through outputs
2. SAM3VideoState is immutable - adding prompts returns NEW state
3. Inference state is reconstructed on-demand
4. Temp directories are automatically cleaned up at process exit
5. No manual SAM3CloseVideoSession needed
"""
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import folder_paths
import comfy.model_management

from .video_state import (
    SAM3VideoState,
    VideoPrompt,
    VideoConfig,
    create_video_state,
    cleanup_temp_dir,
)
from .inference_reconstructor import (
    get_inference_state,
    invalidate_session,
    clear_inference_cache,
)
from .sam3_model_patcher import SAM3ModelWrapper, SAM3ModelPatcher


# =============================================================================
# VRAM Debug Utility
# =============================================================================

def print_vram(label: str, detailed: bool = False):
    """Print current VRAM usage for debugging memory leaks."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label}: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
        if detailed:
            # Print memory stats breakdown
            stats = torch.cuda.memory_stats()
            print(f"[VRAM]   Active: {stats.get('active_bytes.all.current', 0) / 1024**3:.2f}GB")
            print(f"[VRAM]   Inactive: {stats.get('inactive_split_bytes.all.current', 0) / 1024**3:.2f}GB")
            print(f"[VRAM]   Allocated retries: {stats.get('num_alloc_retries', 0)}")


def debug_cuda_tensors():
    """Find all CUDA tensors and their sizes - HACKY but useful for debugging."""
    if not torch.cuda.is_available():
        return

    print("[CUDA DEBUG] Scanning for GPU tensors...")
    tensor_info = {}
    total_size = 0

    # Scan all objects in memory
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.numel() * obj.element_size() / 1024**2
                shape_str = str(tuple(obj.shape))
                dtype_str = str(obj.dtype)
                key = f"{shape_str}_{dtype_str}"
                if key not in tensor_info:
                    tensor_info[key] = {"count": 0, "size_mb": size_mb, "shape": shape_str, "dtype": dtype_str}
                tensor_info[key]["count"] += 1
                total_size += size_mb
        except:
            pass

    # Sort by size and print top 10
    sorted_tensors = sorted(tensor_info.values(), key=lambda x: x["size_mb"] * x["count"], reverse=True)
    print(f"[CUDA DEBUG] Total GPU tensors: {sum(t['count'] for t in tensor_info.values())}, Total size: {total_size/1024:.2f}GB")
    print("[CUDA DEBUG] Top 10 tensor types by total size:")
    for i, t in enumerate(sorted_tensors[:10]):
        total_mb = t["size_mb"] * t["count"]
        print(f"[CUDA DEBUG]   {i+1}. {t['shape']} {t['dtype']}: {t['count']}x {t['size_mb']:.1f}MB = {total_mb:.1f}MB")


def get_sam3_video_models():
    """Get list of available SAM3 models for video."""
    try:
        models = folder_paths.get_filename_list("sam3")
        return models if models else []
    except Exception:
        return []


# =============================================================================
# Video Model Loader
# =============================================================================

class SAM3VideoModelLoader:
    """
    Load SAM3 model for video tracking.

    Uses ComfyUI's model management for GPU/CPU handling.
    """

    # Class variables to track current model state
    _current_predictor = None
    _current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        models = get_sam3_video_models()
        model_choices = models + ["[Download from HuggingFace]"] if models else ["[Download from HuggingFace]"]

        return {
            "required": {
                "model_name": (model_choices, {
                    "default": model_choices[0] if model_choices else "[Download from HuggingFace]",
                    "tooltip": "Select SAM3 model from ComfyUI/models/sam3/ folder"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace token for downloading gated models"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, model_name, hf_token=""):
        # Only reload if model name changes - keeps CUDA caches stable
        # dtype issues are handled by resetting autocast context in propagate
        return model_name

    RETURN_TYPES = ("SAM3_VIDEO_MODEL",)
    RETURN_NAMES = ("video_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3/video"

    def load_model(self, model_name, hf_token=""):
        """Load the SAM3 video model."""
        import os
        from .sam3_lib.model_builder import build_sam3_video_predictor

        # Check if we already have this model loaded - REUSE IT
        if (SAM3VideoModelLoader._current_predictor is not None and
            SAM3VideoModelLoader._current_model_name == model_name and
            hasattr(SAM3VideoModelLoader._current_predictor, 'model')):
            print(f"[SAM3 Video] Reusing already-loaded model: {model_name}")
            print_vram("Reusing model")
            # Just clear sessions, don't reload
            from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
            if Sam3VideoPredictor._ALL_INFERENCE_STATES:
                print(f"[SAM3 Video] Clearing {len(Sam3VideoPredictor._ALL_INFERENCE_STATES)} sessions")
                Sam3VideoPredictor._ALL_INFERENCE_STATES.clear()
            return (SAM3VideoModelLoader._current_predictor,)

        # Set HF token if provided
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # BPE path for tokenizer
        bpe_path = Path(__file__).parent / "sam3_lib" / "bpe_simple_vocab_16e6.txt.gz"
        bpe_path = str(bpe_path)

        # Determine checkpoint path
        if model_name == "[Download from HuggingFace]":
            checkpoint_path = self._download_from_hf(hf_token)
        else:
            checkpoint_path = folder_paths.get_full_path("sam3", model_name)
            if checkpoint_path is None:
                raise FileNotFoundError(f"Model not found: {model_name}")

        print(f"[SAM3 Video] Loading NEW model from {checkpoint_path}")
        print(f"[SAM3 Video] Using BPE tokenizer: {bpe_path}")
        print_vram("Before model load")

        # CRITICAL: Delete old model BEFORE building new one
        # ComfyUI holds reference to old output tuple, so we can't just del the wrapper.
        # We must explicitly delete the INTERNAL PyTorch model to free GPU memory.
        if SAM3VideoModelLoader._current_predictor is not None:
            print("[SAM3 Video] Deleting previous model to free VRAM")
            # Move model to CPU first to force GPU memory release
            # Then delete - this works even if ComfyUI holds a reference
            if hasattr(SAM3VideoModelLoader._current_predictor, 'model'):
                model = SAM3VideoModelLoader._current_predictor.model
                print(f"[SAM3 Video]   Moving model to CPU: {type(model)}")
                try:
                    # Clear any internal caches/dicts that might hold GPU tensors
                    for name in dir(model):
                        if name.startswith('_'):
                            continue
                        try:
                            attr = getattr(model, name, None)
                            if isinstance(attr, dict):
                                attr.clear()
                            elif isinstance(attr, list):
                                attr.clear()
                        except:
                            pass
                    print_vram("After clearing model caches")
                    model.cpu()  # Move all parameters/buffers to CPU
                    print_vram("After model.cpu()")
                except Exception as e:
                    print(f"[SAM3 Video]   Warning: model.cpu() failed: {e}")
                del SAM3VideoModelLoader._current_predictor.model
                del model
                print_vram("After del predictor.model")
            del SAM3VideoModelLoader._current_predictor
            SAM3VideoModelLoader._current_predictor = None
            print_vram("After del _current_predictor")

        # Clear ALL sessions from ALL predictor instances (class variable)
        from .sam3_lib.sam3_video_predictor import Sam3VideoPredictor
        if Sam3VideoPredictor._ALL_INFERENCE_STATES:
            print(f"[SAM3 Video] Clearing {len(Sam3VideoPredictor._ALL_INFERENCE_STATES)} orphaned sessions")
            for sid in list(Sam3VideoPredictor._ALL_INFERENCE_STATES.keys()):
                session = Sam3VideoPredictor._ALL_INFERENCE_STATES.pop(sid, None)
                if session:
                    del session
            print_vram("After clearing sessions")

        # Reset ALL torch caches aggressively
        try:
            import torch._dynamo
            torch._dynamo.reset()
            print("[SAM3 Video] Reset torch._dynamo")
        except Exception as e:
            print(f"[SAM3 Video] Warning: dynamo reset failed: {e}")

        try:
            # Reset inductor cache (compiled kernels)
            import torch._inductor
            if hasattr(torch._inductor, 'codecache'):
                torch._inductor.codecache.cache_clear()
                print("[SAM3 Video] Cleared inductor codecache")
        except Exception as e:
            print(f"[SAM3 Video] Warning: inductor clear failed: {e}")

        # Disable cuDNN benchmark to prevent workspace caching
        torch.backends.cudnn.benchmark = False

        # Force garbage collection and CUDA cleanup
        gc.collect()
        gc.collect()  # Run twice for weak refs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Try to reset CUDA caching allocator
            try:
                torch.cuda.memory.reset_peak_memory_stats()
                torch.cuda.memory.reset_accumulated_memory_stats()
            except:
                pass
            # Force cuDNN to release workspace
            torch.backends.cudnn.benchmark = False
        print_vram("After cleanup before load", detailed=True)

        # HACKY DEBUG: Scan for lingering GPU tensors
        debug_cuda_tensors()

        # Build the video predictor
        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            hf_token=hf_token if hf_token else None,
            gpus_to_use=None,  # Single GPU mode
        )

        # Store reference for reuse
        SAM3VideoModelLoader._current_predictor = predictor
        SAM3VideoModelLoader._current_model_name = model_name

        print_vram("After model load")
        print(f"[SAM3 Video] Model loaded successfully")

        return (predictor,)

    def _download_from_hf(self, hf_token):
        """Download model from HuggingFace if needed."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")

        if not hf_token:
            raise ValueError("HuggingFace token required to download SAM3 model")

        sam3_paths = folder_paths.get_folder_paths("sam3")
        if not sam3_paths:
            raise RuntimeError("sam3 folder not registered")

        models_dir = sam3_paths[0]
        local_path = Path(models_dir) / "sam3.pt"

        if local_path.exists():
            return str(local_path)

        print("[SAM3 Video] Downloading from HuggingFace...")
        hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            token=hf_token,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        return str(local_path)


# =============================================================================
# Video Session Initialization
# =============================================================================

class SAM3InitVideo:
    """
    Initialize a video tracking session.

    Returns immutable SAM3VideoState that flows through subsequent nodes.
    No global state - all data encoded in output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_model": ("SAM3_VIDEO_MODEL", {
                    "tooltip": "SAM3 video model"
                }),
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images"
                }),
            },
            "optional": {
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
                "new_det_thresh": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "New object tracking threshold"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, video_model, video_frames, score_threshold=0.3, new_det_thresh=0.4):
        # Content-based hash
        return hash((
            video_frames.shape,
            float(video_frames.mean()),
            score_threshold,
            new_det_thresh,
        ))

    RETURN_TYPES = ("SAM3_VIDEO_STATE", "STRING")
    RETURN_NAMES = ("video_state", "session_id")
    FUNCTION = "init"
    CATEGORY = "SAM3/video"

    def init(self, video_model, video_frames, score_threshold=0.3, new_det_thresh=0.4):
        """Initialize video session with immutable state."""
        print_vram("Before init video")

        config = VideoConfig(
            score_threshold_detection=score_threshold,
            new_det_thresh=new_det_thresh,
        )

        # Create immutable video state
        video_state = create_video_state(
            video_frames=video_frames,
            config=config,
        )

        print(f"[SAM3 Video] Initialized session {video_state.session_uuid[:8]}")
        print(f"[SAM3 Video] Frames: {video_state.num_frames}, Size: {video_state.width}x{video_state.height}")
        print_vram("After init video")

        return (video_state, video_state.session_uuid)


# =============================================================================
# Add Prompts (Returns NEW State)
# =============================================================================

class SAM3AddVideoPointPrompt:
    """
    Add point prompt to video - accepts SAM3_POINTS_PROMPT from SAM3PointCollector.

    Returns NEW SAM3VideoState with prompt added (immutable update).
    Connect positive_points and negative_points outputs from SAM3PointCollector.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3InitVideo or previous prompt node"
                }),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to add prompt (usually 0 for first frame)"
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Object ID for tracking this object"
                }),
            },
            "optional": {
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Positive points from SAM3PointCollector"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "Negative points from SAM3PointCollector"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "add_prompt"
    CATEGORY = "SAM3/video"

    def add_prompt(self, video_state, frame_idx, obj_id,
                   positive_points=None, negative_points=None):
        """Add point prompt and return NEW state."""
        # Invalidate cached inference state since prompts are changing
        invalidate_session(video_state.session_uuid)

        # Combine positive and negative points
        all_points = []
        all_labels = []

        if positive_points and positive_points.get("points"):
            for pt in positive_points["points"]:
                # Keep normalized coords (0-1) - model expects rel_coordinates=True
                all_points.append([float(pt[0]), float(pt[1])])
                all_labels.append(1)  # Positive

        if negative_points and negative_points.get("points"):
            for pt in negative_points["points"]:
                # Keep normalized coords (0-1) - model expects rel_coordinates=True
                all_points.append([float(pt[0]), float(pt[1])])
                all_labels.append(0)  # Negative

        if not all_points:
            print("[SAM3 Video] No points provided, returning unchanged state")
            return (video_state,)

        # Create prompt and return NEW state
        prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
        new_state = video_state.with_prompt(prompt)

        pos_count = len(positive_points.get("points", [])) if positive_points else 0
        neg_count = len(negative_points.get("points", [])) if negative_points else 0
        print(f"[SAM3 Video] Added point prompt: frame={frame_idx}, obj={obj_id}, "
              f"positive={pos_count}, negative={neg_count}")
        print(f"[SAM3 Video] Points (normalized): {all_points}")

        return (new_state,)


class SAM3AddVideoBoxPrompt:
    """
    Add box prompt to video - accepts SAM3_BOXES_PROMPT from SAM3BBoxCollector.

    Returns NEW SAM3VideoState with prompt added (immutable update).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3InitVideo or previous prompt node"
                }),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to add prompt"
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Object ID for tracking this object"
                }),
            },
            "optional": {
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "Positive boxes from SAM3BBoxCollector"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "add_prompt"
    CATEGORY = "SAM3/video"

    def add_prompt(self, video_state, frame_idx, obj_id, positive_boxes=None):
        """Add box prompt and return NEW state."""
        print(f"[SAM3 Video] SAM3AddVideoBoxPrompt.add_prompt called!")
        print(f"[SAM3 Video]   video_state: {video_state}")
        print(f"[SAM3 Video]   frame_idx: {frame_idx}, obj_id: {obj_id}")
        print(f"[SAM3 Video]   positive_boxes: {positive_boxes}")
        print(f"[SAM3 Video]   positive_boxes type: {type(positive_boxes)}")

        invalidate_session(video_state.session_uuid)

        if not positive_boxes or not positive_boxes.get("boxes"):
            print("[SAM3 Video] No boxes provided, returning unchanged state")
            return (video_state,)

        # Take first box (SAM3 video uses one box per object)
        box_data = positive_boxes["boxes"][0]  # [center_x, center_y, w, h] normalized

        # Convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
        # Keep normalized (0-1) coordinates - model expects normalized
        cx, cy, w, h = box_data
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2])
        new_state = video_state.with_prompt(prompt)

        print(f"[SAM3 Video] Added box prompt: frame={frame_idx}, obj={obj_id}, "
              f"box (normalized)=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

        return (new_state,)


class SAM3AddVideoTextPrompt:
    """
    Add text prompt to video for grounding-based tracking.

    Returns NEW SAM3VideoState with prompt added (immutable update).
    Uses SAM3's grounding capability to detect and track objects by text description.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state from SAM3InitVideo or previous prompt node"
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text description of object to track (e.g., 'person', 'red car', 'dog')"
                }),
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to add prompt"
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Object ID for tracking this object"
                }),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "add_prompt"
    CATEGORY = "SAM3/video"

    def add_prompt(self, video_state, text_prompt, frame_idx, obj_id):
        """Add text prompt and return NEW state."""
        if not text_prompt.strip():
            print("[SAM3 Video] Empty text prompt, returning unchanged state")
            return (video_state,)

        invalidate_session(video_state.session_uuid)

        prompt = VideoPrompt.create_text(frame_idx, obj_id, text_prompt.strip())
        new_state = video_state.with_prompt(prompt)

        print(f"[SAM3 Video] Added text prompt: frame={frame_idx}, obj={obj_id}, "
              f"text='{text_prompt.strip()}'")

        return (new_state,)


# =============================================================================
# Propagation
# =============================================================================

class SAM3Propagate:
    """
    Run video propagation to track objects across frames.

    Reconstructs inference state on-demand from immutable video state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_model": ("SAM3_VIDEO_MODEL", {
                    "tooltip": "SAM3 video model"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state with prompts"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Start frame for propagation"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "End frame (-1 for all)"
                }),
                "reverse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Propagate backwards"
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_STATE")
    RETURN_NAMES = ("masks", "video_state")
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    def propagate(self, video_model, video_state, start_frame=0, end_frame=-1, reverse=False):
        """Run propagation using reconstructed inference state."""
        if len(video_state.prompts) == 0:
            raise ValueError("[SAM3 Video] No prompts added. Add point, box, or text prompts before propagating.")

        print(f"[SAM3 Video] Starting propagation: frames {start_frame} to {end_frame if end_frame >= 0 else 'end'}")
        print(f"[SAM3 Video] Prompts: {len(video_state.prompts)}")
        print_vram("Before propagation start")

        # Determine frame range
        if end_frame < 0:
            end_frame = video_state.num_frames - 1

        # Build propagation request - uses predictor's handle_stream_request API
        propagation_direction = "backward" if reverse else "forward"
        request = {
            "type": "propagate_in_video",
            "session_id": video_state.session_uuid,
            "propagation_direction": propagation_direction,
            "start_frame_index": start_frame,
            "max_frame_num_to_track": end_frame - start_frame + 1,
        }

        # Run ALL inference inside autocast context for dtype consistency
        # SAM3 requires bf16 - wrap reconstruction AND propagation
        masks_dict = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            print_vram("Before reconstruction (in autocast)")
            # Reconstruct inference state from immutable state
            inference_state = get_inference_state(video_model, video_state)
            print_vram("After reconstruction")

            # Run propagation
            try:
                for response in video_model.handle_stream_request(request):
                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None:
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        continue

                    # Try different possible mask keys
                    mask_key = None
                    for key in ["out_binary_masks", "video_res_masks", "masks"]:
                        if key in outputs and outputs[key] is not None:
                            mask_key = key
                            break

                    if mask_key:
                        # Move masks to CPU immediately to free GPU memory
                        mask = outputs[mask_key]
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu()
                        masks_dict[frame_idx] = mask
                        if frame_idx == 0:
                            print(f"[SAM3 Video] Frame 0 mask shape: {mask.shape if hasattr(mask, 'shape') else type(mask)}")
                            if hasattr(mask, 'sum'):
                                print(f"[SAM3 Video] Frame 0 mask sum: {mask.sum()}")

                    # Periodic cleanup and VRAM monitoring
                    if frame_idx % 10 == 0:
                        print_vram(f"Frame {frame_idx}")
                        gc.collect()

            except Exception as e:
                print(f"[SAM3 Video] Propagation error: {e}")
                import traceback
                traceback.print_exc()
                raise

        print_vram("After propagation loop")
        print(f"[SAM3 Video] Propagation complete: {len(masks_dict)} frames processed")

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (masks_dict, video_state)


# =============================================================================
# Output Extraction
# =============================================================================

class SAM3VideoOutput:
    """
    Extract masks from propagation results.

    Converts SAM3_VIDEO_MASKS to ComfyUI-compatible mask tensors.
    Returns all frames as a batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks from SAM3Propagate"
                }),
                "video_state": ("SAM3_VIDEO_STATE", {
                    "tooltip": "Video state for dimensions"
                }),
            },
            "optional": {
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object ID (-1 for all combined)"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "frames")
    FUNCTION = "extract"
    CATEGORY = "SAM3/video"

    def extract(self, masks, video_state, obj_id=-1):
        """Extract all masks as a batch [N, H, W]."""
        from PIL import Image
        import os

        print_vram("Before extract")
        h, w = video_state.height, video_state.width
        num_frames = video_state.num_frames

        if not masks:
            print("[SAM3 Video] No masks to extract")
            empty_mask = torch.zeros(num_frames, h, w)
            empty_frames = torch.zeros(num_frames, h, w, 3)
            return (empty_mask, empty_frames)

        # Process all frames in order
        mask_list = []
        frame_list = []

        for frame_idx in range(num_frames):
            # Get mask for this frame
            if frame_idx in masks:
                frame_mask = masks[frame_idx]

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)

                # Convert mask to ComfyUI format
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)  # Remove batch dim

                if frame_mask.dim() == 3 and frame_mask.shape[0] > 1:
                    # Multiple objects - combine or select
                    if obj_id >= 0 and obj_id < frame_mask.shape[0]:
                        frame_mask = frame_mask[obj_id]
                    else:
                        # Combine all objects
                        frame_mask = frame_mask.max(dim=0)[0]
                elif frame_mask.dim() == 3:
                    frame_mask = frame_mask.squeeze(0)  # [1, H, W] -> [H, W]

                # Ensure [H, W] format
                frame_mask = frame_mask.float()

                # Handle empty masks
                if frame_mask.numel() == 0:
                    frame_mask = torch.zeros(h, w)
                elif frame_mask.max() > 1.0:
                    frame_mask = frame_mask / 255.0
            else:
                # No mask for this frame - use zeros
                frame_mask = torch.zeros(h, w)

            mask_list.append(frame_mask.cpu())

            # Load original frame
            frame_path = os.path.join(video_state.temp_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)  # [H, W, C]
            else:
                img_tensor = torch.zeros(h, w, 3)

            frame_list.append(img_tensor)

        # Stack into batches
        all_masks = torch.stack(mask_list, dim=0)  # [N, H, W]
        all_frames = torch.stack(frame_list, dim=0)  # [N, H, W, C]

        print(f"[SAM3 Video] Output: {all_masks.shape[0]} masks, shape {all_masks.shape}")
        print_vram("After extract")

        return (all_masks, all_frames)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3VideoModelLoader": SAM3VideoModelLoader,
    "SAM3InitVideo": SAM3InitVideo,
    "SAM3AddVideoPointPrompt": SAM3AddVideoPointPrompt,
    "SAM3AddVideoBoxPrompt": SAM3AddVideoBoxPrompt,
    "SAM3AddVideoTextPrompt": SAM3AddVideoTextPrompt,
    "SAM3Propagate": SAM3Propagate,
    "SAM3VideoOutput": SAM3VideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoModelLoader": "SAM3 Video Model Loader",
    "SAM3InitVideo": "SAM3 Init Video",
    "SAM3AddVideoPointPrompt": "SAM3 Add Video Point Prompt",
    "SAM3AddVideoBoxPrompt": "SAM3 Add Video Box Prompt",
    "SAM3AddVideoTextPrompt": "SAM3 Add Video Text Prompt",
    "SAM3Propagate": "SAM3 Propagate",
    "SAM3VideoOutput": "SAM3 Video Output",
}
