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
# Video Segmentation (Unified Node)
# =============================================================================

class SAM3VideoSegmentation:
    """
    Initialize video tracking and add prompts.

    Select prompt_mode to choose between:
    - text: Track objects by text description (comma-separated for multiple)
    - point: Track objects by clicking points (positive/negative)
    - box: Track objects by drawing boxes (positive/negative)

    Note: SAM3 video does NOT support combining different prompt types.
    Each mode is mutually exclusive.
    """

    PROMPT_MODES = ["text", "point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames as batch of images [N, H, W, C]"
                }),
                "prompt_mode": (cls.PROMPT_MODES, {
                    "default": "text",
                    "tooltip": "Prompt type: text (describe objects), point (click on objects), or box (draw rectangles)"
                }),
            },
            "optional": {
                # Text mode inputs
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "[text mode] Text description(s) to track. Comma-separated for multiple objects (e.g., 'person, dog, car')"
                }),
                # Point mode inputs
                "positive_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Positive points - click on objects to track"
                }),
                "negative_points": ("SAM3_POINTS_PROMPT", {
                    "tooltip": "[point mode] Negative points - click on areas to exclude"
                }),
                # Box mode inputs
                "positive_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Positive boxes - draw around objects to track"
                }),
                "negative_boxes": ("SAM3_BOXES_PROMPT", {
                    "tooltip": "[box mode] Negative boxes - draw around areas to exclude"
                }),
                # Common inputs
                "frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to apply prompts (usually 0 for first frame)"
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, video_frames, prompt_mode="text", text_prompt="",
                   positive_points=None, negative_points=None,
                   positive_boxes=None, negative_boxes=None,
                   frame_idx=0, score_threshold=0.3):
        # Content-based hash
        return hash((
            video_frames.shape,
            float(video_frames.mean()),
            prompt_mode,
            text_prompt,
            str(positive_points),
            str(negative_points),
            str(positive_boxes),
            str(negative_boxes),
            frame_idx,
            score_threshold,
        ))

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"

    def segment(self, video_frames, prompt_mode="text", text_prompt="",
                positive_points=None, negative_points=None,
                positive_boxes=None, negative_boxes=None,
                frame_idx=0, score_threshold=0.3):
        """Initialize video state and add prompts based on selected mode."""
        print_vram("Before video segmentation")

        # 1. Initialize video state
        config = VideoConfig(
            score_threshold_detection=score_threshold,
        )
        video_state = create_video_state(
            video_frames=video_frames,
            config=config,
        )

        print(f"[SAM3 Video] Initialized session {video_state.session_uuid[:8]}")
        print(f"[SAM3 Video] Frames: {video_state.num_frames}, Size: {video_state.width}x{video_state.height}")
        print(f"[SAM3 Video] Prompt mode: {prompt_mode}")

        # 2. Add prompts based on mode (mutually exclusive)
        obj_id = 1

        if prompt_mode == "text":
            # Text mode: parse comma-separated text prompts
            if text_prompt and text_prompt.strip():
                for text in text_prompt.split(","):
                    text = text.strip()
                    if text:
                        prompt = VideoPrompt.create_text(frame_idx, obj_id, text)
                        video_state = video_state.with_prompt(prompt)
                        print(f"[SAM3 Video] Added text prompt: obj={obj_id}, text='{text}'")
                        obj_id += 1
            else:
                print("[SAM3 Video] Warning: text mode selected but no text_prompt provided")

        elif prompt_mode == "point":
            # Point mode: combine positive and negative points
            all_points = []
            all_labels = []

            if positive_points and positive_points.get("points"):
                for pt in positive_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(1)  # Positive

            if negative_points and negative_points.get("points"):
                for pt in negative_points["points"]:
                    all_points.append([float(pt[0]), float(pt[1])])
                    all_labels.append(0)  # Negative

            if all_points:
                prompt = VideoPrompt.create_point(frame_idx, obj_id, all_points, all_labels)
                video_state = video_state.with_prompt(prompt)
                pos_count = len(positive_points.get("points", [])) if positive_points else 0
                neg_count = len(negative_points.get("points", [])) if negative_points else 0
                print(f"[SAM3 Video] Added point prompt: obj={obj_id}, "
                      f"positive={pos_count}, negative={neg_count}")
            else:
                print("[SAM3 Video] Warning: point mode selected but no points provided")

        elif prompt_mode == "box":
            # Box mode: add positive and/or negative boxes
            has_boxes = False

            if positive_boxes and positive_boxes.get("boxes"):
                box_data = positive_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=True)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 Video] Added positive box: obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if negative_boxes and negative_boxes.get("boxes"):
                box_data = negative_boxes["boxes"][0]  # First box
                cx, cy, w, h = box_data
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                prompt = VideoPrompt.create_box(frame_idx, obj_id, [x1, y1, x2, y2], is_positive=False)
                video_state = video_state.with_prompt(prompt)
                print(f"[SAM3 Video] Added negative box: obj={obj_id}, "
                      f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                has_boxes = True

            if not has_boxes:
                print("[SAM3 Video] Warning: box mode selected but no boxes provided")

        # Validate at least one prompt was added
        if len(video_state.prompts) == 0:
            print(f"[SAM3 Video] Warning: No prompts added for mode '{prompt_mode}'")

        print(f"[SAM3 Video] Total prompts: {len(video_state.prompts)}")
        print_vram("After video segmentation")

        return (video_state,)


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

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SCORES", "SAM3_VIDEO_STATE")
    RETURN_NAMES = ("masks", "scores", "video_state")
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
        scores_dict = {}  # Store confidence scores per frame
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
                        print(f"[SAM3 Video DEBUG] Response has no frame_idx: {response.keys() if hasattr(response, 'keys') else type(response)}")
                        continue

                    outputs = response.get("outputs", response)
                    if outputs is None:
                        print(f"[SAM3 Video DEBUG] Frame {frame_idx}: outputs is None")
                        continue

                    # Debug: print all output keys for first few frames
                    if frame_idx < 3:
                        print(f"[SAM3 Video DEBUG] Frame {frame_idx} output keys: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")
                        for k, v in outputs.items() if hasattr(outputs, 'items') else []:
                            if hasattr(v, 'shape'):
                                print(f"[SAM3 Video DEBUG]   {k}: shape={v.shape}, dtype={v.dtype if hasattr(v, 'dtype') else 'N/A'}")
                            else:
                                print(f"[SAM3 Video DEBUG]   {k}: {type(v)} = {v}")

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
                        if frame_idx < 3:
                            print(f"[SAM3 Video DEBUG] Frame {frame_idx} mask: shape={mask.shape if hasattr(mask, 'shape') else type(mask)}, sum={mask.sum() if hasattr(mask, 'sum') else 'N/A'}")
                            if hasattr(mask, 'shape') and len(mask.shape) >= 1 and mask.size > 0:
                                print(f"[SAM3 Video DEBUG]   mask dtype={mask.dtype}, min={mask.min()}, max={mask.max()}")
                    else:
                        if frame_idx < 3:
                            print(f"[SAM3 Video DEBUG] Frame {frame_idx}: No mask key found in outputs")

                    # Capture confidence scores
                    for score_key in ["out_probs", "scores", "confidences", "obj_scores"]:
                        if score_key in outputs and outputs[score_key] is not None:
                            probs = outputs[score_key]
                            if hasattr(probs, 'cpu'):
                                probs = probs.cpu()
                            elif isinstance(probs, np.ndarray):
                                probs = torch.from_numpy(probs)
                            scores_dict[frame_idx] = probs
                            if frame_idx == 0:
                                print(f"[SAM3 Video] Frame 0 scores ({score_key}): {probs.tolist() if hasattr(probs, 'tolist') else probs}")
                            break

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
        print(f"[SAM3 Video] Frames with scores: {len(scores_dict)}")

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (masks_dict, scores_dict, video_state)


# =============================================================================
# Output Extraction
# =============================================================================

class SAM3VideoOutput:
    """
    Extract masks from propagation results.

    Converts SAM3_VIDEO_MASKS to ComfyUI-compatible mask tensors.
    Returns all frames as a batch.

    Changing obj_id does NOT re-run propagation - only this node re-executes.
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
                "scores": ("SAM3_VIDEO_SCORES", {
                    "tooltip": "Confidence scores from SAM3Propagate"
                }),
                "obj_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Specific object ID for mask output (-1 for all combined). Changing this is fast - no re-inference needed."
                }),
                "plot_all_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show all object masks in visualization (True) or only selected obj_id (False)"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, masks, video_state, scores=None, obj_id=-1, plot_all_masks=True):
        # Always re-run this node when params change, but this is cheap
        # The key is that changing these here does NOT invalidate upstream cache
        # ComfyUI caches based on input values - masks/video_state don't change
        return (id(masks), video_state.session_uuid, id(scores), obj_id, plot_all_masks)

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "frames", "visualization")
    FUNCTION = "extract"
    CATEGORY = "SAM3/video"

    def _draw_legend(self, vis_frame, num_objects, colors, obj_id=-1, frame_scores=None):
        """Draw a legend showing object IDs, colors, and confidence scores (sorted by confidence)."""
        h, w = vis_frame.shape[:2]

        # Legend parameters
        box_size = max(16, min(32, h // 20))
        padding = max(4, box_size // 4)
        text_width = box_size * 6  # Space for "X: 0.95"
        legend_item_height = box_size + padding

        # Build list of (obj_id, score) pairs
        if obj_id >= 0:
            items = [(obj_id, frame_scores[obj_id] if frame_scores is not None and obj_id < len(frame_scores) else None)]
        else:
            items = []
            for oid in range(num_objects):
                score = frame_scores[oid] if frame_scores is not None and oid < len(frame_scores) else None
                items.append((oid, score))
            # Sort by score descending (highest confidence first), None scores go last
            items.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))

        num_items = len(items)
        legend_height = num_items * legend_item_height + padding
        legend_width = box_size + text_width + padding * 2

        # Position in top-left corner
        start_x = padding
        start_y = padding

        # Draw semi-transparent background
        bg_alpha = 0.7
        for y in range(start_y, min(start_y + legend_height, h)):
            for x in range(start_x, min(start_x + legend_width, w)):
                vis_frame[y, x] = vis_frame[y, x] * (1 - bg_alpha) + torch.tensor([0.1, 0.1, 0.1]) * bg_alpha

        # Draw legend items (already sorted by confidence)
        for idx, (oid, score) in enumerate(items):
            item_y = start_y + padding + idx * legend_item_height

            # Draw color box
            color = torch.tensor(colors[oid % len(colors)])
            for y in range(item_y, min(item_y + box_size, h)):
                for x in range(start_x + padding, min(start_x + padding + box_size, w)):
                    vis_frame[y, x] = color

            # Draw "X: 0.95" text using simple pixel font
            text_x = start_x + padding + box_size + padding
            if score is not None:
                # Format score to 2 decimal places
                score_str = f"{oid}:{score:.2f}"
            else:
                score_str = f"{oid}"
            self._draw_text(vis_frame, score_str, text_x, item_y, box_size)

        return vis_frame

    def _draw_text(self, img, text, x, y, size):
        """Draw simple text using basic shapes (no font dependencies)."""
        # Simple 3x5 pixel font for digits and punctuation
        chars = {
            '0': [[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]],
            '1': [[0,1,0], [1,1,0], [0,1,0], [0,1,0], [1,1,1]],
            '2': [[1,1,1], [0,0,1], [1,1,1], [1,0,0], [1,1,1]],
            '3': [[1,1,1], [0,0,1], [1,1,1], [0,0,1], [1,1,1]],
            '4': [[1,0,1], [1,0,1], [1,1,1], [0,0,1], [0,0,1]],
            '5': [[1,1,1], [1,0,0], [1,1,1], [0,0,1], [1,1,1]],
            '6': [[1,1,1], [1,0,0], [1,1,1], [1,0,1], [1,1,1]],
            '7': [[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]],
            '8': [[1,1,1], [1,0,1], [1,1,1], [1,0,1], [1,1,1]],
            '9': [[1,1,1], [1,0,1], [1,1,1], [0,0,1], [1,1,1]],
            ':': [[0,0,0], [0,1,0], [0,0,0], [0,1,0], [0,0,0]],
            '.': [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,1,0]],
        }

        h, w = img.shape[:2]
        scale = max(1, size // 6)
        char_width = 4 * scale

        curr_x = x
        for char in text:
            if char in chars:
                pattern = chars[char]
                for row_idx, row in enumerate(pattern):
                    for col_idx, pixel in enumerate(row):
                        if pixel:
                            for sy in range(scale):
                                for sx in range(scale):
                                    px = curr_x + col_idx * scale + sx
                                    py = y + row_idx * scale + sy
                                    if 0 <= px < w and 0 <= py < h:
                                        img[py, px] = torch.tensor([1.0, 1.0, 1.0])
                curr_x += char_width
            elif char == ' ':
                curr_x += char_width  # Space

    def extract(self, masks, video_state, scores=None, obj_id=-1, plot_all_masks=True):
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
            return (empty_mask, empty_frames, empty_frames)

        # Debug: print masks dict info
        print(f"[SAM3 Video DEBUG] masks_dict has {len(masks)} frames")
        print(f"[SAM3 Video DEBUG] masks_dict keys (first 5): {list(masks.keys())[:5]}")
        if len(masks) > 0:
            first_key = list(masks.keys())[0]
            first_mask = masks[first_key]
            print(f"[SAM3 Video DEBUG] First mask (frame {first_key}): type={type(first_mask)}, shape={first_mask.shape if hasattr(first_mask, 'shape') else 'N/A'}")

        # Process all frames in order
        mask_list = []
        frame_list = []
        vis_list = []

        # Color palette for multiple objects (RGB, 0-1 range)
        colors = [
            [0.0, 0.5, 1.0],   # Blue
            [1.0, 0.3, 0.3],   # Red
            [0.3, 1.0, 0.3],   # Green
            [1.0, 1.0, 0.0],   # Yellow
            [1.0, 0.0, 1.0],   # Magenta
            [0.0, 1.0, 1.0],   # Cyan
            [1.0, 0.5, 0.0],   # Orange
            [0.5, 0.0, 1.0],   # Purple
        ]

        # Track number of objects for legend
        num_objects = 0

        for frame_idx in range(num_frames):
            # Load original frame
            frame_path = os.path.join(video_state.temp_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)  # [H, W, C]
            else:
                img_tensor = torch.zeros(h, w, 3)

            frame_list.append(img_tensor)

            # Get mask for this frame
            if frame_idx in masks:
                frame_mask = masks[frame_idx]

                # Debug for first few frames
                if frame_idx < 3:
                    print(f"[SAM3 Video DEBUG] Extract frame {frame_idx}: raw mask type={type(frame_mask)}, shape={frame_mask.shape if hasattr(frame_mask, 'shape') else 'N/A'}")
                    if hasattr(frame_mask, 'dtype'):
                        print(f"[SAM3 Video DEBUG]   dtype={frame_mask.dtype}, sum={frame_mask.sum() if hasattr(frame_mask, 'sum') else 'N/A'}")

                # Convert numpy to torch if needed
                if isinstance(frame_mask, np.ndarray):
                    frame_mask = torch.from_numpy(frame_mask)
                    if frame_idx < 3:
                        print(f"[SAM3 Video DEBUG]   Converted from numpy, new shape={frame_mask.shape}")

                # Convert mask to ComfyUI format
                if frame_mask.dim() == 4:
                    frame_mask = frame_mask.squeeze(0)  # Remove batch dim
                    if frame_idx < 3:
                        print(f"[SAM3 Video DEBUG]   Squeezed dim 4->3, new shape={frame_mask.shape}")

                # Create visualization with colored overlays
                vis_frame = img_tensor.clone()

                # Debug: check mask state before processing
                if frame_idx < 3:
                    print(f"[SAM3 Video DEBUG] Frame {frame_idx} before processing: dim={frame_mask.dim()}, shape={frame_mask.shape}, numel={frame_mask.numel()}")

                # Check for empty mask (no detections)
                if frame_mask.numel() == 0 or (frame_mask.dim() == 3 and frame_mask.shape[0] == 0):
                    # No detections - use empty mask
                    if frame_idx < 3:
                        print(f"[SAM3 Video DEBUG] Frame {frame_idx}: EMPTY MASK branch taken")
                    frame_mask = torch.zeros(h, w)
                    # vis_frame stays as original image
                elif frame_mask.dim() == 3 and frame_mask.shape[0] >= 1:
                    if frame_idx < 3:
                        print(f"[SAM3 Video DEBUG] Frame {frame_idx}: MULTI-OBJECT branch taken, {frame_mask.shape[0]} objects")
                    num_objects = max(num_objects, frame_mask.shape[0])
                    combined_mask = torch.zeros(h, w)

                    if plot_all_masks:
                        # Show ALL objects with different colors
                        for oid in range(frame_mask.shape[0]):
                            obj_mask = frame_mask[oid].float()
                            if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                                obj_mask = obj_mask / 255.0
                            color = torch.tensor(colors[oid % len(colors)])
                            mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                            vis_frame = vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
                            combined_mask = torch.max(combined_mask, obj_mask)
                    else:
                        # Show only selected obj_id
                        vis_oid = obj_id if obj_id >= 0 and obj_id < frame_mask.shape[0] else 0
                        obj_mask = frame_mask[vis_oid].float()
                        if obj_mask.numel() > 0 and obj_mask.max() > 1.0:
                            obj_mask = obj_mask / 255.0
                        color = torch.tensor(colors[vis_oid % len(colors)])
                        mask_rgb = obj_mask.unsqueeze(-1) * color.view(1, 1, 3)
                        vis_frame = vis_frame * (1 - 0.5 * obj_mask.unsqueeze(-1)) + 0.5 * mask_rgb
                        # Still compute combined for mask output
                        for oid in range(frame_mask.shape[0]):
                            om = frame_mask[oid].float()
                            if om.numel() > 0 and om.max() > 1.0:
                                om = om / 255.0
                            combined_mask = torch.max(combined_mask, om)

                    # For mask output, select based on obj_id
                    if obj_id >= 0 and obj_id < frame_mask.shape[0]:
                        output_mask = frame_mask[obj_id].float()
                        if output_mask.numel() > 0 and output_mask.max() > 1.0:
                            output_mask = output_mask / 255.0
                    else:
                        output_mask = combined_mask
                    frame_mask = output_mask
                else:
                    # Single mask
                    if frame_mask.dim() == 3:
                        frame_mask = frame_mask.squeeze(0)
                    frame_mask = frame_mask.float()
                    if frame_mask.numel() > 0 and frame_mask.max() > 1.0:
                        frame_mask = frame_mask / 255.0
                    num_objects = max(num_objects, 1)
                    color = torch.tensor(colors[0])
                    mask_rgb = frame_mask.unsqueeze(-1) * color.view(1, 1, 3)
                    vis_frame = vis_frame * (1 - 0.5 * frame_mask.unsqueeze(-1)) + 0.5 * mask_rgb

                # Final check for empty masks
                if frame_mask.numel() == 0:
                    frame_mask = torch.zeros(h, w)

                # Draw legend on visualization
                if num_objects > 0:
                    legend_obj_id = -1 if plot_all_masks else obj_id
                    # Get scores for this frame
                    frame_scores = None
                    if scores is not None and frame_idx in scores:
                        frame_scores_tensor = scores[frame_idx]
                        if hasattr(frame_scores_tensor, 'tolist'):
                            frame_scores = frame_scores_tensor.tolist()
                            # Handle nested lists (e.g., [[0.95, 0.87]])
                            if frame_scores and isinstance(frame_scores[0], list):
                                frame_scores = frame_scores[0]
                        elif hasattr(frame_scores_tensor, '__iter__'):
                            frame_scores = list(frame_scores_tensor)
                    vis_frame = self._draw_legend(vis_frame, num_objects, colors, obj_id=legend_obj_id, frame_scores=frame_scores)

                vis_list.append(vis_frame.clamp(0, 1))
            else:
                # No mask for this frame - use zeros
                frame_mask = torch.zeros(h, w)
                vis_list.append(img_tensor)

            mask_list.append(frame_mask.cpu())

        # Stack into batches
        all_masks = torch.stack(mask_list, dim=0)  # [N, H, W]
        all_frames = torch.stack(frame_list, dim=0)  # [N, H, W, C]
        all_vis = torch.stack(vis_list, dim=0)  # [N, H, W, C]

        print(f"[SAM3 Video] Output: {all_masks.shape[0]} masks, shape {all_masks.shape}")
        print(f"[SAM3 Video] Objects tracked: {num_objects}, plot_all_masks: {plot_all_masks}")
        print_vram("After extract")

        return (all_masks, all_frames, all_vis)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM3VideoModelLoader": SAM3VideoModelLoader,
    "SAM3VideoSegmentation": SAM3VideoSegmentation,
    "SAM3Propagate": SAM3Propagate,
    "SAM3VideoOutput": SAM3VideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoModelLoader": "SAM3 Video Model Loader",
    "SAM3VideoSegmentation": "SAM3 Video Segmentation",
    "SAM3Propagate": "SAM3 Propagate",
    "SAM3VideoOutput": "SAM3 Video Output",
}
