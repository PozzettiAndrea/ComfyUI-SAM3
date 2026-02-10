"""
SAM3 Video Tracking Nodes for ComfyUI

These nodes provide video object tracking and segmentation capabilities using SAM3.
They use the vendored sam3_lib with full video support.
"""

from pathlib import Path
import torch
import numpy as np
import folder_paths

from ..sam3_lib.model_builder import build_sam3_video_predictor


class SAM3VideoModelLoader:
    """Load SAM3 model for video tracking"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "bpe_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "use_gpu_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between inferences (faster but uses more VRAM). Set to False to offload to CPU after each inference."
                }),
            },
            "optional": {
                "use_multi_gpu": ("BOOLEAN", {"default": False}),
                "gpu_ids": ("STRING", {
                    "default": "0",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MODEL",)
    RETURN_NAMES = ("video_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3/video"

    def load_model(self, checkpoint_path, bpe_path, hf_token, use_gpu_cache=True, use_multi_gpu=False, gpu_ids="0"):
        """Load the SAM3 video model"""
        import os
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Parse GPU IDs if using multi-GPU
        gpus_to_use = None
        if use_multi_gpu:
            gpus_to_use = [int(x.strip()) for x in gpu_ids.split(",")]

        print(f"[SAM3 Video] Loading video model from {checkpoint_path}")
        print(f"[SAM3 Video] GPU cache: {'enabled' if use_gpu_cache else 'disabled (will offload to CPU after inference)'}")

        # Build the video predictor
        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path if checkpoint_path else None,
            bpe_path=bpe_path if bpe_path else None,
            gpus_to_use=gpus_to_use,
        )

        print(f"[SAM3 Video] Model loaded successfully")

        # Store the use_gpu_cache setting in the predictor
        predictor.use_gpu_cache = use_gpu_cache

        return (predictor,)


class SAM3InitVideoSession:
    """Initialize a video tracking session"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_model": ("SAM3_VIDEO_MODEL",),
                "video_frames": ("IMAGE",),  # ComfyUI video frames [B, H, W, C]
            },
            "optional": {
                "session_id": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_SESSION", "STRING")
    RETURN_NAMES = ("session", "session_id")
    FUNCTION = "init_session"
    CATEGORY = "SAM3/video"

    def init_session(self, video_model, video_frames, session_id=""):
        """Initialize a tracking session with video frames"""
        # Convert ComfyUI frames to temporary directory
        import tempfile
        import os
        from PIL import Image

        # Create a temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix="sam3_video_")

        # Save frames as JPEG
        num_frames = video_frames.shape[0]
        for i in range(num_frames):
            frame = video_frames[i].cpu().numpy()
            # Convert from [H, W, C] float32 0-1 to uint8 0-255
            frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            img.save(os.path.join(temp_dir, f"{i:05d}.jpg"))

        print(f"[SAM3 Video] Saved {num_frames} frames to {temp_dir}")

        # Start the session
        response = video_model.start_session(
            resource_path=temp_dir,
            session_id=session_id if session_id else None
        )

        actual_session_id = response["session_id"]

        session_data = {
            "model": video_model,
            "session_id": actual_session_id,
            "temp_dir": temp_dir,
            "num_frames": num_frames,
            "height": video_frames.shape[1],
            "width": video_frames.shape[2],
        }

        print(f"[SAM3 Video] Initialized session {actual_session_id}")

        return (session_data, actual_session_id)


class SAM3AddVideoPrompt:
    """Add a text, box, or point prompt to a specific video frame"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("SAM3_VIDEO_SESSION",),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "obj_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "boxes": ("SAM3_BOXES_PROMPT",),
                "points": ("SAM3_POINTS_PROMPT",),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_SESSION",)
    RETURN_NAMES = ("session",)
    FUNCTION = "add_prompt"
    CATEGORY = "SAM3/video"

    def add_prompt(self, session, frame_index, text_prompt="", obj_id=1, boxes=None, points=None):
        """Add a prompt on a specific frame"""
        video_model = session["model"]
        session_id = session["session_id"]

        # Ensure model is on GPU if use_gpu_cache is False (model may have been offloaded)
        if hasattr(video_model, 'use_gpu_cache') and not video_model.use_gpu_cache:
            if hasattr(video_model, 'model'):
                current_device = next(video_model.model.parameters()).device
                if "cpu" in str(current_device):
                    print(f"[SAM3 Video] Moving model from CPU to GPU for inference")
                    video_model.model.to("cuda")

        # Prepare prompt parameters
        bounding_boxes = None
        bounding_box_labels = None
        point_coords = None
        point_labels = None

        if boxes is not None:
            bounding_boxes = boxes["boxes"]
            bounding_box_labels = boxes.get("labels", None)

        if points is not None:
            point_coords = points["points"]
            point_labels = points["labels"]

        print(f"[SAM3 Video] Adding prompt on frame {frame_index}: text='{text_prompt}', obj_id={obj_id}")

        # Add the prompt
        response = video_model.add_prompt(
            session_id=session_id,
            frame_idx=frame_index,
            text=text_prompt if text_prompt else None,
            points=point_coords,
            point_labels=point_labels,
            bounding_boxes=bounding_boxes,
            bounding_box_labels=bounding_box_labels,
            obj_id=obj_id,
        )

        print(f"[SAM3 Video] Prompt added successfully")

        return (session,)


class SAM3PropagateVideo:
    """Propagate masks through the entire video"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("SAM3_VIDEO_SESSION",),
            },
            "optional": {
                "propagation_direction": (["both", "forward", "backward"], {"default": "both"}),
                "start_frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "max_frames": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS",)
    RETURN_NAMES = ("video_masks",)
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    def propagate(self, session, propagation_direction="both", start_frame_index=0, max_frames=-1):
        """Propagate the prompts through all video frames"""
        video_model = session["model"]
        session_id = session["session_id"]
        num_frames = session["num_frames"]

        # Ensure model is on GPU if use_gpu_cache is False (model may have been offloaded)
        if hasattr(video_model, 'use_gpu_cache') and not video_model.use_gpu_cache:
            if hasattr(video_model, 'model'):
                current_device = next(video_model.model.parameters()).device
                if "cpu" in str(current_device):
                    print(f"[SAM3 Video] Moving model from CPU to GPU for inference")
                    video_model.model.to("cuda")

        max_frame_num_to_track = max_frames if max_frames > 0 else num_frames

        print(f"[SAM3 Video] Propagating in {propagation_direction} direction from frame {start_frame_index}")

        # Collect all frames' masks
        all_masks = {}
        all_obj_ids = None

        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": start_frame_index,
            "max_frame_num_to_track": max_frame_num_to_track,
        }

        for response in video_model.handle_stream_request(request):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            all_masks[frame_idx] = outputs
            if all_obj_ids is None:
                all_obj_ids = outputs.get("obj_ids", [])

        print(f"[SAM3 Video] Propagation complete. Tracked {len(all_masks)} frames, {len(all_obj_ids)} objects")

        video_masks = {
            "session": session,
            "masks": all_masks,
            "obj_ids": all_obj_ids,
            "num_frames": num_frames,
        }

        return (video_masks,)


class SAM3VideoOutput:
    """Output video masks as ComfyUI tensors"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_masks": ("SAM3_VIDEO_MASKS",),
            },
            "optional": {
                "obj_id_filter": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "output_masks"
    CATEGORY = "SAM3/video"

    def output_masks(self, video_masks, obj_id_filter=-1):
        """Convert video masks to ComfyUI mask format"""
        masks_dict = video_masks["masks"]
        obj_ids = video_masks["obj_ids"]
        num_frames = video_masks["num_frames"]
        session = video_masks["session"]
        height = session["height"]
        width = session["width"]

        # Create output tensor [N, H, W] where N is number of frames
        output_masks = torch.zeros((num_frames, height, width), dtype=torch.float32)

        for frame_idx in range(num_frames):
            if frame_idx in masks_dict:
                frame_output = masks_dict[frame_idx]

                # Get masks for this frame
                if "video_res_masks" in frame_output:
                    frame_masks = frame_output["video_res_masks"]
                elif "pred_masks" in frame_output:
                    # Resize if needed
                    frame_masks = frame_output["pred_masks"]
                    if frame_masks.shape[-2:] != (height, width):
                        frame_masks = torch.nn.functional.interpolate(
                            frame_masks,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                else:
                    continue

                # Filter by object ID if specified
                if obj_id_filter > 0:
                    try:
                        obj_idx = obj_ids.index(obj_id_filter)
                        mask = frame_masks[obj_idx, 0] > 0.0
                    except (ValueError, IndexError):
                        mask = torch.zeros((height, width), dtype=torch.bool)
                else:
                    # Combine all object masks
                    mask = (frame_masks[:, 0] > 0.0).any(dim=0)

                output_masks[frame_idx] = mask.float().cpu()

        print(f"[SAM3 Video] Output {num_frames} mask frames")

        # Offload model to CPU if use_gpu_cache is False
        video_model = session["model"]
        if hasattr(video_model, 'use_gpu_cache') and not video_model.use_gpu_cache:
            if hasattr(video_model, 'model'):
                current_device = next(video_model.model.parameters()).device
                if "cuda" in str(current_device):
                    print(f"[SAM3 Video] Offloading model to CPU to free VRAM")
                    video_model.model.to("cpu")
                    import torch
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

        return (output_masks,)


class SAM3CloseVideoSession:
    """Close a video tracking session and cleanup resources"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("SAM3_VIDEO_SESSION",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "close_session"
    CATEGORY = "SAM3/video"

    def close_session(self, session):
        """Close the session and cleanup temporary files"""
        import shutil

        video_model = session["model"]
        session_id = session["session_id"]
        temp_dir = session["temp_dir"]

        # Close the session
        video_model.close_session(session_id)

        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            print(f"[SAM3 Video] Cleaned up temp directory: {temp_dir}")

        print(f"[SAM3 Video] Closed session {session_id}")

        return ()


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SAM3VideoModelLoader": SAM3VideoModelLoader,
    "SAM3InitVideoSession": SAM3InitVideoSession,
    "SAM3AddVideoPrompt": SAM3AddVideoPrompt,
    "SAM3PropagateVideo": SAM3PropagateVideo,
    "SAM3VideoOutput": SAM3VideoOutput,
    "SAM3CloseVideoSession": SAM3CloseVideoSession,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoModelLoader": "SAM3 Load Video Model",
    "SAM3InitVideoSession": "SAM3 Init Video Session",
    "SAM3AddVideoPrompt": "SAM3 Add Video Prompt",
    "SAM3PropagateVideo": "SAM3 Propagate Video",
    "SAM3VideoOutput": "SAM3 Video Output",
    "SAM3CloseVideoSession": "SAM3 Close Video Session",
}
