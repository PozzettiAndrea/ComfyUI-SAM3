"""
SAM3 Video State - Immutable state for stateless video tracking

This module provides immutable state structures that flow through ComfyUI nodes
for video tracking. Instead of global mutable state, all session data is encoded
in the output and reconstructed on-demand.

Key design principles:
1. All state is immutable (frozen dataclasses)
2. Adding prompts returns a NEW state (doesn't mutate)
3. Inference state is reconstructed on-demand
4. Temp directories are automatically cleaned up via atexit
"""
import os
import uuid
import shutil
import tempfile
import weakref
import atexit
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path


# =============================================================================
# Auto-cleanup registry for temporary directories
# =============================================================================

_TEMP_DIR_REGISTRY = set()
_CLEANUP_REGISTERED = False


def _cleanup_temp_dirs():
    """Clean up all registered temporary directories at process exit."""
    for path in list(_TEMP_DIR_REGISTRY):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"[SAM3 Video] Cleaned up temp dir: {path}")
        except Exception as e:
            print(f"[SAM3 Video] Failed to cleanup {path}: {e}")
    _TEMP_DIR_REGISTRY.clear()


def _ensure_cleanup_registered():
    """Register cleanup function with atexit (once)."""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup_temp_dirs)
        _CLEANUP_REGISTERED = True


def create_temp_dir(session_uuid: str) -> str:
    """
    Create a temporary directory for video frames with auto-cleanup.

    The directory is registered for automatic cleanup at process exit,
    even if the session is never explicitly closed.

    Args:
        session_uuid: Session identifier for naming

    Returns:
        Path to the created temporary directory
    """
    _ensure_cleanup_registered()
    temp_dir = tempfile.mkdtemp(prefix=f"sam3_{session_uuid[:8]}_")
    _TEMP_DIR_REGISTRY.add(temp_dir)
    return temp_dir


def cleanup_temp_dir(temp_dir: str):
    """
    Explicitly cleanup a temporary directory.

    Also removes it from the registry to avoid double cleanup.

    Args:
        temp_dir: Path to cleanup
    """
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    _TEMP_DIR_REGISTRY.discard(temp_dir)


# =============================================================================
# Immutable State Structures
# =============================================================================

@dataclass(frozen=True)
class VideoPrompt:
    """
    Single prompt for a video frame (immutable).

    Attributes:
        frame_idx: Frame index where prompt is applied
        prompt_type: Type of prompt ("point", "box", "text")
        obj_id: Object ID for tracking
        data: Prompt-specific data (tuple for immutability)
    """
    frame_idx: int
    prompt_type: str  # "point", "box", "text"
    obj_id: int
    data: tuple  # Immutable representation of prompt data

    @classmethod
    def create_point(cls, frame_idx: int, obj_id: int, points: list, labels: list) -> 'VideoPrompt':
        """Create a point prompt."""
        return cls(
            frame_idx=frame_idx,
            prompt_type="point",
            obj_id=obj_id,
            data=(tuple(tuple(p) for p in points), tuple(labels))
        )

    @classmethod
    def create_box(cls, frame_idx: int, obj_id: int, box: list, is_positive: bool = True) -> 'VideoPrompt':
        """Create a box prompt."""
        return cls(
            frame_idx=frame_idx,
            prompt_type="box",
            obj_id=obj_id,
            data=(tuple(box), is_positive)
        )

    @classmethod
    def create_text(cls, frame_idx: int, obj_id: int, text: str) -> 'VideoPrompt':
        """Create a text prompt."""
        return cls(
            frame_idx=frame_idx,
            prompt_type="text",
            obj_id=obj_id,
            data=(text,)
        )


@dataclass(frozen=True)
class VideoConfig:
    """
    Video tracking configuration (immutable).

    Contains all the detection and tracking parameters.
    """
    score_threshold_detection: float = 0.3
    new_det_thresh: float = 0.4
    fill_hole_area: int = 16
    assoc_iou_thresh: float = 0.1
    det_nms_thresh: float = 0.1
    hotstart_unmatch_thresh: int = 8
    hotstart_dup_thresh: int = 8
    init_trk_keep_alive: int = 30
    hotstart_delay: int = 15
    decrease_keep_alive_empty: bool = False
    suppress_unmatched_globally: bool = False


@dataclass(frozen=True)
class SAM3VideoState:
    """
    Immutable video session state that flows through node outputs.

    All state needed to reconstruct inference is encoded here.
    This enables truly stateless nodes - no global mutable state needed.

    Attributes:
        session_uuid: Unique session identifier
        temp_dir: Path to temporary directory containing frames
        num_frames: Total number of video frames
        height: Video height in pixels
        width: Video width in pixels
        config: Tracking configuration
        prompts: Tuple of all prompts (immutable)
    """
    session_uuid: str
    temp_dir: str
    num_frames: int
    height: int
    width: int
    config: VideoConfig = field(default_factory=VideoConfig)
    prompts: Tuple[VideoPrompt, ...] = ()

    def with_prompt(self, prompt: VideoPrompt) -> 'SAM3VideoState':
        """
        Return new state with prompt added (immutable update).

        This is the key pattern - instead of mutating, we create a new state.

        Args:
            prompt: VideoPrompt to add

        Returns:
            New SAM3VideoState with prompt appended
        """
        return SAM3VideoState(
            session_uuid=self.session_uuid,
            temp_dir=self.temp_dir,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            config=self.config,
            prompts=self.prompts + (prompt,),
        )

    def with_config(self, **kwargs) -> 'SAM3VideoState':
        """
        Return new state with updated config.

        Args:
            **kwargs: Config parameters to update

        Returns:
            New SAM3VideoState with updated config
        """
        from dataclasses import asdict
        current_config = asdict(self.config)
        current_config.update(kwargs)
        new_config = VideoConfig(**current_config)

        return SAM3VideoState(
            session_uuid=self.session_uuid,
            temp_dir=self.temp_dir,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            config=new_config,
            prompts=self.prompts,
        )

    def get_prompts_for_frame(self, frame_idx: int) -> Tuple[VideoPrompt, ...]:
        """Get all prompts for a specific frame."""
        return tuple(p for p in self.prompts if p.frame_idx == frame_idx)

    def get_object_ids(self) -> Tuple[int, ...]:
        """Get all unique object IDs in prompts."""
        return tuple(sorted(set(p.obj_id for p in self.prompts)))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to dict for ComfyUI compatibility.

        Returns:
            Dict representation of state
        """
        from dataclasses import asdict
        return {
            "session_uuid": self.session_uuid,
            "temp_dir": self.temp_dir,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "config": asdict(self.config),
            "prompts": [
                {
                    "frame_idx": p.frame_idx,
                    "prompt_type": p.prompt_type,
                    "obj_id": p.obj_id,
                    "data": p.data
                }
                for p in self.prompts
            ]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SAM3VideoState':
        """
        Deserialize state from dict.

        Args:
            d: Dict representation

        Returns:
            SAM3VideoState instance
        """
        prompts = tuple(
            VideoPrompt(
                frame_idx=p["frame_idx"],
                prompt_type=p["prompt_type"],
                obj_id=p["obj_id"],
                data=tuple(p["data"]) if isinstance(p["data"], list) else p["data"]
            )
            for p in d.get("prompts", [])
        )

        config = VideoConfig(**d.get("config", {}))

        return cls(
            session_uuid=d["session_uuid"],
            temp_dir=d["temp_dir"],
            num_frames=d["num_frames"],
            height=d["height"],
            width=d["width"],
            config=config,
            prompts=prompts,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_video_state(
    video_frames,
    config: Optional[VideoConfig] = None,
    session_id: Optional[str] = None
) -> SAM3VideoState:
    """
    Create a new video state from video frames.

    This function:
    1. Generates a unique session UUID
    2. Creates a temp directory for frames
    3. Saves frames as JPEG files
    4. Returns immutable state

    Args:
        video_frames: ComfyUI image tensor [N, H, W, C]
        config: Optional tracking configuration
        session_id: Optional custom session ID (UUID generated if None)

    Returns:
        SAM3VideoState ready for use
    """
    import numpy as np
    from PIL import Image

    session_uuid = session_id if session_id else str(uuid.uuid4())
    temp_dir = create_temp_dir(session_uuid)

    # Save frames as JPEG
    num_frames = video_frames.shape[0]
    height = video_frames.shape[1]
    width = video_frames.shape[2]

    print(f"[SAM3 Video] Saving {num_frames} frames to {temp_dir}")

    for i in range(num_frames):
        frame = video_frames[i].cpu().numpy()
        # Convert from [H, W, C] float32 0-1 to uint8 0-255
        frame = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame)
        img.save(os.path.join(temp_dir, f"{i:05d}.jpg"))

    print(f"[SAM3 Video] Frames saved successfully")

    return SAM3VideoState(
        session_uuid=session_uuid,
        temp_dir=temp_dir,
        num_frames=num_frames,
        height=height,
        width=width,
        config=config or VideoConfig(),
        prompts=(),
    )
