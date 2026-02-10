"""
Inference State Reconstructor - On-demand reconstruction of video inference state

This module provides lazy reconstruction of SAM3 video inference states from
immutable SAM3VideoState objects. Instead of keeping inference states alive
globally, we reconstruct them on-demand and use weak references for caching.

Key design principles:
1. Inference state is reconstructed when needed
2. WeakValueDictionary allows automatic cleanup when not referenced
3. Cache is invalidated when prompts change
4. All reconstruction is from immutable state
"""
import weakref
import gc
import torch
from typing import Optional, Dict, Any

from .video_state import SAM3VideoState, VideoPrompt


def print_vram(label: str):
    """Print current VRAM usage for debugging memory leaks."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label}: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")


class InferenceReconstructor:
    """
    Reconstructs video inference state from immutable SAM3VideoState.

    Uses weak references for automatic memory cleanup when inference
    state is no longer needed.

    This class is a singleton to provide a global cache that can be
    invalidated when prompts change.
    """

    _instance: Optional['InferenceReconstructor'] = None
    _cache: weakref.WeakValueDictionary

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = weakref.WeakValueDictionary()
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'InferenceReconstructor':
        """Get the singleton instance."""
        return cls()

    def get_inference_state(
        self,
        model,
        video_state: SAM3VideoState,
        force_reconstruct: bool = False
    ) -> Dict[str, Any]:
        """
        Get or reconstruct inference state for a video session.

        Args:
            model: SAM3 video predictor model
            video_state: Immutable video state
            force_reconstruct: Force reconstruction even if cached

        Returns:
            Inference state dict ready for propagation
        """
        # Create cache key from session UUID and prompt count
        # (prompt count ensures we reconstruct when prompts change)
        cache_key = f"{video_state.session_uuid}_{len(video_state.prompts)}"

        if not force_reconstruct and cache_key in self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                print(f"[SAM3 Video] Using cached inference state for {video_state.session_uuid[:8]}")
                return cached

        print(f"[SAM3 Video] Reconstructing inference state for {video_state.session_uuid[:8]}")
        print_vram("Before start_session")

        # CRITICAL: Close ALL existing sessions to prevent VRAM leak
        # _ALL_INFERENCE_STATES is a class variable that persists across model reloads
        existing_sessions = list(model._ALL_INFERENCE_STATES.keys())
        if existing_sessions:
            print(f"[SAM3 Video] Closing {len(existing_sessions)} old sessions to free VRAM")
            for old_session_id in existing_sessions:
                try:
                    model.close_session(old_session_id)
                except Exception as e:
                    print(f"[SAM3 Video] Warning: Failed to close session {old_session_id[:8]}: {e}")
            print_vram("After closing old sessions")

        # Apply config to model
        self._apply_config(model, video_state.config)

        # Initialize fresh inference state - this stores state in predictor's _ALL_INFERENCE_STATES
        inference_state = model.start_session(
            resource_path=video_state.temp_dir,
            session_id=video_state.session_uuid
        )
        print_vram("After start_session")

        # Re-apply all prompts in order using session_id
        for prompt in video_state.prompts:
            self._apply_prompt(model, video_state.session_uuid, prompt)
            print_vram(f"After apply prompt obj={prompt.obj_id}")

        # Store with weak reference
        # Create a wrapper to allow weak referencing
        state_wrapper = InferenceStateWrapper(inference_state, video_state.session_uuid)
        self._cache[cache_key] = state_wrapper

        return inference_state

    def _apply_config(self, model, config):
        """Apply VideoConfig to model."""
        model.model.score_threshold_detection = config.score_threshold_detection
        model.model.new_det_thresh = config.new_det_thresh
        model.model.fill_hole_area = config.fill_hole_area
        model.model.assoc_iou_thresh = config.assoc_iou_thresh
        model.model.det_nms_thresh = config.det_nms_thresh
        model.model.hotstart_unmatch_thresh = config.hotstart_unmatch_thresh
        model.model.hotstart_dup_thresh = config.hotstart_dup_thresh
        model.model.init_trk_keep_alive = config.init_trk_keep_alive
        model.model.hotstart_delay = config.hotstart_delay
        model.model.decrease_trk_keep_alive_for_empty_masklets = config.decrease_keep_alive_empty
        model.model.suppress_unmatched_only_within_hotstart = not config.suppress_unmatched_globally

    def _apply_prompt(self, model, session_id: str, prompt: VideoPrompt):
        """Apply a single prompt to inference state using the predictor's API."""

        if prompt.prompt_type == "point":
            points, labels = prompt.data
            # Convert tuples back to lists for the model
            points_list = [list(p) for p in points]
            labels_list = list(labels)

            print(f"[SAM3 Video] Applying point prompt: frame={prompt.frame_idx}, obj={prompt.obj_id}")
            print(f"[SAM3 Video] Points to model: {points_list}, labels: {labels_list}")

            model.add_prompt(
                session_id=session_id,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.obj_id,
                points=points_list,
                point_labels=labels_list,
            )

        elif prompt.prompt_type == "box":
            # Handle both old format (just box coords) and new format (box coords, is_positive)
            if isinstance(prompt.data[0], (list, tuple)) or (len(prompt.data) == 2 and isinstance(prompt.data[1], bool)):
                # New format: (box_coords, is_positive)
                box = list(prompt.data[0])
                is_positive = prompt.data[1] if len(prompt.data) > 1 else True
            else:
                # Old format: just box coords
                box = list(prompt.data)
                is_positive = True

            # Convert box to two corner points with special labels (2=top-left, 3=bottom-right)
            # This is how SAM2/SAM3 tracker handles boxes internally - as two points with labels 2,3
            x1, y1, x2, y2 = box
            points = [[x1, y1], [x2, y2]]
            labels = [2, 3]  # Special box corner labels used by SAM tracker

            print(f"[SAM3 Video] Applying box as corner points: frame={prompt.frame_idx}, obj={prompt.obj_id}")
            print(f"[SAM3 Video] Box [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}] -> points with labels [2, 3]")

            model.add_prompt(
                session_id=session_id,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.obj_id,
                points=points,
                point_labels=labels,
            )

        elif prompt.prompt_type == "text":
            text = prompt.data[0]

            print(f"[SAM3 Video] Applying text prompt: frame={prompt.frame_idx}, obj={prompt.obj_id}")

            model.add_prompt(
                session_id=session_id,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.obj_id,
                text=text,
            )

    def invalidate(self, session_uuid: str):
        """
        Invalidate all cached states for a session.

        Called when prompts change to ensure reconstruction.

        Args:
            session_uuid: Session identifier to invalidate
        """
        # Remove all keys that start with this session UUID
        keys_to_remove = [
            k for k in self._cache.keys()
            if k.startswith(session_uuid)
        ]
        for key in keys_to_remove:
            try:
                del self._cache[key]
            except KeyError:
                pass

        print(f"[SAM3 Video] Invalidated cache for {session_uuid[:8]}")

    def clear_all(self):
        """Clear all cached inference states."""
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SAM3 Video] Cleared all inference state cache")


class InferenceStateWrapper:
    """
    Wrapper for inference state to allow weak referencing.

    Python dicts can't be weakly referenced directly, so we wrap them.
    """

    def __init__(self, inference_state: dict, session_uuid: str):
        self._state = inference_state
        self._session_uuid = session_uuid

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def get(self, key, default=None):
        return self._state.get(key, default)

    @property
    def state(self) -> dict:
        return self._state

    def __del__(self):
        """Cleanup when wrapper is garbage collected."""
        print(f"[SAM3 Video] Inference state for {self._session_uuid[:8]} garbage collected")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_inference_state(model, video_state: SAM3VideoState) -> Dict[str, Any]:
    """
    Get inference state for a video session.

    Convenience function that uses the singleton reconstructor.

    Args:
        model: SAM3 video predictor
        video_state: Immutable video state

    Returns:
        Inference state dict
    """
    return InferenceReconstructor.get_instance().get_inference_state(model, video_state)


def invalidate_session(session_uuid: str):
    """
    Invalidate cached inference state for a session.

    Call this when prompts change.

    Args:
        session_uuid: Session to invalidate
    """
    InferenceReconstructor.get_instance().invalidate(session_uuid)


def clear_inference_cache():
    """Clear all cached inference states."""
    InferenceReconstructor.get_instance().clear_all()
