"""
Helpers for finding ComfyUI core modules from comfy-env isolation workers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_comfyui_base(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "folder_paths.py").exists()
        and (path / "comfy_api").is_dir()
    )


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []

    for key in ("COMFYUI_BASE", "COMFYUI_PATH", "COMFYUI_DIR"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))

    host_python_dir = os.environ.get("COMFYUI_HOST_PYTHON_DIR")
    if host_python_dir:
        host_dir = Path(host_python_dir)
        candidates.extend(
            [
                host_dir,
                host_dir.parent,
                host_dir.parent.parent,
                host_dir.parent.parent.parent,
                host_dir.parent.parent / "resources" / "ComfyUI",
            ]
        )

    local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    candidates.append(local_appdata / "Programs" / "ComfyUI" / "resources" / "ComfyUI")

    cwd = Path.cwd()
    candidates.extend([cwd, *list(cwd.parents)[:4]])

    here = Path(__file__).resolve()
    candidates.extend(list(here.parents)[:6])

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def ensure_comfyui_base() -> Path | None:
    """Add the ComfyUI core directory to sys.path if it can be located."""
    for candidate in _candidate_paths():
        if not _is_comfyui_base(candidate):
            continue

        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        os.environ.setdefault("COMFYUI_BASE", candidate_str)
        return candidate

    return None
