"""
SAM3 Video Tracking Nodes for ComfyUI
Multi-Prompt Edition (Points + Boxes + Text + Precision Coordinate Scaling)
"""

import os
import json
import base64
import hashlib
import logging
import tempfile
import shutil
import asyncio
from io import BytesIO
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from aiohttp import web
from server import PromptServer
import folder_paths
from .utils import get_comfy_models_dir
from .sam3_lib.model_builder import build_sam3_video_predictor

log = logging.getLogger("sam3.video")

_VIDEO_MODEL_CACHE = {}
_SESSION_FRAMES_CACHE = {}
_PREVIEW_CACHE = {}

def _frames_signature(video_frames):
    if video_frames is None or not hasattr(video_frames, "shape"):
        return "none"
    n, h, w = int(video_frames.shape[0]), int(video_frames.shape[1]), int(video_frames.shape[2])
    try:
        sub_0 = video_frames[0, ::32, ::32].contiguous().cpu().numpy()
        sub_m = video_frames[n // 2, ::32, ::32].contiguous().cpu().numpy()
        sub_l = video_frames[n - 1, ::32, ::32].contiguous().cpu().numpy()
        val_hash = hashlib.md5(sub_0.tobytes() + sub_m.tobytes() + sub_l.tobytes()).hexdigest()
    except Exception:
        val_hash = "error"
    return f"{n}x{h}x{w}_{val_hash}"

def _encode_frames_for_ui(frames, max_w=640, q=75):
    out = []
    for i in range(int(frames.shape[0])):
        arr = (np.clip(frames[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if pil.width > max_w:
            pil = pil.resize((max_w, int(pil.height * max_w / pil.width)), Image.BILINEAR)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out

def _extract_frames_from_file(filename, max_w=640, max_frames=150, q=75):
    file_path = None
    try: file_path = folder_paths.get_annotated_filepath(filename)
    except: pass

    if not file_path or not os.path.exists(file_path):
        p = Path(filename)
        if p.exists(): file_path = str(p)
        else:
            cand = Path(folder_paths.get_input_directory()) / filename
            if cand.exists(): file_path = str(cand)

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename}")

    out_b64 = []
    w, h = 0, 0
    try:
        im = Image.open(file_path)
        if getattr(im, "is_animated", False):
            for i in range(min(im.n_frames, max_frames)):
                im.seek(i)
                f = im.convert("RGB")
                if i == 0: w, h = f.width, f.height
                if f.width > max_w: f = f.resize((max_w, int(f.height * max_w / f.width)), Image.BILINEAR)
                buf = BytesIO()
                f.save(buf, format="JPEG", quality=q)
                out_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            return out_b64, w, h
        else:
            f = im.convert("RGB")
            w, h = f.width, f.height
            if f.width > max_w: f = f.resize((max_w, int(f.height * max_w / f.width)), Image.BILINEAR)
            buf = BytesIO()
            f.save(buf, format="JPEG", quality=q)
            out_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            return out_b64, w, h
    except: pass

    try:
        import cv2
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            c = 0
            while cap.isOpened() and c < max_frames:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pi = Image.fromarray(frame)
                if pi.width > max_w: pi = pi.resize((max_w, int(pi.height * max_w / pi.width)), Image.BILINEAR)
                buf = BytesIO()
                pi.save(buf, format="JPEG", quality=q)
                out_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                c += 1
            cap.release()
            if out_b64: return out_b64, w, h
    except: pass
    raise RuntimeError(f"Could not read frames from: {file_path}")

def _session_alive(vm, sid):
    if not sid: return False
    try:
        g = getattr(vm, "_get_session", None)
        if g is None: return True
        g(sid)
        return True
    except: return False

def _ensure_video_model_on_device(vm):
    m = getattr(vm, "model", None)
    if m is None: return False
    try: current_device = next(m.parameters()).device
    except StopIteration: return False
    target = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if current_device != target:
        m.to(device=target)
        return True
    return False

def _video_autocast(vm):
    m = getattr(vm, "model", None)
    if m is None: return nullcontext()
    try: device = next(m.parameters()).device
    except StopIteration: return nullcontext()
    if device.type != "cuda": return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

def _video_add_prompt(session, frame_idx, obj_id, points=None, point_labels=None, boxes=None, box_labels=None, text=None):
    vm = session["model"]
    _ensure_video_model_on_device(vm)

    p_arr = np.array(points, dtype=np.float32) if points is not None and len(points) > 0 else None
    pl_arr = np.array(point_labels, dtype=np.int32) if point_labels is not None and len(point_labels) > 0 else None

    b_arr = np.array(boxes, dtype=np.float32) if boxes is not None and len(boxes) > 0 else None
    bl_arr = np.array(box_labels, dtype=np.int32) if box_labels is not None and len(box_labels) > 0 else None

    txt = text.strip() if text and text.strip() else None

    with torch.inference_mode():
        with _video_autocast(vm):
            return vm.add_prompt(
                session_id=session["session_id"],
                frame_idx=int(frame_idx),
                text=txt,
                points=p_arr,
                point_labels=pl_arr,
                bounding_boxes=b_arr,
                bounding_box_labels=bl_arr,
                obj_id=int(obj_id),
            )

def _recover_session(session, skip_node=None):
    vm = session["model"]
    frames = session.get("frames")
    if frames is None: raise RuntimeError("[SAM3] Cannot recover session: frames missing.")
    _ensure_video_model_on_device(vm)
    old_sid = session.get("session_id")
    if old_sid:
        try: vm.close_session(old_sid)
        except: pass

    tdir = session.get("temp_dir")
    if not tdir or not Path(tdir).exists():
        tdir = tempfile.mkdtemp(prefix="sam3_video_")
        for i in range(int(frames.shape[0])):
            arr = (np.clip(frames[i].detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(tdir, f"{i:05d}.jpg"))
        session["temp_dir"] = tdir

    with torch.inference_mode():
        with _video_autocast(vm):
            resp = vm.start_session(resource_path=tdir, session_id=None)
    session["session_id"] = resp["session_id"]

    if old_sid in _SESSION_FRAMES_CACHE:
        _SESSION_FRAMES_CACHE[session["session_id"]] = _SESSION_FRAMES_CACHE.pop(old_sid)

    for entry in session.get("_prompts_seq", []):
        if skip_node is not None and str(entry.get("node")) == str(skip_node): continue
        m = getattr(vm, "model", None)
        old_t = getattr(m, "score_threshold_detection", None)
        try:
            if old_t is not None: m.score_threshold_detection = float(entry.get("threshold", old_t))
            
            ebixes = entry.get("boxes")
            epoints = entry.get("points")
            etxt = entry.get("text")

            # Двухэтапная отправка для обхода AssertionError в SAM3 core
            if ebixes or etxt:
                _video_add_prompt(
                    session=session, frame_idx=entry["frame_idx"], obj_id=entry["obj_id"],
                    points=None, point_labels=None,
                    boxes=ebixes, box_labels=entry.get("box_labels"), text=etxt
                )
            if epoints:
                _video_add_prompt(
                    session=session, frame_idx=entry["frame_idx"], obj_id=entry["obj_id"],
                    points=epoints, point_labels=entry.get("point_labels"),
                    boxes=None, box_labels=None, text=None
                )
        finally:
            if old_t is not None: m.score_threshold_detection = old_t

def _is_recoverable_session_error(error):
    m = str(error).lower()
    return any(x in m for x in ("cannot find session", "might have expired", "input type", "weight type", "bias type", "should be the same", "expected all tensors"))

# REST API
@PromptServer.instance.routes.post("/sam3/prepare_frames")
async def sam3_prepare_frames(request):
    try:
        body = await request.json()
        key = body.get("preview_key")
        filename = body.get("filename")
        if not key or not filename: return web.json_response({"error": "missing key/filename"}, status=400)
        if key in _PREVIEW_CACHE:
            c = _PREVIEW_CACHE[key]
            return web.json_response({"cached": True, "n": c["n"], "w": c["w"], "h": c["h"], "preview_key": key})
        loop = asyncio.get_event_loop()
        b64_list, w, h = await loop.run_in_executor(None, _extract_frames_from_file, filename)
        n = len(b64_list)
        _PREVIEW_CACHE[key] = {"b64": b64_list, "w": w, "h": h, "n": n}
        return web.json_response({"cached": False, "n": n, "w": w, "h": h, "preview_key": key})
    except Exception as e: return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/sam3/preview_frames/{key}/{idx}")
async def sam3_preview_frame(request):
    key = request.match_info["key"]
    idx = int(request.match_info["idx"])
    c = _PREVIEW_CACHE.get(key)
    if not c or idx < 0 or idx >= c["n"]: return web.json_response({"error": "not found"}, status=404)
    return web.json_response({"preview_key": key, "idx": idx, "n": c["n"], "w": c["w"], "h": c["h"], "b64": c["b64"][idx]})

@PromptServer.instance.routes.get("/sam3/video_frames/list")
async def sam3_video_frames_list(request):
    return web.json_response({"sessions": [{"sid": k, "n": v["n"], "w": v["w"], "h": v["h"]} for k, v in _SESSION_FRAMES_CACHE.items()]})

@PromptServer.instance.routes.get("/sam3/video_frames/{sid}")
async def sam3_video_frames_meta(request):
    sid = request.match_info["sid"]
    c = _SESSION_FRAMES_CACHE.get(sid)
    if not c: return web.json_response({"error": "not found"}, status=404)
    return web.json_response({"sid": sid, "n": c["n"], "w": c["w"], "h": c["h"]})

@PromptServer.instance.routes.get("/sam3/video_frames/{sid}/{idx}")
async def sam3_video_frames_get(request):
    sid = request.match_info["sid"]
    idx = int(request.match_info["idx"])
    c = _SESSION_FRAMES_CACHE.get(sid)
    if not c or idx < 0 or idx >= c["n"]: return web.json_response({"error": "not found"}, status=404)
    return web.json_response({"sid": sid, "idx": idx, "n": c["n"], "w": c["w"], "h": c["h"], "b64": c["b64"][idx]})

# NODES
class SAM3VideoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": "", "multiline": False}),
                "use_gpu_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {"hf_token": ("STRING", {"default": ""})}
        }
    RETURN_TYPES = ("SAM3_VIDEO_MODEL",)
    RETURN_NAMES = ("video_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3/video"

    def load_model(self, checkpoint_path="", use_gpu_cache=True, hf_token=""):
        res = self._resolve(checkpoint_path)
        global _VIDEO_MODEL_CACHE
        if res in _VIDEO_MODEL_CACHE:
            p = _VIDEO_MODEL_CACHE[res]
            p.use_gpu_cache = use_gpu_cache
            if use_gpu_cache and hasattr(p, "model") and torch.cuda.is_available(): p.model.to("cuda")
            return (p,)
        bpe = Path(__file__).parent / "sam3_lib" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.exists(): bpe = Path(__file__).parent.parent / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
        p = build_sam3_video_predictor(checkpoint_path=res, bpe_path=str(bpe), hf_token=hf_token or None, gpus_to_use=None)
        p.use_gpu_cache = use_gpu_cache
        p.model.eval()
        _VIDEO_MODEL_CACHE[res] = p
        return (p,)

    @staticmethod
    def _resolve(user_path):
        from folder_paths import base_path
        if user_path and user_path.strip():
            p = Path(user_path.strip())
            if p.exists() and p.is_file(): return str(p.resolve())
            raise FileNotFoundError(f"Not found: {user_path}")
        mdir = Path(base_path) / "models" / "sam3"
        for n in ("sam3.safetensors", "sam3.pt"):
            c = mdir / n
            if c.exists() and c.is_file() and c.stat().st_size > 1000000: return str(c.resolve())
        raise FileNotFoundError("No sam3 checkpoint found")


class SAM3InitVideoSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_model": ("SAM3_VIDEO_MODEL",),
                "video_frames": ("IMAGE",),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),
                "score_threshold_detection": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "new_det_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    @classmethod
    def IS_CHANGED(cls, video_frames=None, session_id="", **kwargs):
        params = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if k != "video_model")
        return f"{session_id}|{_frames_signature(video_frames)}|{params}"
    RETURN_TYPES = ("SAM3_VIDEO_SESSION", "STRING")
    RETURN_NAMES = ("session", "session_id")
    FUNCTION = "init_session"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True

    def init_session(self, video_model, video_frames, session_id="", score_threshold_detection=0.3, new_det_thresh=0.4):
        m = video_model.model
        m.score_threshold_detection = score_threshold_detection
        m.new_det_thresh = new_det_thresh
        m.fill_hole_area = 16
        m.assoc_iou_thresh = 0.1
        m.det_nms_thresh = 0.1
        m.hotstart_unmatch_thresh = 8
        m.hotstart_dup_thresh = 8
        m.init_trk_keep_alive = 30
        m.hotstart_delay = 15
        m.decrease_trk_keep_alive_for_empty_masklets = False
        m.suppress_unmatched_only_within_hotstart = True

        sid = session_id or None
        if sid:
            try: video_model.get_session_state(sid)
            except:
                import uuid
                sid = f"auto_recovered_{uuid.uuid4().hex[:8]}"

        tdir = tempfile.mkdtemp(prefix="sam3_video_")
        num = int(video_frames.shape[0])
        for i in range(num):
            arr = (video_frames[i].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(tdir, f"{i:05d}.jpg"))

        try: resp = video_model.start_session(resource_path=tdir, session_id=sid)
        except:
            import uuid
            resp = video_model.start_session(resource_path=tdir, session_id=f"emergency_{uuid.uuid4().hex[:8]}")

        actual_sid = resp["session_id"]
        h, w = int(video_frames.shape[1]), int(video_frames.shape[2])
        sd = {
            "model": video_model, "session_id": actual_sid, "temp_dir": tdir,
            "num_frames": num, "height": h, "width": w,
            "frames": video_frames.detach().cpu(), "_prompt_history": [],
        }
        b64 = _encode_frames_for_ui(video_frames)
        _SESSION_FRAMES_CACHE[actual_sid] = {"b64": b64, "w": w, "h": h, "n": num}
        sd["_ui_frames_b64"] = b64
        sd["_ui_cache_key"] = f"sam3-video:{actual_sid}"

        return {"ui": {"session_id": [actual_sid], "num_frames": [num], "width": [w], "height": [h]}, "result": (sd, actual_sid)}


class SAM3InitVideoSessionAdvanced(SAM3InitVideoSession):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"].update({
            "fill_hole_area": ("INT", {"default": 16, "min": 0, "max": 1000}),
            "assoc_iou_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            "det_nms_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            "hotstart_unmatch_thresh": ("INT", {"default": 8, "min": 0, "max": 999}),
            "hotstart_dup_thresh": ("INT", {"default": 8, "min": 0, "max": 999}),
            "init_trk_keep_alive": ("INT", {"default": 30, "min": -10, "max": 50}),
            "hotstart_delay": ("INT", {"default": 15, "min": 0, "max": 200}),
            "decrease_keep_alive_empty": ("BOOLEAN", {"default": False}),
            "suppress_unmatched_globally": ("BOOLEAN", {"default": True}),
        })
        return base
    def init_session(self, video_model, video_frames, session_id="", score_threshold_detection=0.3, new_det_thresh=0.4, **kw):
        m = video_model.model
        m.score_threshold_detection = score_threshold_detection
        m.new_det_thresh = new_det_thresh
        m.fill_hole_area = kw.get("fill_hole_area", 16)
        m.assoc_iou_thresh = kw.get("assoc_iou_thresh", 0.1)
        m.det_nms_thresh = kw.get("det_nms_thresh", 0.1)
        m.hotstart_unmatch_thresh = kw.get("hotstart_unmatch_thresh", 8)
        m.hotstart_dup_thresh = kw.get("hotstart_dup_thresh", 8)
        m.init_trk_keep_alive = kw.get("init_trk_keep_alive", 30)
        m.hotstart_delay = kw.get("hotstart_delay", 15)
        m.decrease_trk_keep_alive_for_empty_masklets = kw.get("decrease_keep_alive_empty", False)
        m.suppress_unmatched_only_within_hotstart = not kw.get("suppress_unmatched_globally", True)
        return super().init_session(video_model, video_frames, session_id, score_threshold_detection, new_det_thresh)


class SAM3VideoPromptEditor:
    DESCRIPTION = """
### 🎬 SAM3 Video Prompt Editor (Multi-Prompt Edition)

One node = one frame = one **obj_id**. Chain multiple nodes to refine tracking across different frames.

---

#### 🧠 Combined Prompting (Simultaneous Modes)
Supports **Points + Boxes + Text** simultaneously on the same frame! 
All tools are active at the same time and triggered via specific shortcuts without any mode switches.

#### 🎨 Canvas Controls & Shortcuts
* 🟢 **Ctrl + LMB** — Place **Positive Point** (add character detail)
* 🔴 **Ctrl + RMB** — Place **Negative Point** (exclude background/artifacts)
* 🔵 **Shift + LMB Drag** — Draw **Positive Box** (main character search zone)
* 🟠 **Shift + RMB Drag** — Draw **Negative Box** (forced exclusion zone)
* 📐 **LMB Drag** — Move points or resize box handles
* ❌ **RMB / Delete / Backspace** — Delete element under cursor
* 🖐️ **MMB Drag** — Pan image
* 🔍 **Mouse Wheel** — Zoom in/out
* 📺 **F / Double-Click** — Fit image to view

---

#### 🖼️ Optional Image Input (Character Sheet)
Connect an external image (e.g., a **Character Sheet** with different angles) to the `image` input.
* The editor will display this reference image for prompt placement.
* Coordinates are automatically scaled to fit the underlying video session size.
"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"session": ("SAM3_VIDEO_SESSION",)},
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional override (e.g. Character Sheet). Displays these reference angles for easy prompt placement."}),
                "prompt_mode": (["points", "boxes"], {"default": "points", "tooltip": "Active drawing tool. Points and Boxes are preserved together."}),
                "text_prompt": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional text concept (e.g. 'character', 'red dress')."}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "obj_id": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "score_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "points_json": ("STRING", {"default": "[]", "multiline": True}),
                "boxes_json": ("STRING", {"default": "[]", "multiline": True}),
                "ui_cache_key": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SAM3_VIDEO_SESSION",)
    RETURN_NAMES = ("session",)
    FUNCTION = "apply"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, prompt_mode="points", frame_index=0, obj_id=1, score_threshold=0.3, points_json="[]", boxes_json="[]", text_prompt="", image=None, **kw):
        img_sig = _frames_signature(image) if image is not None else "none"
        return f"{prompt_mode}|{frame_index}|{obj_id}|{score_threshold}|{points_json}|{boxes_json}|{text_prompt}|{img_sig}"

    @staticmethod
    def _load(s, fb):
        try: return json.loads(s) if s else fb
        except: return fb

    def apply(self, session, image=None, prompt_mode="points", text_prompt="", frame_index=0, obj_id=1, score_threshold=0.30, points_json="[]", boxes_json="[]", ui_cache_key="", unique_id="0"):
        if session is None: raise ValueError("Session empty")
        vm = session["model"]
        _ensure_video_model_on_device(vm)

        frames = image.detach().cpu() if image is not None else session.get("frames")
        if frames is None: raise ValueError("No frames found.")

        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        frame_index = max(0, min(int(frame_index), n - 1))
        
        all_b64 = _encode_frames_for_ui(frames)
        cache_key = session.get("_ui_cache_key") or f"sam3-video:{session['session_id']}"
        session["_ui_cache_key"] = cache_key

        pts = self._load(points_json, [])
        boxes = self._load(boxes_json, [])

        # 1. NORMALIZED POINTS [0.0, 1.0]
        point_coords = []
        point_labels = []
        if pts:
            for p in pts:
                pw = float(p.get("img_w")) if p.get("img_w") else float(w)
                ph = float(p.get("img_h")) if p.get("img_h") else float(h)
                if pw <= 0 or ph <= 0:
                    continue
                
                px_n = float(p["x"]) / pw
                py_n = float(p["y"]) / ph
                
                px_n = max(0.0, min(px_n, 1.0))
                py_n = max(0.0, min(py_n, 1.0))
                
                point_coords.append([px_n, py_n])
                point_labels.append(int(p.get("label", 1)))

        # 2. NORMALIZED BOXES [0.0, 1.0]
        api_boxes = []
        api_box_labels = []
        if boxes:
            for b in boxes:
                pw = float(b.get("img_w")) if b.get("img_w") else float(w)
                ph = float(b.get("img_h")) if b.get("img_h") else float(h)
                if pw <= 0 or ph <= 0:
                    continue

                x0_n = min(float(b["x0"]), float(b["x1"])) / pw
                y0_n = min(float(b["y0"]), float(b["y1"])) / ph
                x1_n = max(float(b["x0"]), float(b["x1"])) / pw
                y1_n = max(float(b["y0"]), float(b["y1"])) / ph

                x0_n = max(0.0, min(x0_n, 1.0))
                y0_n = max(0.0, min(y0_n, 1.0))
                x1_n = max(0.0, min(x1_n, 1.0))
                y1_n = max(0.0, min(y1_n, 1.0))

                if (x1_n - x0_n) < 1e-4 or (y1_n - y0_n) < 1e-4:
                    continue

                api_boxes.append([x0_n, y0_n, x1_n, y1_n])
                api_box_labels.append(1 if b.get("positive", True) else 0)

        # 3. ОБРАБОТКА ОГРАНИЧЕНИЯ: Максимум 1 бокс на кадр
        final_boxes = []
        final_box_labels = []
        if api_boxes:
            # Ищем первый позитивный бокс как основной
            primary_idx = -1
            for idx, label in enumerate(api_box_labels):
                if label == 1:
                    primary_idx = idx
                    break
            if primary_idx == -1:
                primary_idx = 0  # Если позитивных нет, берем первый негативный
            
            # Оставляем только ОДИН бокс
            final_boxes.append(api_boxes[primary_idx])
            final_box_labels.append(api_box_labels[primary_idx])
            
            # Все остальные боксы конвертируем в точки, чтобы не потерять разметку пользователя
            for idx, (box, label) in enumerate(zip(api_boxes, api_box_labels)):
                if idx == primary_idx:
                    continue
                bx0, by0, bx1, by1 = box
                cx = (bx0 + bx1) / 2.0
                cy = (by0 + by1) / 2.0
                point_coords.append([cx, cy])
                point_labels.append(label)
                log.info(f"[SAM3] Multi-box fallback: Converted extra box {box} into point {[cx, cy]} (label {label}) because SAM3 only supports 1 box per frame.")

        # 4. SMART AUTO-ANCHORING (Якорение основного бокса)
        if final_boxes and point_coords:
            main_box = final_boxes[0]
            main_label = final_box_labels[0]
            if main_label == 1:  # Только для позитивных боксов
                bx0, by0, bx1, by1 = main_box
                bcx = (bx0 + bx1) / 2.0
                bcy = (by0 + by1) / 2.0
                
                already_anchored = any(abs(p[0] - bcx) < 0.05 and abs(p[1] - bcy) < 0.05 for p in point_coords)
                if not already_anchored:
                    point_coords.insert(0, [bcx, bcy])  # Приоритетный якорь
                    point_labels.insert(0, 1)
                    log.info(f"[SAM3] Smart Anchor: Injected positive point at box center {[bcx, bcy]} to protect mask from negative points suppression.")

        # Smart fallback for multimodal text prompt
        txt = text_prompt.strip()
        if not txt and (point_coords or final_boxes):
            txt = "object"
            log.info("[SAM3] Empty text prompt. Auto-fallback to 'object' to satisfy model attention.")

        if point_coords:
            log.info(f"[SAM3] Points sent (NORMALIZED): {point_coords}")
        if final_boxes:
            log.info(f"[SAM3] Boxes sent (NORMALIZED): {final_boxes}")

        has_prompt = bool(point_coords or final_boxes or txt)

        entry = {
            "node": unique_id, "frame_idx": frame_index, "obj_id": int(obj_id),
            "threshold": float(score_threshold),
            "points": point_coords if point_coords else None,
            "point_labels": point_labels if point_labels else None,
            "boxes": final_boxes if final_boxes else None,
            "box_labels": final_box_labels if final_box_labels else None,
            "text": txt if txt else None,
            "cache_key": [cache_key],
        }
        seq = session.setdefault("_prompts_seq", [])
        for i, e in enumerate(seq):
            if e.get("node") == unique_id:
                if has_prompt: seq[i] = entry
                else: seq.pop(i)
                break
        else:
            if has_prompt: seq.append(entry)

        was_dead = not _session_alive(vm, session.get("session_id"))
        if was_dead:
            _recover_session(session)
            vm = session["model"]
            _ensure_video_model_on_device(vm)

        done = session.setdefault("_committed", {})
        sig = f"{unique_id}|{frame_index}|{obj_id}|{score_threshold}|{points_json}|{boxes_json}|{txt}"
        committed = False

        if was_dead:
            if has_prompt:
                done[unique_id] = sig
                committed = True
        elif has_prompt and done.get(unique_id) != sig:
            m = getattr(vm, "model", None)
            old = getattr(m, "score_threshold_detection", None)
            try:
                if old is not None: m.score_threshold_detection = float(score_threshold)
                try:
                    # Двухэтапная отправка
                    if final_boxes or txt:
                        _video_add_prompt(
                            session=session, frame_idx=frame_index, obj_id=obj_id,
                            points=None, point_labels=None,
                            boxes=final_boxes, box_labels=final_box_labels, text=txt
                        )
                    if point_coords:
                        _video_add_prompt(
                            session=session, frame_idx=frame_index, obj_id=obj_id,
                            points=point_coords, point_labels=point_labels,
                            boxes=None, box_labels=None, text=None
                        )
                except RuntimeError as e:
                    if not _is_recoverable_session_error(e): raise
                    _recover_session(session, skip_node=unique_id)
                    
                    if final_boxes or txt:
                        _video_add_prompt(
                            session=session, frame_idx=frame_index, obj_id=obj_id,
                            points=None, point_labels=None,
                            boxes=final_boxes, box_labels=final_box_labels, text=txt
                        )
                    if point_coords:
                        _video_add_prompt(
                            session=session, frame_idx=frame_index, obj_id=obj_id,
                            points=point_coords, point_labels=point_labels,
                            boxes=None, box_labels=None, text=None
                        )
            finally:
                if old is not None: m.score_threshold_detection = old
            done[unique_id] = sig
            session.setdefault("_prompt_history", []).append({
                "node": unique_id, "frame_idx": frame_index, "obj_id": obj_id,
                "threshold": score_threshold, "has_text": bool(txt)
            })
            committed = True

        return {
            "ui": {
                "all_frames": all_b64, "num_frames": [n], "width": [w], "height": [h],
                "frame_index": [frame_index], "cache_key": [cache_key], "committed": [committed],
                "session_id": [session["session_id"]],
            },
            "result": (session,),
        }

class SAM3PropagateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"session": ("SAM3_VIDEO_SESSION",)},
            "optional": {
                "propagation_direction": (["both", "forward", "backward"], {"default": "both"}),
                "start_frame_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "max_frames": ("INT", {"default": -1, "min": -1, "max": 10000}),
            }
        }
    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SESSION")
    RETURN_NAMES = ("video_masks", "session")
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    def propagate(self, session, propagation_direction="both", start_frame_index=0, max_frames=-1):
        vm = session["model"]
        if _ensure_video_model_on_device(vm) or not _session_alive(vm, session.get("session_id")):
            _recover_session(session)
            vm = session["model"]

        req = {
            "type": "propagate_in_video",
            "session_id": session["session_id"],
            "propagation_direction": propagation_direction,
            "start_frame_index": int(start_frame_index),
            "max_frame_num_to_track": int(max_frames if max_frames > 0 else session["num_frames"]),
        }
        all_masks = {}
        all_obj_ids = None
        with torch.inference_mode():
            with _video_autocast(vm):
                for resp in vm.handle_stream_request(req):
                    idx = resp["frame_index"]
                    all_masks[idx] = resp["outputs"]
                    if all_obj_ids is None: all_obj_ids = resp["outputs"].get("obj_ids", [])
                    
        # Safely convert numpy array to list to prevent ambiguous truth value errors
        if all_obj_ids is None:
            all_obj_ids = []
        elif hasattr(all_obj_ids, "tolist"):
            all_obj_ids = all_obj_ids.tolist()
            
        return ({"session": session, "masks": all_masks, "obj_ids": all_obj_ids, "num_frames": session["num_frames"]}, session)

class SAM3VideoOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video_masks": ("SAM3_VIDEO_MASKS",)},
            "optional": {
                "obj_id_filter": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert mask (swap black and white)"}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "output_masks"
    CATEGORY = "SAM3/video"

    def output_masks(self, video_masks, obj_id_filter=-1, invert_mask=False):
        mdict = video_masks["masks"]
        num = video_masks["num_frames"]
        h, w = video_masks["session"]["height"], video_masks["session"]["width"]
        out = torch.zeros((num, h, w), dtype=torch.float32)

        for i in range(num):
            if i not in mdict: continue
            fo = mdict[i]

            frame_masks = None
            if "video_res_masks" in fo:
                frame_masks = fo["video_res_masks"]
            elif "pred_masks" in fo:
                frame_masks = fo["pred_masks"]
            elif "out_binary_masks" in fo:
                frame_masks = fo["out_binary_masks"]
            else:
                continue

            if frame_masks is None:
                continue

            if isinstance(frame_masks, np.ndarray):
                frame_masks = torch.from_numpy(frame_masks)

            frame_masks = frame_masks.cpu()

            # Ensure 3D tensor: (num_objects, height, width)
            if frame_masks.ndim == 4:
                if frame_masks.shape[1] == 1:
                    frame_masks = frame_masks.squeeze(1)
                elif frame_masks.shape[0] == 1:
                    frame_masks = frame_masks.squeeze(0)
            elif frame_masks.ndim == 2:
                frame_masks = frame_masks.unsqueeze(0)

            # Resize to session width & height if needed
            if frame_masks.shape[-2:] != (h, w):
                frame_masks = torch.nn.functional.interpolate(
                    frame_masks.float().unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

            ids = fo.get("obj_ids", [])
            if hasattr(ids, "tolist"):
                ids = ids.tolist()

            num_objs = frame_masks.shape[0]

            if obj_id_filter > 0:
                try:
                    idx = ids.index(obj_id_filter)
                    if idx < num_objs:
                        mask = frame_masks[idx] > 0.0
                    else:
                        mask = torch.zeros((h, w), dtype=torch.bool)
                except (ValueError, IndexError):
                    mask = torch.zeros((h, w), dtype=torch.bool)
            else:
                mask = (frame_masks > 0.0).any(dim=0)

            out[i] = mask.float()

        if invert_mask:
            out = 1.0 - out

        return (out,)

class SAM3CloseVideoSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("SAM3_VIDEO_SESSION",),
                "close_session": ("BOOLEAN", {"default": False}),
                "cleanup_temp_files": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "close_session"
    CATEGORY = "SAM3/video"

    def close_session(self, session, close_session=False, cleanup_temp_files=True):
        vm = session["model"]
        sid = session["session_id"]
        tdir = session.get("temp_dir")
        if not close_session: return (f"Session {sid} kept alive.",)
        try:
            vm.close_session(sid)
            status = f"Session {sid} closed"
        except Exception as e: status = f"Close warning: {e}"
        _SESSION_FRAMES_CACHE.pop(sid, None)
        if cleanup_temp_files and tdir and Path(tdir).exists():
            shutil.rmtree(tdir, ignore_errors=True)
            status += "; temp cleaned"
        if not getattr(vm, "use_gpu_cache", True) and hasattr(vm, "model"):
            vm.model.to("cpu")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()
        return (status,)

NODE_CLASS_MAPPINGS = {
    "SAM3VideoModelLoader": SAM3VideoModelLoader,
    "SAM3InitVideoSession": SAM3InitVideoSession,
    "SAM3InitVideoSessionAdvanced": SAM3InitVideoSessionAdvanced,
    "SAM3VideoPromptEditor": SAM3VideoPromptEditor,
    "SAM3PropagateVideo": SAM3PropagateVideo,
    "SAM3VideoOutput": SAM3VideoOutput,
    "SAM3CloseVideoSession": SAM3CloseVideoSession,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoModelLoader": "SAM3 Load Video Model",
    "SAM3InitVideoSession": "SAM3 Init Video Session",
    "SAM3InitVideoSessionAdvanced": "SAM3 Init Video Session (Advanced)",
    "SAM3VideoPromptEditor": "SAM3 Video Prompt Editor",
    "SAM3PropagateVideo": "SAM3 Propagate Video",
    "SAM3VideoOutput": "SAM3 Video Output",
    "SAM3CloseVideoSession": "SAM3 Close Video Session",
}
