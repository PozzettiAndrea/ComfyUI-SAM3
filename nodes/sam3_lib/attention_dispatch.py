"""Centralized attention dispatch with GPU auto-detection.

User-facing backends:
- auto: Auto-detect fastest backend for current GPU + installed packages (recommended)
- sdpa: PyTorch's F.scaled_dot_product_attention (always available, any GPU)
- flash_attn: FlashAttention (FA4 on Blackwell, FA3 on Hopper, FA2 on Ampere+)
- sage: SageAttention (v3 FP4 on Blackwell, v2 INT8 on Ampere+)

Internal resolved backends:
- sdpa, flash_attn (FA2), flash_attn_fp8 (FA3/FA4), sage2, sage3

Speed tiers by GPU generation:
- Blackwell (SM 10.x): sage3 > FA4 > sage2 > FA2 > sdpa
- Hopper   (SM 9.0):   FA3  > sage2 > FA2 > sdpa
- Ada      (SM 8.9):   sage2 > FA2 > sdpa
- Ampere   (SM 8.x):   sage2 ~ FA2 > sdpa
- Older:                sdpa only
"""

import logging
import torch
import torch.nn.functional as F
import comfy.model_management

logger = logging.getLogger("SAM3")

_active_backend = "sdpa"
_gpu_arch = None  # cached (major, minor) tuple
_backend_unsupported_dims = {}  # {backend_name: set(head_dims)} — skip try/except for known failures


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------

def _get_gpu_arch():
    """Return (major, minor) compute capability, cached after first call."""
    global _gpu_arch
    if _gpu_arch is None:
        if comfy.model_management.get_torch_device().type == "cuda":
            _gpu_arch = torch.cuda.get_device_capability()
        else:
            _gpu_arch = (0, 0)
    return _gpu_arch


def _can_import(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def auto_detect_precision():
    """Return the best inference dtype for the current GPU.

    Returns:
        torch.bfloat16 on Ampere+ (SM 8.0+)
        torch.float16  on Volta/Turing (SM 7.x)
        torch.float32  on older / CPU
    """
    major, _ = _get_gpu_arch()
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_BACKEND_LABELS = {
    "sdpa": "sdpa (PyTorch native)",
    "flash_attn": "FlashAttention 2 (fp16/bf16)",
    "flash_attn_fp8": "FlashAttention 3/4 (FP8)",
    "sage2": "SageAttention v2 (INT8)",
    "sage3": "SageAttention v3 (Blackwell FP4)",
}


def _auto_select():
    """Pick the fastest available backend for the current GPU."""
    major, _ = _get_gpu_arch()

    if major >= 10:  # Blackwell (SM 10.0+)
        for backend, module in [
            ("sage3", "sageattn3"),
            ("flash_attn_fp8", "flash_attn_interface"),
            ("sage2", "sageattention"),
            ("flash_attn", "flash_attn"),
        ]:
            if _can_import(module):
                return backend

    elif major == 9:  # Hopper (SM 9.0)
        for backend, module in [
            ("flash_attn_fp8", "flash_attn_interface"),
            ("sage2", "sageattention"),
            ("flash_attn", "flash_attn"),
        ]:
            if _can_import(module):
                return backend

    elif major == 8:  # Ampere / Ada (SM 8.0-8.9)
        for backend, module in [
            ("sage2", "sageattention"),
            ("flash_attn", "flash_attn"),
        ]:
            if _can_import(module):
                return backend

    return "sdpa"


def set_backend(name: str) -> str:
    """Set the active attention backend.

    Args:
        name: One of "auto", "sdpa", "flash_attn", "sage".

    Returns:
        The resolved internal backend name.
    """
    global _active_backend

    if name == "auto":
        resolved = _auto_select()
        _active_backend = resolved
        logger.info(f"Attention backend (auto): {_BACKEND_LABELS.get(resolved, resolved)}")
        return resolved

    if name == "sdpa":
        _active_backend = "sdpa"
        logger.info("Attention backend: sdpa (PyTorch native)")
        return "sdpa"

    if name == "flash_attn":
        major, _ = _get_gpu_arch()
        # Try FA3/FA4 on Hopper+
        if major >= 9 and _can_import("flash_attn_interface"):
            _active_backend = "flash_attn_fp8"
            label = "FlashAttention 4 (FP8)" if major >= 10 else "FlashAttention 3 (FP8)"
            logger.info(f"Attention backend: {label}")
            return "flash_attn_fp8"
        # Fall back to FA2
        if _can_import("flash_attn"):
            _active_backend = "flash_attn"
            logger.info("Attention backend: FlashAttention 2 (fp16/bf16)")
            return "flash_attn"
        logger.warning("flash-attn package not installed, falling back to sdpa")
        _active_backend = "sdpa"
        return "sdpa"

    if name == "sage":
        major, _ = _get_gpu_arch()
        # Try sage3 on Blackwell
        if major >= 10 and _can_import("sageattn3"):
            _active_backend = "sage3"
            logger.info("Attention backend: SageAttention v3 (Blackwell FP4)")
            return "sage3"
        # Fall back to sage2
        if _can_import("sageattention"):
            _active_backend = "sage2"
            logger.info("Attention backend: SageAttention v2 (INT8)")
            return "sage2"
        logger.warning("sageattention package not installed, falling back to sdpa")
        _active_backend = "sdpa"
        return "sdpa"

    logger.warning(f"Unknown attention backend '{name}', falling back to sdpa")
    _active_backend = "sdpa"
    return "sdpa"


def get_backend() -> str:
    """Return the currently active internal backend name."""
    return _active_backend


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """Route attention to the active backend.

    Args:
        q, k, v: Tensors of shape (B, H, N, D)
        attn_mask: Optional mask — forces SDPA when present.
        dropout_p: Dropout probability.

    Returns:
        Output tensor (B, H, N, D)
    """
    if attn_mask is not None or _active_backend == "sdpa":
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

    if _active_backend == "flash_attn":
        return _dispatch_flash_attn_fa2(q, k, v, dropout_p)

    if _active_backend == "flash_attn_fp8":
        return _dispatch_flash_attn_fp8(q, k, v)

    if _active_backend == "sage3":
        return _dispatch_sage3(q, k, v)

    if _active_backend == "sage2":
        return _dispatch_sage2(q, k, v)

    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _dispatch_flash_attn_fa2(q, k, v, dropout_p):
    """FlashAttention 2 -- fp16/bf16, Ampere+. Layout: (B,H,N,D) -> (B,N,H,D)."""
    if q.dtype == torch.float32:
        logger.debug("FA2 requires fp16/bf16, falling back to sdpa for this call")
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    try:
        from flash_attn import flash_attn_func
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q, k, v, dropout_p=dropout_p)
        return out.transpose(1, 2)
    except Exception as e:
        logger.warning(f"FA2 failed ({e}), falling back to sdpa")
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


@torch.compiler.disable
def _dispatch_flash_attn_fp8(q, k, v):
    """FlashAttention 3/4 -- FP8, Hopper/Blackwell. Layout: (B,H,N,D) -> (B,N,H,D)."""
    try:
        from flash_attn_interface import flash_attn_func as fa_func
        orig_dtype = q.dtype
        fp8 = torch.float8_e4m3fn
        q = q.transpose(1, 2).contiguous().to(fp8)
        k = k.transpose(1, 2).contiguous().to(fp8)
        v = v.transpose(1, 2).contiguous().to(fp8)
        out = fa_func(q, k, v)
        return out.to(orig_dtype).transpose(1, 2)
    except Exception as e:
        logger.warning(f"FA3/FA4 FP8 failed ({e}), falling back to sdpa")
        return F.scaled_dot_product_attention(q, k, v)


@torch.compiler.disable
def _dispatch_sage3(q, k, v):
    """SageAttention v3 -- Blackwell FP4. Hidden from torch.compile."""
    head_dim = q.shape[-1]
    if head_dim in _backend_unsupported_dims.get("sage3", set()):
        return F.scaled_dot_product_attention(q, k, v)
    try:
        from sageattn3 import sageattn3
        return sageattn3(q, k, v)
    except Exception as e:
        _backend_unsupported_dims.setdefault("sage3", set()).add(head_dim)
        logger.warning(f"sage3 doesn't support head_dim={head_dim}, using sdpa for these layers")
        return F.scaled_dot_product_attention(q, k, v)


@torch.compiler.disable
def _dispatch_sage2(q, k, v):
    """SageAttention v2 -- INT8. Hidden from torch.compile."""
    head_dim = q.shape[-1]
    if head_dim in _backend_unsupported_dims.get("sage2", set()):
        return F.scaled_dot_product_attention(q, k, v)
    try:
        from sageattention import sageattn
        return sageattn(q, k, v)
    except Exception as e:
        _backend_unsupported_dims.setdefault("sage2", set()).add(head_dim)
        logger.warning(f"sage2 doesn't support head_dim={head_dim}, using sdpa for these layers")
        return F.scaled_dot_product_attention(q, k, v)
