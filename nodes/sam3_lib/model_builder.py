# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
import re
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# iopath is optional - fall back to regular file operations
try:
    from iopath.common.file_io import g_pathmgr
    IOPATH_AVAILABLE = True
except ImportError:
    IOPATH_AVAILABLE = False
    g_pathmgr = None

# Optional safetensors support
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def _remap_transformers_keys(state_dict: dict) -> dict:
    """
    Remap HuggingFace Transformers-format keys to our checkpoint format.

    This handles the structural differences between HF Transformers Sam3Model
    and our native checkpoint format, including tensor combinations where
    HF uses separate q/k/v projections but native uses combined in_proj.
    """
    remapped = {}

    # First pass: collect keys that need to be combined
    # Text encoder attention: q_proj + k_proj + v_proj -> in_proj
    qkv_groups = {}  # base_key -> {q: tensor, k: tensor, v: tensor}

    for k, v in state_dict.items():
        # Check for text encoder q/k/v projections that need combining
        # Keys look like: detector_model.text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight
        if "detector_model.text_encoder.text_model.encoder.layers." in k and ".self_attn." in k:
            if ".q_proj." in k or ".k_proj." in k or ".v_proj." in k:
                # Extract base key (without q/k/v_proj part)
                base = k.replace(".q_proj.", ".PROJ.").replace(".k_proj.", ".PROJ.").replace(".v_proj.", ".PROJ.")
                if base not in qkv_groups:
                    qkv_groups[base] = {}
                if ".q_proj." in k:
                    qkv_groups[base]["q"] = (k, v)
                elif ".k_proj." in k:
                    qkv_groups[base]["k"] = (k, v)
                elif ".v_proj." in k:
                    qkv_groups[base]["v"] = (k, v)

        # Also handle vision encoder q/k/v if they exist separately
        # Keys look like: detector_model.vision_encoder.backbone.encoder.layers.0.self_attn.q_proj.weight
        elif "detector_model.vision_encoder.backbone.encoder.layers." in k and ".self_attn." in k:
            if ".q_proj." in k or ".k_proj." in k or ".v_proj." in k:
                base = k.replace(".q_proj.", ".PROJ.").replace(".k_proj.", ".PROJ.").replace(".v_proj.", ".PROJ.")
                if base not in qkv_groups:
                    qkv_groups[base] = {}
                if ".q_proj." in k:
                    qkv_groups[base]["q"] = (k, v)
                elif ".k_proj." in k:
                    qkv_groups[base]["k"] = (k, v)
                elif ".v_proj." in k:
                    qkv_groups[base]["v"] = (k, v)

    # Combine q/k/v projections
    combined_keys = set()
    for base, qkv in qkv_groups.items():
        if "q" in qkv and "k" in qkv and "v" in qkv:
            q_key, q_val = qkv["q"]
            k_key, k_val = qkv["k"]
            v_key, v_val = qkv["v"]
            combined_keys.add(q_key)
            combined_keys.add(k_key)
            combined_keys.add(v_key)

            # Create combined in_proj key
            combined_val = torch.cat([q_val, k_val, v_val], dim=0)
            # Map to native format
            if "detector_model.text_encoder.text_model.encoder.layers." in q_key:
                match = re.search(r'\.layers\.(\d+)\.', q_key)
                if match:
                    layer_num = match.group(1)
                    # Native format uses in_proj_weight/in_proj_bias (no dot before weight/bias)
                    suffix = "_weight" if ".weight" in q_key else "_bias"
                    new_key = f"detector.backbone.language_backbone.encoder.transformer.resblocks.{layer_num}.attn.in_proj{suffix}"
                    remapped[new_key] = combined_val
            elif "detector_model.vision_encoder.backbone.encoder.layers." in q_key:
                match = re.search(r'\.layers\.(\d+)\.', q_key)
                if match:
                    layer_num = match.group(1)
                    suffix = ".weight" if ".weight" in q_key else ".bias"
                    new_key = f"detector.backbone.vision_backbone.trunk.blocks.{layer_num}.attn.qkv{suffix}"
                    remapped[new_key] = combined_val

    # Second pass: process remaining keys
    for k, v in state_dict.items():
        # Skip keys that were already combined
        if k in combined_keys:
            continue

        new_key = k

        # === DETECTOR MAPPINGS ===

        # Vision encoder backbone -> backbone.vision_backbone.trunk
        if k.startswith("detector_model.vision_encoder.backbone."):
            new_key = k.replace(
                "detector_model.vision_encoder.backbone.",
                "detector.backbone.vision_backbone.trunk."
            )
            # HF uses "embeddings.patch_embeddings.projection" -> we use "patch_embed.proj"
            new_key = new_key.replace(".embeddings.patch_embeddings.projection.", ".patch_embed.proj.")
            # HF uses "embeddings.position_embedding" -> we use "pos_embed"
            new_key = new_key.replace(".embeddings.position_embedding", ".pos_embed")
            # HF uses "encoder.layers" -> we use "blocks"
            new_key = new_key.replace(".encoder.layers.", ".blocks.")
            # HF uses "layer_norm1/2" -> we use "norm1/2"
            new_key = new_key.replace(".layer_norm1.", ".norm1.")
            new_key = new_key.replace(".layer_norm2.", ".norm2.")
            # HF uses "self_attn" -> we use "attn"
            new_key = new_key.replace(".self_attn.", ".attn.")
            # HF uses "in_proj" -> we use "qkv"
            new_key = new_key.replace(".in_proj_weight", ".qkv.weight")
            new_key = new_key.replace(".in_proj_bias", ".qkv.bias")
            # HF uses "out_proj" -> we use "proj"
            new_key = new_key.replace(".out_proj.", ".proj.")
            # HF uses "mlp.fc1/fc2" -> we use "mlp.fc1/fc2" (same)
            # HF uses "pre_layernorm" -> we use "ln_pre"
            new_key = new_key.replace(".pre_layernorm.", ".ln_pre.")

        # Vision encoder neck -> backbone.vision_backbone.convs/sam2_convs
        elif k.startswith("detector_model.vision_encoder.neck."):
            new_key = k.replace(
                "detector_model.vision_encoder.neck.",
                "detector.backbone.vision_backbone."
            )
            # HF uses "layers" -> we use "convs" or "sam2_convs"
            new_key = new_key.replace(".layers.", ".convs.")
            # HF uses "deconv" -> we use "dconv_2x2"
            new_key = new_key.replace(".deconv.", ".dconv_2x2.")

        # Text encoder -> backbone.language_backbone.encoder
        elif k.startswith("detector_model.text_encoder.text_model."):
            new_key = k.replace(
                "detector_model.text_encoder.text_model.",
                "detector.backbone.language_backbone.encoder."
            )
            # HF uses "embeddings.position_embedding.weight" -> "positional_embedding"
            new_key = new_key.replace(".embeddings.position_embedding.weight", ".positional_embedding")
            # HF uses "embeddings.token_embedding" -> "token_embedding"
            new_key = new_key.replace(".embeddings.token_embedding.", ".token_embedding.")
            # HF uses "encoder.layers" -> "transformer.resblocks"
            new_key = new_key.replace(".encoder.layers.", ".transformer.resblocks.")
            # HF uses "final_layer_norm" -> "ln_final"
            new_key = new_key.replace(".final_layer_norm.", ".ln_final.")
            # HF uses "self_attn" -> "attn"
            new_key = new_key.replace(".self_attn.", ".attn.")
            # HF uses "layer_norm1/2" -> "ln_1/2"
            new_key = new_key.replace(".layer_norm1.", ".ln_1.")
            new_key = new_key.replace(".layer_norm2.", ".ln_2.")
            # HF uses "mlp.fc1/fc2" -> "mlp.c_fc/c_proj"
            new_key = new_key.replace(".mlp.fc1.", ".mlp.c_fc.")
            new_key = new_key.replace(".mlp.fc2.", ".mlp.c_proj.")

        # Text encoder projection
        elif k.startswith("detector_model.text_encoder.text_projection"):
            new_key = k.replace(
                "detector_model.text_encoder.text_projection",
                "detector.backbone.language_backbone.encoder.text_projection"
            )

        # Text projection (separate module)
        elif k.startswith("detector_model.text_projection."):
            new_key = k.replace(
                "detector_model.text_projection.",
                "detector.backbone.language_backbone.text_proj."
            )

        # DETR encoder -> transformer.encoder
        elif k.startswith("detector_model.detr_encoder."):
            new_key = k.replace(
                "detector_model.detr_encoder.",
                "detector.transformer.encoder."
            )
            # HF layer naming adjustments
            new_key = new_key.replace(".self_attn_layer_norm.", ".norm1.")
            new_key = new_key.replace(".cross_attn_layer_norm.", ".cross_attn_norm.")
            new_key = new_key.replace(".mlp_layer_norm.", ".norm2.")
            new_key = new_key.replace(".mlp.fc1.", ".linear1.")
            new_key = new_key.replace(".mlp.fc2.", ".linear2.")

        # DETR decoder -> transformer.decoder
        elif k.startswith("detector_model.detr_decoder."):
            new_key = k.replace(
                "detector_model.detr_decoder.",
                "detector.transformer.decoder."
            )
            # HF uses "ref_point_head.layer1/2/3" -> "ref_point_head.layers.0/1/2"
            new_key = new_key.replace(".ref_point_head.layer1.", ".ref_point_head.layers.0.")
            new_key = new_key.replace(".ref_point_head.layer2.", ".ref_point_head.layers.1.")
            new_key = new_key.replace(".ref_point_head.layer3.", ".ref_point_head.layers.2.")
            # HF uses "box_head.layer1/2/3" -> "box_head.layers.0/1/2"
            new_key = new_key.replace(".box_head.layer1.", ".box_head.layers.0.")
            new_key = new_key.replace(".box_head.layer2.", ".box_head.layers.1.")
            new_key = new_key.replace(".box_head.layer3.", ".box_head.layers.2.")
            # Layer norm and attention naming
            new_key = new_key.replace(".self_attn_layer_norm.", ".norm1.")
            new_key = new_key.replace(".vision_cross_attn_layer_norm.", ".ca_norm.")
            new_key = new_key.replace(".text_cross_attn_layer_norm.", ".text_ca_norm.")
            new_key = new_key.replace(".mlp_layer_norm.", ".ffn_norm.")
            new_key = new_key.replace(".mlp.fc1.", ".linear1.")
            new_key = new_key.replace(".mlp.fc2.", ".linear2.")

        # Geometry encoder (mostly same structure)
        elif k.startswith("detector_model.geometry_encoder."):
            new_key = k.replace(
                "detector_model.geometry_encoder.",
                "detector.geometry_encoder."
            )
            # HF uses "output_layer_norm" -> "norm"
            new_key = new_key.replace(".output_layer_norm.", ".norm.")
            # HF uses "prompt_layer_norm" -> "encode_norm"
            new_key = new_key.replace(".prompt_layer_norm.", ".encode_norm.")
            # HF uses "vision_layer_norm" -> "img_pre_norm"
            new_key = new_key.replace(".vision_layer_norm.", ".img_pre_norm.")

        # Dot product scoring
        elif k.startswith("detector_model.dot_product_scoring."):
            new_key = k.replace(
                "detector_model.dot_product_scoring.",
                "detector.dot_prod_scoring."
            )
            # HF uses "query_proj" -> "hs_proj"
            new_key = new_key.replace(".query_proj.", ".hs_proj.")
            # HF uses "text_mlp" -> "prompt_mlp"
            new_key = new_key.replace(".text_mlp.", ".prompt_mlp.")
            # HF uses "text_proj" -> "prompt_proj"
            new_key = new_key.replace(".text_proj.", ".prompt_proj.")
            new_key = new_key.replace(".text_mlp_out_norm.", ".prompt_mlp.out_norm.")

        # Mask decoder -> segmentation_head
        elif k.startswith("detector_model.mask_decoder."):
            new_key = k.replace(
                "detector_model.mask_decoder.",
                "detector.segmentation_head."
            )
            # HF uses "instance_projection" -> "instance_seg_head"
            new_key = new_key.replace(".instance_projection.", ".instance_seg_head.")
            # HF uses "semantic_projection" -> "semantic_seg_head"
            new_key = new_key.replace(".semantic_projection.", ".semantic_seg_head.")
            # HF uses "mask_embedder" -> "mask_predictor"
            new_key = new_key.replace(".mask_embedder.", ".mask_predictor.")
            # HF uses "prompt_cross_attn" -> "cross_attend_prompt"
            new_key = new_key.replace(".prompt_cross_attn.", ".cross_attend_prompt.")
            new_key = new_key.replace(".prompt_cross_attn_norm.", ".cross_attn_norm.")

        # === TRACKER MAPPINGS ===

        # Tracker mask decoder -> sam_mask_decoder
        elif k.startswith("tracker_model.mask_decoder."):
            new_key = k.replace(
                "tracker_model.mask_decoder.",
                "tracker.sam_mask_decoder."
            )
            # HF uses "upscale_conv1/2" -> "output_upscaling.0/3"
            new_key = new_key.replace(".upscale_conv1.", ".output_upscaling.0.")
            new_key = new_key.replace(".upscale_conv2.", ".output_upscaling.3.")
            new_key = new_key.replace(".upscale_layer_norm.", ".output_upscaling.1.")

        # Tracker mask downsample (same)
        elif k.startswith("tracker_model.mask_downsample."):
            new_key = k.replace("tracker_model.mask_downsample.", "tracker.mask_downsample.")

        # Memory attention -> transformer.encoder
        elif k.startswith("tracker_model.memory_attention."):
            new_key = k.replace(
                "tracker_model.memory_attention.",
                "tracker.transformer.encoder."
            )

        # Memory encoder -> maskmem_backbone
        elif k.startswith("tracker_model.memory_encoder."):
            new_key = k.replace(
                "tracker_model.memory_encoder.",
                "tracker.maskmem_backbone."
            )
            # HF uses "feature_projection" -> "pix_feat_proj"
            new_key = new_key.replace(".feature_projection.", ".pix_feat_proj.")
            # HF uses "projection" -> "out_proj"
            if ".projection." in new_key and ".feature_projection." not in k:
                new_key = new_key.replace(".projection.", ".out_proj.")
            # HF uses "memory_fuser" -> "fuser"
            new_key = new_key.replace(".memory_fuser.", ".fuser.")
            # HF mask_downsampler layers
            new_key = new_key.replace(".mask_downsampler.layers.0.conv.", ".mask_downsampler.encoder.0.")
            new_key = new_key.replace(".mask_downsampler.layers.1.conv.", ".mask_downsampler.encoder.3.")
            new_key = new_key.replace(".mask_downsampler.layers.2.conv.", ".mask_downsampler.encoder.6.")
            new_key = new_key.replace(".mask_downsampler.layers.3.conv.", ".mask_downsampler.encoder.9.")
            new_key = new_key.replace(".mask_downsampler.layers.0.layer_norm.", ".mask_downsampler.encoder.1.")
            new_key = new_key.replace(".mask_downsampler.layers.1.layer_norm.", ".mask_downsampler.encoder.4.")
            new_key = new_key.replace(".mask_downsampler.layers.2.layer_norm.", ".mask_downsampler.encoder.7.")
            new_key = new_key.replace(".mask_downsampler.layers.3.layer_norm.", ".mask_downsampler.encoder.10.")

        # Prompt encoder -> sam_prompt_encoder
        elif k.startswith("tracker_model.prompt_encoder."):
            new_key = k.replace(
                "tracker_model.prompt_encoder.",
                "tracker.sam_prompt_encoder."
            )
            # HF uses "mask_embed.conv1/2/3" -> "mask_downscaling.0/3/6"
            new_key = new_key.replace(".mask_embed.conv1.", ".mask_downscaling.0.")
            new_key = new_key.replace(".mask_embed.conv2.", ".mask_downscaling.3.")
            new_key = new_key.replace(".mask_embed.conv3.", ".mask_downscaling.6.")
            new_key = new_key.replace(".mask_embed.layer_norm1.", ".mask_downscaling.1.")
            new_key = new_key.replace(".mask_embed.layer_norm2.", ".mask_downscaling.4.")
            # HF uses "point_embed" -> "point_embeddings"
            new_key = new_key.replace(".point_embed.", ".point_embeddings.")
            # HF uses "shared_embedding" -> "shared_image_embedding" (handled separately)

        # Object pointer projection
        elif k.startswith("tracker_model.object_pointer_proj."):
            new_key = k.replace(
                "tracker_model.object_pointer_proj.",
                "tracker.obj_ptr_proj."
            )

        # Temporal positional encoding projection
        elif k.startswith("tracker_model.temporal_positional_encoding_projection_layer."):
            new_key = k.replace(
                "tracker_model.temporal_positional_encoding_projection_layer.",
                "tracker.obj_ptr_tpos_proj."
            )

        # Special single-key mappings
        elif k == "tracker_model.memory_temporal_positional_encoding":
            new_key = "tracker.maskmem_tpos_enc"
        elif k == "tracker_model.no_memory_embedding":
            new_key = "tracker.no_mem_embed"
        elif k == "tracker_model.no_memory_positional_encoding":
            new_key = "tracker.no_mem_pos_enc"
        elif k == "tracker_model.no_object_pointer":
            new_key = "tracker.no_obj_ptr"
        elif k == "tracker_model.occlusion_spatial_embedding_parameter":
            new_key = "tracker.no_obj_embed_spatial"
        elif k.startswith("tracker_model.shared_image_embedding."):
            new_key = k.replace(
                "tracker_model.shared_image_embedding.positional_embedding",
                "tracker.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
            )

        # Tracker neck -> backbone.vision_backbone.sam2_convs (if present)
        elif k.startswith("tracker_neck."):
            # This maps to the SAM2-style neck in the vision backbone
            new_key = k.replace("tracker_neck.", "detector.backbone.vision_backbone.sam2_neck.")

        # Default: keep as is (shouldn't happen for valid keys)
        else:
            new_key = k

        remapped[new_key] = v

    return remapped


def _load_checkpoint_file(checkpoint_path: str) -> dict:
    """
    Load checkpoint from file, supporting both .pt and .safetensors formats.
    Automatically detects and remaps HuggingFace Transformers-format keys.

    Args:
        checkpoint_path: Path to checkpoint file (.pt, .pth, or .safetensors)

    Returns:
        Dictionary containing model state dict
    """
    is_safetensors = checkpoint_path.endswith('.safetensors')

    if is_safetensors:
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install it with: pip install safetensors"
            )
        print(f"[SAM3] Loading safetensors checkpoint: {checkpoint_path}")
        state_dict = load_safetensors(checkpoint_path)

        # Auto-detect and remap HuggingFace Transformers format if needed
        sample_keys = list(state_dict.keys())[:10]
        is_transformers_format = any(
            k.startswith('detector_model.') or k.startswith('tracker_model.')
            for k in sample_keys
        )

        if is_transformers_format:
            print("[SAM3] Detected HuggingFace Transformers format, remapping keys...")
            state_dict = _remap_transformers_keys(state_dict)

        return state_dict
    else:
        print(f"[SAM3] Loading PyTorch checkpoint: {checkpoint_path}")
        if IOPATH_AVAILABLE:
            with g_pathmgr.open(checkpoint_path, "rb") as f:
                return torch.load(f, map_location="cpu", weights_only=True)
        else:
            with open(checkpoint_path, "rb") as f:
                return torch.load(f, map_location="cpu", weights_only=True)
from .model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from .model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .model.geometry_encoders import SequenceGeometryEncoder
from .model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from .model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .model.necks import Sam3DualViTDetNeck
from .model.position_encoding import PositionEmbeddingSine
from .model.sam1_task_predictor import SAM3InteractiveImagePredictor
from .model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from .model.sam3_tracking_predictor import Sam3TrackerPredictor
from .model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from .sam3_video_predictor import Sam3VideoPredictor, Sam3VideoPredictorMultiGPU
from .model.text_encoder_ve import VETextEncoder
from .model.tokenizer_ve import SimpleTokenizer
from .model.vitdet import ViT
from .model.vl_combiner import SAM3VLBackbone
from .sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    # Matcher is only needed for training, always None for inference-only mode
    matcher = None
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file (supports .pt and .safetensors)."""
    ckpt = _load_checkpoint_file(checkpoint_path)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    # Remap detector.* -> * for image model
    sam3_image_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("detector."):
            sam3_image_ckpt[k.replace("detector.", "")] = v

    # Remap tracker.* -> inst_interactive_predictor.model.* if needed
    if model.inst_interactive_predictor is not None:
        for k, v in ckpt.items():
            if k.startswith("tracker."):
                sam3_image_ckpt[k.replace("tracker.", "inst_interactive_predictor.model.")] = v
    # Debug: show what we're loading
    inst_keys = [k for k in sam3_image_ckpt.keys() if 'inst_interactive_predictor' in k]
    print(f"[SAM3] Loading checkpoint with {len(sam3_image_ckpt)} keys ({len(inst_keys)} for inst_interactive_predictor)")

    missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=False)

    # Check for missing inst_interactive_predictor keys
    critical_missing = [k for k in missing_keys if 'inst_interactive_predictor' in k]
    if critical_missing:
        print(f"[SAM3] WARNING: Missing inst_interactive_predictor keys: {len(critical_missing)}")
        for k in critical_missing[:10]:
            print(f"[SAM3]   MISSING: {k}")

    # Check for unexpected keys
    if unexpected_keys:
        inst_unexpected = [k for k in unexpected_keys if 'inst_interactive_predictor' in k]
        if inst_unexpected:
            print(f"[SAM3] WARNING: Unexpected inst_interactive_predictor keys: {len(inst_unexpected)}")
            for k in inst_unexpected[:5]:
                print(f"[SAM3]   UNEXPECTED: {k}")

    if len(missing_keys) > 0:
        print(f"[SAM3] Total missing keys: {len(missing_keys)}")


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    elif device != "cpu":
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        load_from_HF: Whether to download from HuggingFace if checkpoint not found
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile: Whether to enable torch compilation for speed

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        # Path to bundled BPE tokenizer vocabulary in sam3_lib/
        bpe_path = os.path.join(
            os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
        )
    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        # Build the tracker base model for SAM2-style point/box segmentation
        sam3_tracker_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_tracker_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def download_ckpt_from_hf():
    """
    Download SAM3 checkpoint from HuggingFace (public repo, no token needed).

    Returns:
        Path to downloaded checkpoint
    """
    SAM3_MODEL_ID = "1038lab/sam3"
    SAM3_CKPT_NAME = "sam3.pt"

    checkpoint_path = hf_hub_download(
        repo_id=SAM3_MODEL_ID,
        filename=SAM3_CKPT_NAME,
    )
    return checkpoint_path


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
    enable_inst_interactivity: bool = False,
):
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        # Path to bundled BPE tokenizer vocabulary in sam3_lib/
        bpe_path = os.path.join(
            os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone(enable_inst_interactivity=enable_inst_interactivity)
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build instance interactive predictor if enabled (for SAM2-style point/box segmentation)
    if enable_inst_interactivity:
        sam3_tracker_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_tracker_base)
    else:
        inst_predictor = None

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
        inst_interactive_predictor=inst_predictor,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.3,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.4,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.3,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.4,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # Load checkpoint if provided (supports .pt and .safetensors)
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        ckpt = _load_checkpoint_file(checkpoint_path)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        # Keys should already be in detector.*/tracker.* format
        # (incompatible formats are rejected by _load_checkpoint_file)
        remapped_ckpt = dict(ckpt)

        # If inst_interactive_predictor is enabled, remap tracker weights for it
        if enable_inst_interactivity and inst_predictor is not None:
            inst_predictor_keys = {
                k.replace("tracker.", "detector.inst_interactive_predictor.model."): v
                for k, v in remapped_ckpt.items()
                if k.startswith("tracker.")
            }
            remapped_ckpt.update(inst_predictor_keys)
            print(f"[SAM3] Added {len(inst_predictor_keys)} keys for detector.inst_interactive_predictor")

        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")

    # Keep model in float32; autocast will convert activations to bfloat16 dynamically
    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    # Use single-device predictor on CPU, multi-GPU predictor only when CUDA is available
    if not torch.cuda.is_available():
        return Sam3VideoPredictor(*model_args, **model_kwargs)
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
