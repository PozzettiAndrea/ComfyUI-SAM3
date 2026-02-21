# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# ComfyUI-native model builder.
# Consolidated from: model_builder.py

"""
SAM3 Library - Vendored for ComfyUI-SAM3

This is a vendored version of Meta's SAM3 (Segment Anything Model 3) library,
containing only the essential components needed for image segmentation inference.
"""

import logging
import os
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

import comfy.ops
import comfy.utils
import comfy.model_management

ops = comfy.ops.manual_cast

log = logging.getLogger("sam3")

# ---------------------------------------------------------------------------
# Imports from new flat files
# ---------------------------------------------------------------------------

from .attention import (
    SplitMultiheadAttention,
    RoPEAttention,
)
from .text_encoder import VETextEncoder
from .model import (
    # Core model classes
    ViT,
    Sam3DualViTDetNeck,
    SAM3VLBackbone,
    PositionEmbeddingSine,
    TransformerWrapper,
    TransformerEncoderFusion,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
    SequenceGeometryEncoder,
    PixelDecoder,
    UniversalSegmentationHead,
    DotProductScoring,
    MLP,
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
    Sam3Image,
    Sam3ImageOnVideoMultiGPU,
    Sam3TrackerPredictor,
    Sam3VideoInferenceWithInstanceInteractivity,
    SAM3InteractiveImagePredictor,
)
from .predictor import Sam3VideoPredictor, Sam3VideoPredictorMultiGPU
from .tokenizer import SimpleTokenizer


# ---------------------------------------------------------------------------
# Weight key conversion for nn.MultiheadAttention -> SplitMultiheadAttention
# ---------------------------------------------------------------------------

def convert_mha_state_dict(state_dict):
    """
    Convert nn.MultiheadAttention combined in_proj_weight/bias to split q/k/v.

    nn.MultiheadAttention stores Q, K, V projections as a single combined tensor:
        in_proj_weight: (3*embed_dim, embed_dim)
        in_proj_bias: (3*embed_dim,)

    SplitMultiheadAttention uses separate projections:
        to_q.weight, to_k.weight, to_v.weight: (embed_dim, embed_dim) each
        to_q.bias, to_k.bias, to_v.bias: (embed_dim,) each

    Note: out_proj keys are NOT renamed because SplitMultiheadAttention uses
    self.out_proj (same name as nn.MultiheadAttention).
    """
    new_sd = {}
    for key, value in state_dict.items():
        if 'in_proj_weight' in key:
            prefix = key.replace('in_proj_weight', '')
            q, k, v = value.chunk(3, dim=0)
            new_sd[prefix + 'to_q.weight'] = q
            new_sd[prefix + 'to_k.weight'] = k
            new_sd[prefix + 'to_v.weight'] = v
        elif 'in_proj_bias' in key:
            prefix = key.replace('in_proj_bias', '')
            q, k, v = value.chunk(3, dim=0)
            new_sd[prefix + 'to_q.bias'] = q
            new_sd[prefix + 'to_k.bias'] = k
            new_sd[prefix + 'to_v.bias'] = v
        else:
            new_sd[key] = value
    return new_sd


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint_file(checkpoint_path: str) -> dict:
    """
    Load checkpoint from file, supporting both .pt and .safetensors formats.

    Args:
        checkpoint_path: Path to checkpoint file (.pt, .pth, or .safetensors)

    Returns:
        Dictionary containing model state dict

    Note:
        Safetensors files must use the same key format as the native .pt checkpoint.
        The HuggingFace Transformers format (detector_model.*/tracker_model.* keys)
        is NOT supported.
    """
    log.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = comfy.utils.load_torch_file(str(checkpoint_path))

    # Check if this is an unsupported HuggingFace Transformers format
    sample_keys = list(state_dict.keys())[:10]
    if any(k.startswith('detector_model.') or k.startswith('tracker_model.') for k in sample_keys):
        raise ValueError(
            "This checkpoint uses the HuggingFace Transformers key format "
            "(detector_model.*/tracker_model.*), which is not compatible with this loader. "
            "Please use the native sam3.pt checkpoint and convert it to safetensors if needed."
        )

    return state_dict


def download_ckpt_from_hf():
    """
    Download SAM3 checkpoint from HuggingFace (public repo, no token needed).

    Returns:
        Path to downloaded checkpoint
    """
    SAM3_MODEL_ID = "apozz/sam3-safetensors"
    SAM3_CKPT_NAME = "sam3.safetensors"

    checkpoint_path = hf_hub_download(
        repo_id=SAM3_MODEL_ID,
        filename=SAM3_CKPT_NAME,
    )
    return checkpoint_path


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone():
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
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
        self_attention=SplitMultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        ),
        cross_attention=SplitMultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
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
        cross_attention=SplitMultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
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
        out_norm=ops.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head():
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
    )

    cross_attend_prompt = SplitMultiheadAttention(
        embed_dim=256,
        num_heads=8,
        dropout=0,
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
        self_attention=SplitMultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=SplitMultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
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
    apply_temporal_disambiguation: bool, with_backbone: bool = False,
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
        vision_backbone = _create_vision_backbone()
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
    enable_inst_interactivity=True,
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone()
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


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

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
    log.info(f"Loading checkpoint with {len(sam3_image_ckpt)} keys ({len(inst_keys)} for inst_interactive_predictor)")

    # Convert nn.MultiheadAttention in_proj_weight/bias to split q/k/v
    sam3_image_ckpt = convert_mha_state_dict(sam3_image_ckpt)

    missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=False, assign=True)

    # Check for missing inst_interactive_predictor keys
    critical_missing = [k for k in missing_keys if 'inst_interactive_predictor' in k]
    if critical_missing:
        log.warning(f"Missing inst_interactive_predictor keys: {len(critical_missing)}")
        for k in critical_missing[:10]:
            log.warning(f"  MISSING: {k}")

    # Check for unexpected keys
    if unexpected_keys:
        inst_unexpected = [k for k in unexpected_keys if 'inst_interactive_predictor' in k]
        if inst_unexpected:
            log.warning(f"Unexpected inst_interactive_predictor keys: {len(inst_unexpected)}")
            for k in inst_unexpected[:5]:
                log.warning(f"  UNEXPECTED: {k}")

    if len(missing_keys) > 0:
        log.info(f"Total missing keys: {len(missing_keys)}")




# ---------------------------------------------------------------------------
# Public build functions
# ---------------------------------------------------------------------------

def build_sam3_image_model(
    bpe_path=None,
    device=None,
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    **kwargs,
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

    Returns:
        A SAM3 image model
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
        )

    # Create visual components
    vision_encoder = _create_vision_backbone(
        enable_inst_interactivity=enable_inst_interactivity
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
    segmentation_head = _create_segmentation_head() if enable_segmentation else None

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
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
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    model = model.to(device)
    if eval_mode:
        model.eval()

    return model


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device=None,
    enable_inst_interactivity: bool = False,
    **kwargs,
):
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    # Configure attention backend before building any modules
    attention_backend = kwargs.pop("attention_backend", "auto")
    kwargs.pop("compile", None)  # consumed by caller, not needed here
    from .attention import set_sam3_backend
    set_sam3_backend(attention_backend)

    if device is None:
        device = comfy.model_management.get_torch_device()

    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone(enable_inst_interactivity=True)
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
        out_norm=ops.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build instance interactive predictor if enabled
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
        )
    else:
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
        )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        ckpt = _load_checkpoint_file(checkpoint_path)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        remapped_ckpt = dict(ckpt)

        # If inst_interactive_predictor is enabled, remap tracker weights for it
        if enable_inst_interactivity and inst_predictor is not None:
            inst_predictor_keys = {
                k.replace("tracker.", "detector.inst_interactive_predictor.model."): v
                for k, v in remapped_ckpt.items()
                if k.startswith("tracker.")
            }
            remapped_ckpt.update(inst_predictor_keys)
            log.info(f"Added {len(inst_predictor_keys)} keys for detector.inst_interactive_predictor")

        # Convert nn.MultiheadAttention in_proj_weight/bias to split q/k/v
        remapped_ckpt = convert_mha_state_dict(remapped_ckpt)

        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_ckpt, strict=strict_state_dict_loading, assign=True
        )
        if missing_keys:
            log.info(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            log.info(f"Unexpected keys: {len(unexpected_keys)}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    # Use single-device predictor on CPU, multi-GPU predictor only when CUDA is available
    if comfy.model_management.get_torch_device().type != "cuda":
        return Sam3VideoPredictor(*model_args, **model_kwargs)
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )


__version__ = "0.1.0"
__all__ = [
    "build_sam3_image_model",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "_load_checkpoint_file",
    "convert_mha_state_dict",
]
