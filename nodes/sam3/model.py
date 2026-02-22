# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# ComfyUI-native model module — re-export shim.
# The actual implementations live in model_*.py files.

from .model_components import (  # noqa: F401
    _DeviceCacheMixin,
    _build_linear_stack,
    _dtype_debug,
    _SAM3_DEBUG,
    CXBlock,
    MLP,
    PatchEmbed,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
    SamMLP,
    TransformerWrapper,
    VitMlp,
    concat_rel_pos,
    get_abs_pos,
    get_rel_pos,
    window_partition,
    window_unpartition,
)

from .model_backbone import (  # noqa: F401
    Block,
    SAM3VLBackbone,
    Sam3DualViTDetNeck,
    ViT,
    ViTAttention,
)

from .model_transformer import (  # noqa: F401
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv1,
    TransformerDecoderLayerv2,
    TransformerEncoder,
    TransformerEncoderCrossAttention,
    TransformerEncoderFusion,
    TransformerEncoderLayer,
    _TransformerSelfCrossAttnLayer,
    pool_text_feat,
)

from .model_heads import (  # noqa: F401
    DotProductScoring,
    LinearPresenceHead,
    MaskDecoder,
    MaskEncoder,
    MaskPredictor,
    PixelDecoder,
    PromptEncoder,
    SegmentationHead,
    SequenceGeometryEncoder,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
    UniversalSegmentationHead,
)

from .model_detector import (  # noqa: F401
    Sam3Image,
    Sam3ImageOnVideoMultiGPU,
    _update_out,
)

from .model_tracker import (  # noqa: F401
    NO_OBJ_SCORE,
    SAM3InteractiveImagePredictor,
    Sam3TrackerBase,
    Sam3TrackerPredictor,
    concat_points,
)

from .model_video import (  # noqa: F401
    MaskletConfirmationStatus,
    Sam3VideoBase,
    Sam3VideoInference,
    Sam3VideoInferenceWithInstanceInteractivity,
)
