# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# ComfyUI-native text encoder.
# Consolidated from: model/text_encoder_ve.py, model/model_misc.py (LayerScale)

from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

import comfy.ops

ops = comfy.ops.manual_cast

from .attention import SplitMultiheadAttention


# ---------------------------------------------------------------------------
# LayerScale (from model/model_misc.py) — only nn.Parameter, no operations needed
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ---------------------------------------------------------------------------
# ResidualAttentionBlock
# ---------------------------------------------------------------------------

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        # Attention — SplitMultiheadAttention replaces nn.MultiheadAttention
        self.attn = SplitMultiheadAttention(
            d_model, n_head, batch_first=True,
            dtype=dtype, device=device, operations=operations,
        )

        # LayerNorm
        self.ln_1 = operations.LayerNorm(d_model, dtype=dtype, device=device)
        self.ln_2 = operations.LayerNorm(d_model, dtype=dtype, device=device)

        # LayerScale
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        # MLP — preserve "c_fc" and "c_proj" key names for state dict compatibility
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", operations.Linear(d_model, mlp_width, dtype=dtype, device=device)),
                    ("gelu", act_layer()),
                    ("c_proj", operations.Linear(mlp_width, d_model, dtype=dtype, device=device)),
                ]
            )
        )

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None:
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        dtype=None,
        device=None,
        operations=ops,
        **kwargs,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def text_global_pool(
    x: torch.Tensor, text: Optional[torch.Tensor] = None, pool_type: str = "argmax"
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


# ---------------------------------------------------------------------------
# TextTransformer
# ---------------------------------------------------------------------------

class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        dtype=None,
        device=None,
        operations=ops,
        **kwargs,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.token_embedding = operations.Embedding(self.vocab_size, width, dtype=dtype, device=device)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.ln_final = (
            operations.LayerNorm(width, dtype=dtype, device=device)
            if use_ln_post
            else nn.Identity()
        )
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )
        if proj_bias:
            self.text_projection = operations.Linear(width, output_dim, dtype=dtype, device=device)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self) -> torch.Tensor:
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(
        self, text: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = text.shape[1]
        x = self.token_embedding(text)

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(x.dtype)
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection.to(pooled.dtype)
        if self.output_tokens:
            return pooled, tokens
        return pooled


# ---------------------------------------------------------------------------
# VETextEncoder — top-level text encoder
# ---------------------------------------------------------------------------

class VETextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        tokenizer: Callable,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            output_tokens=True,
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.resizer = operations.Linear(self.encoder.width, d_model, dtype=dtype, device=device)

    def forward(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(text[0], str):
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            tokenized = self.tokenizer(text, context_length=self.context_length).to(
                device
            )
            text_attention_mask = (tokenized != 0).bool()

            inputs_embeds = self.encoder.token_embedding(tokenized)
            _, text_memory = self.encoder(tokenized)

            assert text_memory.shape[1] == inputs_embeds.shape[1]
            text_attention_mask = text_attention_mask.ne(1)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = self.resizer(text_memory)
        else:
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert (
                input_boxes is None or len(input_boxes) == 0
            ), "Can't replace boxes in text if it's already encoded"

        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )
