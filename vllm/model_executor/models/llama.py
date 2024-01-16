# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul, DequantSiluAndMulQuant
from vllm.model_executor.layers.attention import PagedAttention, DequantPagedAttentionQuant
from vllm.model_executor.layers.layernorm import RMSNorm, RMSNormQuant, DequantAddResidualI8RMSNormQuant
from vllm.model_executor.layers.fusion import DequantAddResidual
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               SQRowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.use_int8 = quant_config is not None and quant_config.get_name(
        ) == "smoothquant"
        self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2,
                bias=False,
                linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        if self.use_int8:
            self.down_proj = SQRowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                linear_method=linear_method)
            self.act_fn = DequantSiluAndMulQuant(use_per_token_quant=True)
        else:
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                linear_method=linear_method)
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        scale = None
        if self.use_int8:
            gate_dequant_scale = self.gate_up_proj.gate_dequant_scale.item()
            up_dequant_scale = self.gate_up_proj.up_dequant_scale.item()
            x, *scale = self.act_fn(gate_up, gate_dequant_scale, up_dequant_scale)
            scale = scale[0] if scale is not None else None
            x, _ = self.down_proj(x, scale)
        else:
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        return x, scale


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
        quant_config: Optional[QuantizationConfig] = None,
        quant_kv_cache: bool = False,
        kv_quant_params: List[float] = None
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_int8 = quant_config is not None and quant_config.get_name(
        ) == "smoothquant"
        self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                linear_method=linear_method,
            )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            need_dequant=self.use_int8,
        )
        if self.use_int8:
            self.o_proj = SQRowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                linear_method=linear_method,
            )
            self.attn = DequantPagedAttentionQuant(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                quant_kv_cache=quant_kv_cache,
                kv_quant_params=kv_quant_params,
                use_per_token_quant=True)
        else:
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                linear_method=linear_method,
            )
            self.attn = PagedAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                quant_kv_cache=quant_kv_cache,
                kv_quant_params=kv_quant_params)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k_cache, v_cache = kv_cache
        scale = None
        if self.use_int8:
            q_dequant_scale = self.qkv_proj.q_dequant_scale.item()
            k_dequant_scale = self.qkv_proj.k_dequant_scale.item()
            v_dequant_scale = self.qkv_proj.v_dequant_scale.item()
            q, k, v = self.rotary_emb(positions, q, k, v,
                                      q_dequant_scale,
                                      k_dequant_scale,
                                      v_dequant_scale)
            attn_output, *scale = self.attn(q, k, v, k_cache,
                                            v_cache, input_metadata,
                                            q_dequant_scale,
                                            k_dequant_scale,
                                            v_dequant_scale)
            scale = scale[0] if scale is not None else None
            output, _ = self.o_proj(attn_output, scale)
        else:
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, k_cache, v_cache,
                                    input_metadata)
            output, _ = self.o_proj(attn_output)
        return output, scale


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        quant_config: Optional[QuantizationConfig] = None,
        quant_kv_cache: bool = False,
        kv_quant_params: List[float] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_int8 = quant_config is not None and quant_config.get_name(
        ) == "smoothquant"
        self.tp_size = get_tensor_model_parallel_world_size()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
            quant_config=quant_config,
            quant_kv_cache=quant_kv_cache,
            kv_quant_params=kv_quant_params,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
            quant_config=quant_config,
        )
        if self.use_int8:
            self.input_layernorm = RMSNormQuant(config.hidden_size,
                                                eps=config.rms_norm_eps)
            if self.tp_size > 1:
                self.post_attention_layernorm = RMSNormQuant(config.hidden_size, eps=config.rms_norm_eps)
            else:
                self.post_attention_layernorm = DequantAddResidualI8RMSNormQuant(config.hidden_size, eps=config.rms_norm_eps)
                self.dequant_add_residual = DequantAddResidual()
        else:
            self.input_layernorm = RMSNorm(config.hidden_size,
                                           eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states, scale = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        if self.use_int8:
            if self.tp_size > 1:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)
                hidden_states, _ = self.mlp(hidden_states)
            else:
                o_dequant_scale = self.self_attn.o_proj.dequant_scale.item()
                down_dequant_scale = self.mlp.down_proj.dequant_scale.item()
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual, o_dequant_scale, scale)
                hidden_states, scale = self.mlp(hidden_states)
                hidden_states, residual = self.dequant_add_residual(hidden_states, residual, down_dequant_scale, scale)
        else:
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states, _ = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        quant_config: Optional[QuantizationConfig] = None,
        quant_kv_cache: bool = False,
        kv_quant_params_list: List[List[float]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                config, linear_method, quant_config, quant_kv_cache,
                kv_quant_params_list[i] if quant_kv_cache else None)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        quant_config: Optional[QuantizationConfig] = None,
        quant_kv_cache: bool = False,
        kv_quant_params_list: List[List[float]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.quant_config = quant_config
        self.model = LlamaModel(config, linear_method, quant_config, quant_kv_cache,
                                kv_quant_params_list)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        # For SmoothQuant
        int8_fusion = False
        if self.quant_config is not None and self.quant_config.get_name(
        ) == "smoothquant":
            int8_fusion = True
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # bias is useless for llama
            if "bias" in name:
                continue
            # load dequant scale for qkv_proj and gate_up_proj
            if int8_fusion:
                is_fusion_scale = False
                if "scale" in name:
                    for (param_name, weight_name, shard_id) in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        prefix = weight_name.split('_')[0]
                        suffix = name.split('.')[-1]
                        new_name = prefix + '_' + suffix
                        param = params_dict[name.replace(suffix, new_name)]
                        param.copy_(loaded_weight)
                        is_fusion_scale = True
                        break
                    if is_fusion_scale:
                        continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
