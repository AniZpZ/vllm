#pragma once

#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  bool enable_quant = false,
  float k_scale = 1.0f,
  float k_zp = 0.0f,
  float v_scale = 1.0f,
  float v_zp = 0.0f);

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  bool enable_quant = false,
  float k_scale = 1.0f,
  float k_zp = 0.0f,
  float v_scale = 1.0f,
  float v_zp = 0.0f);

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void fused_add_rms_norm(
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon);

void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox);

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

#ifndef USE_ROCM
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);
#endif

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama);

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm);

// These are kernels used by smoothquant
void rms_norm_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void dequant_add_residual_rms_norm_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& gamma,
  float scale,
  float epsilon);

void dequant_add_residual_rms_norm_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& gamma,
  torch::Tensor& scale,
  float epsilon,
  float weight_dequant_scale);

void add_residual_rms_norm_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon);

void dequant_rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox,
  torch::Tensor& query_out,
  torch::Tensor& key_out,
  float query_scale,
  float key_scale);

void dequant_silu_and_mul_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  float gate_scale,
  float up_scale,
  float out_scale);

void dequant_silu_and_mul_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  float gate_scale,
  float up_scale,
  torch::Tensor& out_scale,
  torch::Tensor& tmp);

void dequant_add_residual(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  float scale);

void dequant_add_residual(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& scale,
  float weight_dequant_scale);

void dequant(
  torch::Tensor& out,
  torch::Tensor& input,
  float scale);

void dequant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& scale,
  float weight_dequant_scale);

void quant(
  torch::Tensor& out,
  torch::Tensor& input,
  float scale);

void quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& scale);
