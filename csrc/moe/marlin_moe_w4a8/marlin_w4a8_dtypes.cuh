#pragma once

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

#ifndef MARLIN_W4A8_NAMESPACE_NAME
  #define MARLIN_W4A8_NAMESPACE_NAME marlin_moe_w4a8
#endif

namespace MARLIN_W4A8_NAMESPACE_NAME {

// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int USER_THREADS = 256; // Note: This is only used with user-provided thread_k/n
const int pipe_stages = 4; // 4 pipeline stages fit into shared memory
// const int SHARED_MEM = 96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

static constexpr int pack_factor_4bit =
    8;  // We have 8 4-bit vals inside a 32 bit

constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// No support for async
#else

// Predicated asynchronous global->shared copy; used for inputs A where we apply predication to handle batchsizes that
// are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}


// NOTE(HandH1998): cp.async.cg only support BYTES = 16, however,
// cp.async.ca can support BYTES = 4, 8, 16;
// as s1's shape is equal to prob_m, we need set s1 to float type,
// and cp_size = 1 float, i.e., 4 BYTES
// Asynchronous global->shared copy for activation quantizaton scales s1
__device__ inline void cp_async1(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.ca.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Matrix fragments for tensor core instructions; their precise layout is documented here: 
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-integer-type
using FragA = Vec<uint32_t, 2>;
using FragB = Vec<uint32_t, 1>;
using FragC = Vec<int, 4>;
using FragS_GROUP = Vec<half2, 1>; // weight per-group quantization scales
using FragS_CHANNEL = Vec<float, 2>; // weight per-channel quantization scales or activaton per-token quantization scales


inline __device__ half2 float2_to_half2(float2 f) {
  uint32_t res;
  // NOTE(HandH1998): h0,h1 should be uint16_t, not half
  uint16_t h0, h1;
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h0) : "f"(f.x));
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h1) : "f"(f.y));
  asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(res) : "h"(h0), "h"(h1));
  return reinterpret_cast<half2&>(res);
}

inline __device__ float int32_to_float(int h) {
  float res;
  asm volatile("cvt.rn.f32.s32 %0, %1;\n" : "=f"(res) : "r"(h));
  return res;
}

#endif

}  // namespace MARLIN_W4A8_NAMESPACE_NAME
