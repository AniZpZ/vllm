/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

 #ifndef MARLIN_W4A8_NAMESPACE_NAME
 #define MARLIN_W4A8_NAMESPACE_NAME marlin_moe_w4a8
#endif

#include "core/scalar_type.hpp"
#include "marlin_w4a8_dtypes.cuh"


namespace MARLIN_W4A8_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <typename scalar_t,  // compute dtype, half or nv_float16
         const int threads,          // number of threads in a threadblock
         const int thread_m_blocks,  // number of 16x16 blocks in the m
                                     // dimension (batchsize) of the
                                     // threadblock
         const int thread_n_blocks,  // same for n dimension (output)
         const int thread_k_blocks,  // same for k dimension (reduction)
         const int stages,  // number of stages for the async global->shared
                            // fetch pipeline
         const int group_blocks,    // number of consecutive 16x16 blocks
                                    // with a separate quantization scale
         >
__global__ void Marlin(
   const int4* __restrict__ A,  // int8 input matrix of shape mxk 
   const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
       int4* __restrict__ C, // int32 global_reduce buffer of shape (max_par*16*4)xn , as int8 tensor core's output is int32 dtype
       int4* __restrict__ D, // fp16 output buffer of shape mxn
   const float* __restrict__ s1, // fp32 activation per-token quantization scales of shape mx1
   const int4* __restrict__ s2, // fp32 weight per-channel quantization scales of shape 1xn 
   const int4* __restrict__ s3, // fp16 weight per-group quantization scales of shape (k/groupsize)xn, when group_blocks=-1, it should be nullptr
   const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
   const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
   const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
   const float* __restrict__ topk_weights_ptr,              // moe top weights
   int top_k,              // num of experts per token
   bool mul_topk_weights,  // mul topk weights or not
   bool is_ep,             // expert parallelism
   int num_groups,         // number of scale groups per output channel
   int prob_m,             // batch dimension m
   int prob_n,             // output dimension n
   int prob_k,             // reduction dimension k
   int* locks,             // extra global storage for barrier synchronization
   bool use_atomic_add,    // whether to use atomic add to reduce
   bool use_fp32_reduce    // whether to use fp32 global reduce
)  {}

}  // namespace MARLIN_W4A8_NAMESPACE_NAME

#else

// m16n8k16 tensor core mma instruction with int8 inputs and int32 output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
 const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
 const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
 int* c = reinterpret_cast<int*>(&frag_c);
 asm volatile(
   "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
   : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
   :  "r"(a[0]),  "r"(a[1]),  "r"(b[0]),
      "r"(c[0]),  "r"(c[1]),  "r"(c[2]),  "r"(c[3])
 );
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
 uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
 uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
 asm volatile(
   "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
   : "=r"(a[0]), "=r"(a[1]) : "r"(smem)
 );
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
 uint32_t res;
 asm volatile(
   "lop3.b32 %0, %1, %2, %3, %4;\n"
   : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
 );
 return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
 uint32_t res;
 asm volatile("prmt.b32 %0, %1, %2, %3;\n"
              : "=r"(res)
              : "r"(a), "n"(start_byte), "n"(mask));
 return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 int8 values for weight per channel dequant.
__device__ inline FragB dequant_per_channel(int q) {
 static constexpr int MASK = 0xf0f0f0f0;
 FragB frag_b;
 frag_b[0] = (q & MASK);
 return frag_b;
}

// TODO(HandH1998): optimize dequant_per_group, as it doesn't have a very good performance for now
// Efficiently dequantize an int32 value into a full B-fragment of 4 int8 values for weight per group dequant.
__device__ inline FragB dequant_per_group(int q, FragS_GROUP& frag_s, int i) {
 // convert 4 int8 to 4 half
 static constexpr uint32_t LO = 0x000f000f;
 static constexpr uint32_t HI = 0x00f000f0;
 static constexpr uint32_t EX = 0x64006400;
 // Guarantee that the `(a & b) | c` operations are LOP3s.
 uint32_t t0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
 uint32_t t1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
 // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
 static constexpr uint32_t SUB = 0x64086408;
 static constexpr uint32_t MUL = 0x2c002c00;
 static constexpr uint32_t ADD = 0xd480d480;
 *reinterpret_cast<half2*>(&t0) = __hsub2(
   *reinterpret_cast<half2*>(&t0),
   *reinterpret_cast<const half2*>(&SUB)
 );
 *reinterpret_cast<half2*>(&t1) = __hfma2(
   *reinterpret_cast<half2*>(&t1),
   *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
 );

 uint16_t s = reinterpret_cast<uint16_t*>(&frag_s)[i];
 uint32_t double_s;
 // pack 2xfp16 to half2
 asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(double_s) : "h"(s), "h"(s));
 // dequant and convert 4 half to 4 uint8 (be placed at the low 8 bits of 4 half, respectively)
 static constexpr uint32_t MAGIC_NUM = 0x64806480;
 *reinterpret_cast<half2*>(&t0) = __hfma2(
   *reinterpret_cast<half2*>(&t0),
   *reinterpret_cast<half2*>(&double_s), *reinterpret_cast<const half2*>(&MAGIC_NUM)
 );
 *reinterpret_cast<half2*>(&t1) = __hfma2(
   *reinterpret_cast<half2*>(&t1),
   *reinterpret_cast<half2*>(&double_s), *reinterpret_cast<const half2*>(&MAGIC_NUM)
 );
 // take out the 4 uint8 from 4 half, then convert them to 4 int8 and pack 4 int8 into 1 uint32
 FragB frag_b;
 uint32_t uint8s;
 static constexpr uint32_t MASK_0246     = 0x6420;
 static constexpr uint32_t UINT8s_TO_INT8s_MASK    = 0x80808080;
 asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(uint8s) : "r"(t0), "r"(t1), "n"(MASK_0246));
 frag_b[0] = (uint8s ^ UINT8s_TO_INT8s_MASK);
 return frag_b;
}


// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
 if (threadIdx.x == 0) {
   int state = -1;
   do
     // Guarantee that subsequent writes by this threadblock will be visible
     // globally.
     asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                  : "=r"(state)
                  : "l"(lock));
   while (state != count);
 }
 __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
 __syncthreads();
 if (threadIdx.x == 0) {
   if (reset) {
     lock[0] = 0;
     return;
   }
   int val = 1;
   // Make sure that all writes since acquiring this barrier are visible
   // globally, while releasing the barrier.
   asm volatile("fence.acq_rel.gpu;\n");
   asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                :
                : "l"(lock), "r"(val));
 }
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
 if (threadIdx.x == 0) {
   int state = 0;
   do
     // Guarantee that subsequent writes by this threadblock will be visible
     // globally.
     asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                  : "=r"(state)
                  : "l"(lock));
   while (state >= 0);
   atomicAdd(lock, 1);
 }
 __syncthreads();
}


template <typename scalar_t,  // compute dtype, half or nv_float16
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,    // number of consecutive 16x16 blocks
                                     // with a separate quantization scale
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // int8 input matrix of shape mxk 
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
        int4* __restrict__ C, // int32 global_reduce buffer of shape (max_par*16*4)xn , as int8 tensor core's output is int32 dtype
        int4* __restrict__ D, // fp16 output buffer of shape mxn
    const float* __restrict__ s1, // fp32 activation per-token quantization scales of shape mx1
    const int4* __restrict__ s2, // fp32 weight per-channel quantization scales of shape 1xn 
    const int4* __restrict__ s3, // fp16 weight per-group quantization scales of shape (k/groupsize)xn, when group_blocks=-1, it should be nullptr
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    bool is_ep,             // expert parallelism
    int num_groups,         // number of scale groups per output channel
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks,             // extra global storage for barrier synchronization
    bool use_atomic_add,    // whether to use atomic add to reduce
    bool use_fp32_reduce    // whether to use fp32 global reduce
) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  // 4bit weight
  constexpr int pack_factor = 32 / 4;  
  constexpr int moe_block_size = 16 * thread_m_blocks;

  // const int scales_expert_stride = prob_n * prob_k / group_size / 8;
  // hongbo: check this
  const int s2_expert_stride = prob_n * prob_k / group_size / 8; // 每个 expert 的 scale 偏移
  const int s3_expert_stride = prob_n / 4; // 每个 expert 的列方向 s2 scale 偏移（int4，每个含4个float）

  // parallel: num valid moe blocks
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int parallel = num_tokens_past_padded / moe_block_size;
  int num_valid_blocks = parallel;
  if (is_ep) {
    for (int i = 0; i < parallel; i++) {
      if (expert_ids_ptr[i] == -1) num_valid_blocks--;
    }
  }
  int num_invalid_blocks = parallel - num_valid_blocks;
  parallel = num_valid_blocks;

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

  if constexpr (group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              ceildiv(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;  // number of threadblock tiles in the current slice
  int slice_count =
      0;          // total number of active threadblocks in the current slice
  int slice_idx;  // index of threadblock in current slice; numbered bottom to
                  // top

  // moe related
  int par_id = 0;
  int block_id = -1;
  int64_t expert_id = 0;  // use int64 to avoid computation result overflow
  int64_t old_expert_id = 0;
  int64_t B_expert_off = 0;

  float block_topk_weights[moe_block_size];
  int32_t block_sorted_ids[moe_block_size];
  int32_t block_num_valid_tokens = 0;
  int32_t locks_off = 0;

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    slice_col = slice_col_par % n_tiles;
    par_id = slice_col_par / n_tiles;
    // hongbosherlock: when to load s1
    s1 += (slice_col_par / n_tiles) * 16 * thread_m_blocks;

  }
  if (parallel * n_tiles >= gridDim.x) {
    // when parallel * n_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  // read moe block data given block_id
  // block_sorted_ids / block_num_valid_tokens / block_topk_weights
  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] =
          ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= prob_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }

    if (mul_topk_weights) {
      for (int i = 0; i < block_num_valid_tokens; i++) {
        block_topk_weights[i] = topk_weights_ptr[block_sorted_ids[i]];
      }
    }
  };

  // when move to next moe block, find the next block_id and expert_id
  // and then read moe block data
  auto update_next_moe_block_data = [&]() {
    if (par_id >= parallel) return;

    old_expert_id = expert_id;
    if (num_invalid_blocks > 0) {
      int skip_count = block_id == -1 ? par_id : 0;
      block_id++;
      for (int i = block_id; i < num_tokens_past_padded / moe_block_size; i++) {
        expert_id = expert_ids_ptr[i];
        if (expert_id != -1) {
          if (skip_count == 0) {
            block_id = i;
            break;
          };
          skip_count--;
        };
      }
    } else {
      block_id = par_id;
      expert_id = expert_ids_ptr[block_id];
    }

    B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);
    // scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;
    // update scale pointer（s2/s3）
    s3 += (expert_id - old_expert_id) * s3_expert_stride;
    s2 += (expert_id - old_expert_id) * s2_expert_stride;

    read_moe_block_data(block_id);
  };

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&](bool first_init = false) {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (parallel * n_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          ceildiv(block_num_valid_tokens, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < block_num_valid_tokens) {
          int64_t sorted_row = block_sorted_ids[row];
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
      // After write zero to output, write a negative value to lock.
      // Every SM that processes the same slice would wait for
      // the negative value, and then atomicAdd 1 to it.
      // After all SMs are processed, the lock value would back to 0 again.
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      // hongbosherlock: when to update s1
      s1 += 16 * thread_m_blocks;
      slice_col = 0;
      par_id++;
      update_next_moe_block_data();
    }
  };

  update_next_moe_block_data();
  init_slice(true);

  // start 
  // A sizes/strides
  int a_gl_stride = prob_k / 16; // stride of the A matrix in global memory
  // We typically use `constexpr` to indicate that this value is a compile-time constant
  constexpr int a_sh_stride = 16 * thread_k_blocks / 16; // stride of an A matrix tile in shared memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 16; // delta between subsequent A tiles in global memory
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o); // between subsequent accesses within a tile
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o); // between shared memory writes
  constexpr int a_sh_rd_delta_o = 1 * ((threads / 32) / (thread_n_blocks / 4)); // between shared memory tile reads
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16; // within a shared memory tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks); // overall size of a tile
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta); // number of shared write iterations for a tile

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // s1: per-token activation scale
  constexpr int s1_sh_stride = 16 * thread_m_blocks;
  int s1_gl_rd = threadIdx.x;
  // NOTE(HandH1998): activation scale s1 need shuffle to [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
  // for example, 0, 8 row scales serve for thread 0, 1, 2, 3. For more details, refer to mma operand A layout
  // as s1's size is not fixed, we can not shuffle before inference
  // we shuffle it when fetching s1 from global memory to shared memory, that's why s1_sh_wr is like this
  int s1_sh_wr = (threadIdx.x / 16) * 16 + (threadIdx.x % 8) * 2 + (threadIdx.x % 16) / 8;
  int s1_sh_rd = (threadIdx.x % 32) / 4;
  bool s1_sh_wr_pred = threadIdx.x < prob_m;

  // s2: per-channel weight scale
  constexpr int s2_sh_stride = 16 * thread_n_blocks / 4;
  // hongbosherlcok: check if need this
  // int slice_col = 0
  int s2_gl_rd = s2_sh_stride * slice_col + threadIdx.x;
  int s2_sh_wr = threadIdx.x;
  int s2_sh_rd = 16 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + 2 * ((threadIdx.x % 32) % 4);
  bool s2_sh_wr_pred = threadIdx.x < s2_sh_stride;

  // s3: grouped scale (k/groupsize x n)
  int s3_gl_stride = prob_n / 8;
  constexpr int s3_sh_stride = 16 * thread_n_blocks / 8;
  
  constexpr int s_tb_groups =
      group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s3_sh_stage = s_tb_groups * s3_sh_stride;

  // constexpr int s3_sh_stage = s3_sh_stride;
  int s3_gl_rd_delta = s3_gl_stride;

  int s3_gl_rd, s3_sh_wr, s3_sh_rd;
  bool s3_sh_wr_pred;
  if constexpr (group_blocks != -1) {
    s3_gl_rd = s3_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s3_sh_stride * slice_col + threadIdx.x;
    s3_sh_wr = threadIdx.x;
    // NOTE(HandH1998): s3_sh_rd is related to mma output C
    s3_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
    s3_sh_wr_pred = threadIdx.x < s3_sh_stride;
  }

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  // NOTE(HandH1998): int8 input a only need 16 threads to load 16x16 matrix
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16);
  a_sh_rd += 1 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;


  // hongbosherlcok: don't need this
  // Precompute which thread should not read memory in which iterations; this is needed if there are more threads than
  // required for a certain tilesize or when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;


  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s1 = sh_b + (stages * b_sh_stage);
  int4* sh_s2 = sh_s1 + s1_sh_stride;
  int4* sh_s3 = sh_s2 + s2_sh_stride;
  // hongbosherlock: optinal values
//   int4* sh_g_idx = sh_b + (stages * b_sh_stage);
//   int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
//   int4* sh_red = sh_s + (stages * s_sh_stage);

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS_GROUP frag_s3[2][4];
  FragS_CHANNEL frag_s1[thread_m_blocks];
  FragS_CHANNEL frag_s2[2][4];

  // hongbosherlock: optinal values
  FragS frag_s[2][4];                    // No act-order
//   FragS act_frag_s[2][4][4];             // For act-order
//   int frag_qzp[2][num_ints_per_thread];  // Zero-points
//   FragZP frag_zp;                        // Zero-points in fp16
//   FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ

 // end 

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;
  constexpr int sh_max_num_groups = 32;

  auto fetch_scales_to_shared = [&](bool is_async, int first_group_id,
                                    int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups < sh_max_num_groups) {
      sh_num_groups = sh_max_num_groups;
    }

    if (sh_first_group_id + sh_num_groups > num_groups) {
      sh_num_groups = num_groups - sh_first_group_id;
    }

    int row_offset = first_group_id * s_gl_stride;

    if (is_async) {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          cp_async4_pred(&sh_s[(i * s_sh_stride) + threadIdx.x],
                         &scales_ptr[row_offset + (i * s_gl_stride) +
                                     slice_n_offset + threadIdx.x]);
        }
      }
    } else {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          sh_s[(i * s_sh_stride) + threadIdx.x] =
              scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset +
                         threadIdx.x];
        }
      }
    }
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        int a_idx = a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off;
        int row = a_idx / a_gl_stride;
        int64_t sorted_row = block_sorted_ids[row] / top_k;
        int64_t true_idx = sorted_row * a_gl_stride + a_idx % a_gl_stride;
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[true_idx],
                       row < block_num_valid_tokens);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j],
                    B_ptr[i] + j + B_expert_off);
        B_ptr[i] += b_gl_rd_delta_o;
      }

        if constexpr (group_blocks != -1) {
          // int4* sh_s_stage = sh_s + s_sh_stage * pipe;
          int4* sh_s3_stage = sh_s3 + s3_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
            if (s3_sh_wr_pred) {
              cp_async4(&sh_s3_stage[s3_sh_wr], &s3[s3_gl_rd]);
            }
            // Only fetch scales if this tile starts a new group
            if ((pipe + 1) % (group_blocks / thread_k_blocks) == 0) {
                s3_gl_rd += s3_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < s_tb_groups; i++) {
              if (s3_sh_wr_pred) {
                cp_async4(&sh_s3_stage[i * s3_sh_stride + s3_sh_wr],
                          &s3[s3_gl_rd]);
              }
              s3_gl_rd += s3_gl_rd_delta;
            }
          }
        }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };


  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    
    // hongbosherlock: how to load scale
    // It may seem inefficient that we reload the groups for every sub-tile; however, this does not seem to be a
    // significant bottleneck, while some theoretically better attempts have lead to bad instruction ordering by the
    // compiler and correspondingly a noticable drop in performance.
    if constexpr (group_blocks != -1) {
      int4* sh_s3_stage = sh_s3 + s3_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
      reinterpret_cast<int4*>(&frag_s3[k % 2])[0] = sh_s3_stage[s3_sh_rd];
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i],
                      &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };


  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;

    if constexpr (!has_act_order) {
      // No act-order case
      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        } else {
          int warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
              sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
        }
      }

      return;
    }

    // Act-order case
  };

  // hongbosherlock: optional
  // auto fetch_zp_to_registers 

  // Execute the actual tensor core matmul of a sub-tile. 
  auto matmul = [&] (int k) {
    // We have the m dimension as the inner loop in order to encourage overlapping dequantization and matmul operations.
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];
      // int b_quant_shift = b_quant << 4;
      FragB frag_b0, frag_b1;
      // If there are no groups, we can just scale the final output once and can avoid doing so for each weight.
      if constexpr (group_blocks != -1) {
        int b_quant_shift = b_quant >> 8;
        frag_b0 = dequant_per_group(b_quant, frag_s3[k % 2][j], 0);
        frag_b1 = dequant_per_group(b_quant_shift, frag_s3[k % 2][j], 1);
      } else {
        int b_quant_shift = b_quant << 4;
        frag_b0 = dequant_per_channel(b_quant);
        frag_b1 = dequant_per_channel(b_quant_shift);
      }
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                int* c_rd = reinterpret_cast<int*>(
                    &sh[red_sh_delta * j + red_sh_rd]);
                int* c_wr = reinterpret_cast<int*>(&sh[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };


  // Since multiple threadblocks may process parts of the same column slice, we finally have to globally reduce over
  // the results. As the striped partioning minimizes the number of such reductions and our outputs are usually rather
  // small, we perform this reduction serially in L2 cache.
  // global_reduce works on INT32 elements, which are the results of INT8 GEMM.
  // This is why we need another INT32 maxtrix `C` to reduce instead of the
  // original half matrix `D`.
  auto global_reduce = [&] (bool first = false, bool last = false) {
      // We are very careful here to reduce directly in the output buffer to maximize L2 cache utilization in this step. 
     // To do this, we write out results in FP16 (but still reduce with FP32 compute).
      constexpr int active_threads = 32 * thread_n_blocks / 4;
      if (threadIdx.x < active_threads) {
            int c_gl_stride = prob_n / 4;
            int c_gl_wr_delta_o = 8 * c_gl_stride;
            int c_gl_wr_delta_i = 8 * (active_threads / 32);
            int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 8 * (threadIdx.x / 32) + (threadIdx.x % 4) * 2;
            c_gl_wr += (4 * thread_n_blocks) * slice_col;
            constexpr int c_sh_wr_delta = active_threads * 2;
            int c_sh_wr = 2 * threadIdx.x;
            int row = (threadIdx.x % 32) / 4;

            if (!first) {
                #pragma unroll
                for (int i = 0; i < thread_m_blocks * 4; i++) {
                    int c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
                    int64_t sorted_row = block_sorted_ids[c_idx / c_gl_stride];
                    int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
                    if (c_idx / c_gl_stride < block_num_valid_tokens)
                      sh[c_sh_wr + c_sh_wr_delta * i] = C[true_idx];
                    
                    // cp_async4_pred(
                    //     &sh[c_sh_wr + c_sh_wr_delta * i],
                    //     &C_int[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
                    //     i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
                    // cp_async4_pred(
                    //     &sh[c_sh_wr + c_sh_wr_delta * i + 1],
                    //     &C_int[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1],
                    //     i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
                }
                // cp_async_fence();
                // cp_async_wait<0>();
            }

            #pragma unroll
            for (int i = 0; i < thread_m_blocks * 4; i++) {
                if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
                    if (!first) {
                        int4 d_red1 = sh[c_sh_wr + i * c_sh_wr_delta];
                        int4 d_red2 = sh[c_sh_wr + i * c_sh_wr_delta + 1];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += 
                                reinterpret_cast<int*>(&d_red1)[j];
                        }
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)] += 
                                reinterpret_cast<int*>(&d_red2)[j];
                        }
                    }
                    if (!last) {
                        int4 d1, d2;
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&d1)[j] = 
                                reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)];
                        }
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&d2)[j] = 
                                reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)];
                        }
                    int c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
                    int64_t sorted_row = block_sorted_ids[c_idx / c_gl_stride];
                    int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
                    if (c_idx / c_gl_stride < block_num_valid_tokens) C[true_idx] = d1;
                    // hongbo: why d1 and d2
                        // C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = d1;
                        // C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1] = d2;
                    }
                }
            }
        }
    };

  // Write out the reduce final result in the correct layout. We only actually reshuffle matrix fragments in this step,
  // the reduction above is performed in fragment layout. 
  auto write_result = [&] () {
    int d_gl_stride = prob_n / 8;
    constexpr int d_sh_stride = 2 * thread_n_blocks + 1;
    int d_gl_wr_delta = d_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int d_sh_rd_delta = d_sh_stride * (threads / (2 * thread_n_blocks));

    int d_gl_wr = d_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    d_gl_wr += (2 * thread_n_blocks) * slice_col;
    int d_sh_wr = (4 * d_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    d_sh_wr += 32 * (threadIdx.x / 32);
    int d_sh_rd = d_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int d_gl_wr_end = d_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final global write patterns
    auto write = [&] (int idx, int c0, int c1, float a_s, FragS_CHANNEL& w_s) {
      float2 deq_res;
      deq_res.x = int32_to_float(c0) * w_s[0] * a_s;
      deq_res.y = int32_to_float(c1) * w_s[1] * a_s;
      ((half2*) sh)[idx] = float2_to_half2(deq_res);
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = d_sh_wr + 8 * j;
          write(wr + (4 * d_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s1[i][0], frag_s2[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * d_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s1[i][1], frag_s2[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * d_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s1[i][0], frag_s2[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * d_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s1[i][1], frag_s2[j / 2][2 * (j % 2) + 1]);
        }
        d_sh_wr += 16 * (4 * d_sh_stride);
      }
    }
    __syncthreads();

  // hongbosherlock todo check this code
  #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      int row = d_gl_wr / d_gl_stride;
      int64_t sorted_row = block_sorted_ids[row];
      int64_t true_idx = sorted_row * d_gl_stride + d_gl_wr % d_gl_stride;
      half2 topk_weight_score = __float2half2_rn(block_topk_weights[row]);
      if (row < block_num_valid_tokens) {
          if ((use_atomic_add && slice_count > 1) || mul_topk_weights) {
            half2* C_half2 = reinterpret_cast<half2*>(&C[true_idx]);
            half2* sh_red_half2 = reinterpret_cast<half2*>(&sh[d_sh_rd]);
    #pragma unroll
            for (int a = 0; a < 4; a++) {
                half2 res = sh_red_half2[a];
                if (mul_topk_weights){
                  res = __hmul2(res, topk_weight_score);
                } 
                if (use_atomic_add && slice_count > 1) {
                        atomicAdd(&C_half2[a], res);
                    } else {
                        C_half2[a] = res;
                      }
                    }
                } else {
                    D[true_idx] = sh[d_sh_rd];
                }
                d_gl_wr += d_gl_wr_delta;
                d_sh_rd += d_sh_rd_delta;
            }
        }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&] () {
  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      fetch_to_shared(i, i, i < slice_iters);
    }
    zero_accums();
    wait_for_stage();
    // init_same_group(0);
    fetch_to_registers(0, 0);
    // fetch_scales_to_registers(0, 0);
    // fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    // slice_k_start_shared_fetch += tb_k * (stages - 1); // act_order
  };
  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        // fetch_scales_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before write-out
      // hongbo: if add this condition
      if constexpr (group_blocks == -1) {
          if (last || use_atomic_add) {
            if (s1_sh_wr_pred) {
              cp_async1(&sh_s1[s1_sh_wr], &s1[s1_gl_rd]);
            }
            if (s2_sh_wr_pred) {
              cp_async4(&sh_s2[s2_sh_wr], &s2[s2_gl_rd]);
            }
            cp_async_fence();
          }
      }

      thread_block_reduce();
      // hongbosherlock why "group_blocks == -1"
      if constexpr (group_blocks == -1) {
          if (last || use_atomic_add) {
            cp_async_wait<0>();
            __syncthreads();
            if (threadIdx.x / 32 < thread_n_blocks / 4) {
              #pragma unroll
              for (int i = 0; i < thread_m_blocks; i++) {
                frag_s1[i][0] = *reinterpret_cast<float*>(&sh_s1[16 * i + 2 * s1_sh_rd]);
                frag_s1[i][1] = *reinterpret_cast<float*>(&sh_s1[16 * i + 2 * s1_sh_rd + 1]);
              }
              reinterpret_cast<int4*>(&frag_s2)[0] = sh_s2[s2_sh_rd + 0];
              reinterpret_cast<int4*>(&frag_s2)[1] = sh_s2[s2_sh_rd + 1];
              reinterpret_cast<int4*>(&frag_s2)[2] = sh_s2[s2_sh_rd + 8];
              reinterpret_cast<int4*>(&frag_s2)[3] = sh_s2[s2_sh_rd + 9];
            }
          }
      }

      if (slice_count > 1) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[locks_off], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[locks_off], last);
      }
      // if (use_atomic_add && slice_count > 1 && slice_idx != 0)
      //   wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
  #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
              B_ptr[i] -= b_gl_stride;
        }
        s3_gl_rd = s3_sh_stride * slice_col + threadIdx.x;
        s2_gl_rd = s2_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

}  // namespace MARLIN_W4A8_NAMESPACE_NAME

#endif
