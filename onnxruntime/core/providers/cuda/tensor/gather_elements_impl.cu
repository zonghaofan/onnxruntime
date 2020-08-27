// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_elements_impl.h"

#include <thread>

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int threads_per_block = GridDim::maxThreadsPerBlock;
constexpr int thread_worksize = 16;
}
__host__ __device__ inline int64_t GetIndexValue(const void* index_data, size_t index_element_size, size_t offset) {
  switch (index_element_size) {
    case sizeof(int32_t):
      return *(reinterpret_cast<const int32_t*>(index_data) + offset);
      break;
    case sizeof(int64_t):
      return *(reinterpret_cast<const int64_t*>(index_data) + offset);
      break;
    default:
      break;
  }
  // This should cause abort
  assert(false);
  return std::numeric_limits<int64_t>::max();
}

template <typename T>
__global__ void _GatherElementsKernel(
    const T* input_data,
    const TArray<int64_t> axis_input_strides,
    const TArray<fast_divmod> axis_index_strides,
    const fast_divmod axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const fast_divmod output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    size_t index_element_size,
    T* output_data) {

  CUDA_LONG indices_index = threads_per_block * thread_worksize * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < thread_worksize; ++i) {
    if (indices_index < indices_size) {
      const int64_t input_batch_index = output_batch_size.div(indices_index);
      // Lookup the axis dim for input which is the input_block_index
      int64_t input_block_index = GetIndexValue(indices_data, index_element_size, indices_index);
      if (input_block_index < -axis_input_dim_value || input_block_index >= axis_input_dim_value) {
        // invalid index
        return;
      }

      if (input_block_index < 0) {
        input_block_index += axis_input_dim_value;
      }

      int32_t index_block_offset = axis_index_block_size.mod(indices_index);
      int32_t input_block_offset = 0;

      for (int32_t i = 0; i < axis_index_strides.Size() && index_block_offset > 0; ++i) {
        int dim;
        axis_index_strides[i].divmod(index_block_offset, dim, index_block_offset);
        input_block_offset += static_cast<int32_t>(axis_input_strides[i] * dim);
      }

      const T* input_block = input_data + input_batch_index * input_batch_size + input_block_index * axis_input_block_size;
      output_data[indices_index] = input_block[input_block_offset];

      indices_index += threads_per_block;
    }
  }
}

template<typename T>
void GatherElementsImpl(
    const T* input_data,
    const TArray<int64_t> axis_input_strides,
    const TArray<fast_divmod> axis_index_strides,
    const int64_t axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    size_t index_element_size,
    T* output_data) {
  if (indices_size > 0) {
    // Elements per thread
    dim3 block(threads_per_block);
    dim3 blocksPerGrid((static_cast<int>(indices_size + block.x * thread_worksize - 1) / (block.x * thread_worksize)));

    fast_divmod div_block(static_cast<int32_t>(axis_index_block_size));
    fast_divmod div_batch(static_cast<int32_t>(output_batch_size));

    _GatherElementsKernel<T><<<blocksPerGrid, block, 0>>>(
        input_data,
        axis_input_strides,
        axis_index_strides,
        div_block,
        axis_input_block_size,
        axis_input_dim_value,
        input_batch_size,
        div_batch,
        indices_data,
        indices_size,
        index_element_size,
        output_data);
  }
}

template
void GatherElementsImpl(
    const int8_t* input_data,
    const TArray<int64_t> axis_input_strides,
    const TArray<fast_divmod> axis_index_strides,
    const int64_t axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    size_t index_element_size,
    int8_t* output_data);

template
void GatherElementsImpl(
        const int16_t* input_data,
        const TArray<int64_t> axis_input_strides,
        const TArray<fast_divmod> axis_index_strides,
        const int64_t axis_index_block_size,
        const int64_t axis_input_block_size,
        const int64_t axis_input_dim_value,
        const int64_t input_batch_size,
        const int64_t output_batch_size,
        const void* indices_data,
        const int64_t indices_size,
        size_t index_element_size,
        int16_t* output_data);

template
void GatherElementsImpl(
    const int32_t* input_data,
    const TArray<int64_t> axis_input_strides,
    const TArray<fast_divmod> axis_index_strides,
    const int64_t axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    size_t index_element_size,
    int32_t* output_data);

template
void GatherElementsImpl(
    const int64_t* input_data,
    const TArray<int64_t> axis_input_strides,
    const TArray<fast_divmod> axis_index_strides,
    const int64_t axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    size_t index_element_size,
    int64_t* output_data);

}  // namespace cuda
}  // namespace onnxruntime
