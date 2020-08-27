// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_elements_impl.h"

#include <thread>

namespace onnxruntime {
namespace cuda {

__host__ __device__ inline int64_t GetIndexValue(const void* index_data, size_t index_element_size, size_t offset) {
  switch (index_element_size) {
    case sizeof(int32_t):
      return *(reinterpret_cast<const int32_t*>(index_data) + offset);
      break;
    case sizeof(int64_t):
      return *(reinterpret_cast<const int64_t*>(index_data) + offset);
      break;
  }
  // This should cause abort
  assert(false);
  return std::numeric_limits<int64_t>::max();
}

/// <summary>
/// This is a CUDA kernel that performs GatherElements.
/// Each thread executing this kernel will process an index value.
/// Since index values are applied at axis, input_batch index i.e. a chunk of data
/// before axis would be the same for both input and index although their sizes are different.
/// We substitute the axis dim value in the input for the looked up index value (input_block_index)
//  We then calculate the input block offset from index block offset and copy the value.
///
/// @param input_data - base ptr to input_data
/// @param axis_input_strides - input pitches/strides for dims after axis
/// @param axis_index_strides - indices pitches/strides for dims after axis
/// @param axis_index_block_size - index block size after axis
/// @param axis_input_block_size - input block size after axis
/// @param input_batch_size - input data size from(axis)
/// @param output_batch_size - output batch size from(axis)
/// @param indices_data - base ptr to indices
/// @param indices_size - number of indices
/// @param index_element_size - size of 
/// @param output_data base ptr to output, same size as indices
/// </summary>
//template <typename T>
//__global__ void _GatherElementsKernel(
template <typename Func>
void _GatherElementsKernel(
    CUDA_LONG blockIdx,
    CUDA_LONG blockDim,
    CUDA_LONG threadIdx,
    const int64_t indices_size,
    Func f) {

  // CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);
  CUDA_LONG indices_index = blockDim * blockIdx + threadIdx;
  if (indices_index >= indices_size) {
    return;
  }

  f(indices_index);
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
    printf("input_batch_size: %lld, axis_index_block_size: %lld, axis_input_block_size: %lld\n",
           input_batch_size, axis_index_block_size, axis_input_block_size);

    fast_divmod div_batch(output_batch_size);
    fast_divmod div_block(axis_index_block_size);

    auto gather = [=] __host__ __device__(int32_t indices_index) {
      const int64_t input_batch_index = div_batch.div(indices_index);
      // Lookup the axis dim for input which is the input_block_index
      int64_t input_block_index = GetIndexValue(indices_data, index_element_size, indices_index);
      if (input_block_index < -axis_input_dim_value || input_block_index >= axis_input_dim_value) {
        // invalid index
        return;
      }

      if (input_block_index < 0) {
        input_block_index += axis_input_dim_value;
      }

      int32_t index_block_offset = div_block.mod(indices_index);
      int32_t input_block_offset = 0;

      printf("\t\tindex_block_offset: %d, Axis input dims: ", index_block_offset);
      for (int32_t i = 0; i < axis_index_strides.Size() && index_block_offset > 0; ++i) {
        int dim;
        axis_index_strides[i].divmod(index_block_offset, dim, index_block_offset);
        input_block_offset += static_cast<int32_t>(axis_input_strides[i] * dim);
        printf("%d ", dim);
      }

      printf("\n\tinput_batch_index: %lld, input_block_index: %lld, input_block_offset: %d\n",
             input_batch_index, input_block_index, input_block_offset);

      const T* input_block = input_data + input_batch_index * input_batch_size + input_block_index * axis_input_block_size;
      output_data[indices_index] = input_block[input_block_offset];
    };

    const int num_threads_per_block = static_cast<int>(std::min<int64_t>(indices_size, GridDim::maxThreadsPerBlock));
    const int blocksPerGrid = static_cast<int>((indices_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

    for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
      for (int threadIdx = 0; threadIdx < num_threads_per_block; ++threadIdx)
        _GatherElementsKernel(
            blockId, 1, threadIdx,
            indices_size,
            gather);
    }

    //_GatherElementsKernel<int8_t><<<blocksPerGrid, num_threads_per_block, 0>>>(
    //        reinterpret_cast<const int8_t*>(input_data), axis_input_strides, axis_index_strides,
    //         axis_index_block_size, axis_input_block_size, axis_input_dim_value,
    //        input_batch_size, output_batch_size, indices_data, indices_size, index_element_size, reinterpret_cast<int8_t*>(output_data));
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
