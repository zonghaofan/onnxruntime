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
/// Each thread executing this kernel will process an element within a input_batch.
/// That element is determined by the threadIdx. The thread will hop blocks to copy
/// that specific element. Essentially, we want to have as many threads as index entries
/// and as many thread blocks as input_batches which is the outer_dims_prod.
///
/// @param input_data - base ptr to input_data
/// @param outer_dims_prod - product of dims before(axis) off index_shape, as we drive our
///   thread blocks off index_shape because:
///     1. we need to cover all indexes
///     2. input and indexes have the same rank
///     3. no dims of index_shape or index value is guaranteed to be out of bounds with input_data
/// @param axis_input_block_size - product of dims from (axis + 1) in the input_data
/// @param input_batch_size - product of dims from (axis) in the input_data
/// @param output_batch_size = product
/// @param output_batch_size - product of dims from(axis) in index_data
/// @param indices_data - base ptr to indices
/// @param axis_index_block_size - product of dims from(axis + 1) from index_shape
/// @param indices_size - number of indices
/// @param output_data base ptr to output, same size as indices
/// </summary>
template <typename T>
__host__ __device__ void _GatherElementsKernel(
    CUDA_LONG blockIdx,
    CUDA_LONG blockDim,
    CUDA_LONG threadIdx,
    const T* input_data,
    const int64_t axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    const size_t index_element_size,
    T* output_data) {

  // CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);
  CUDA_LONG indices_index = blockDim * blockIdx + threadIdx;
  if (indices_index >= indices_size) {
    return;
  }

  printf("indices_index: %d\n", indices_index);

  // We calculate the batch size for indices and use it for input
  // bc outer_dims_prod from indices dictate the batch we use
  const int64_t input_batch_index = indices_index / output_batch_size;
  // Lookup the axis dim for input which is the input_block_index
  int64_t input_block_index = GetIndexValue(indices_data, index_element_size, indices_index);
  if (input_block_index < -axis_input_dim_value || input_block_index >= axis_input_dim_value) {
    // invalid index
    return;
  }

  if (input_block_index < 0) {
    input_block_index += axis_input_dim_value;
  }

  // This calculates offset into an index block under axis
  const int64_t input_block_offset = indices_index % axis_index_block_size;

  printf("\tinput_batch_index: %lld, input_block_index: %lld, input_block_offset: %lld\n",
          input_batch_index, input_block_index, input_block_offset);

  const T* input_block = input_data + input_batch_index * input_batch_size + input_block_index * axis_input_block_size;
  output_data[indices_index] = input_block[input_block_offset];
}

void GatherElementsImpl(
    const void* input_data,
    const int64_t axis_index_block_size,
    const int64_t axis_input_block_size,
    const int64_t axis_input_dim_value,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const void* indices_data,
    const int64_t indices_size,
    size_t index_element_size,
    void* output_data,
    size_t element_size) {
  if (indices_size > 0) {
    //const int blocksPerGrid = static_cast<int>(std::min<int64_t>(outer_dims_prod, GridDim::maxThreadBlocks));
    //const auto num_threads_per_block = static_cast<int>(std::min<int64_t>((indices_size * axis_index_block_size), GridDim::maxThreadsPerBlock));

    printf("input_batch_size: %lld, axis_index_block_size: %lld, axis_input_block_size: %lld\n", 
      input_batch_size, axis_index_block_size, axis_input_block_size);

    const auto num_threads_per_block = std::min<int64_t>(indices_size, GridDim::maxThreadsPerBlock);
    const int blocksPerGrid = static_cast<int>((indices_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

    switch (element_size) {
      {
        case sizeof(int8_t):
          for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {

            for (int threadIdx = 0; threadIdx < num_threads_per_block; ++threadIdx)
              _GatherElementsKernel<int8_t>(
                  blockId, 1, threadIdx,
                  reinterpret_cast<const int8_t*>(input_data), axis_index_block_size, axis_input_block_size, axis_input_dim_value,
                  input_batch_size, output_batch_size, indices_data, indices_size, index_element_size, reinterpret_cast<int8_t*>(output_data));
          }
          //_GatherElementsKernel<int8_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
          //    reinterpret_cast<const int8_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
          //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int8_t*>(output_data));
      }
      break;

      case sizeof(int16_t): {
        for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
          for (int threadIdx = 0; threadIdx < num_threads_per_block; ++threadIdx)
            _GatherElementsKernel<int16_t>(
                blockId, 1, threadIdx,
                reinterpret_cast<const int16_t*>(input_data), axis_index_block_size, axis_input_block_size, axis_input_dim_value,
                input_batch_size, output_batch_size, indices_data, indices_size, index_element_size, reinterpret_cast<int16_t*>(output_data));
        }
        //_GatherElementsKernel<int16_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
        //    reinterpret_cast<const int16_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
        //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int16_t*>(output_data));
      } break;

      case sizeof(int32_t): {
        for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
          for (int threadIdx = 0; threadIdx < num_threads_per_block; ++threadIdx)
            _GatherElementsKernel<int32_t>(
                blockId, 1, threadIdx,
                reinterpret_cast<const int32_t*>(input_data), axis_index_block_size, axis_input_block_size, axis_input_dim_value,
                input_batch_size, output_batch_size, indices_data, indices_size, index_element_size, reinterpret_cast<int32_t*>(output_data));
        }

        //_GatherElementsKernel<int32_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
        //    reinterpret_cast<const int32_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
        //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int32_t*>(output_data));
      } break;

      case sizeof(int64_t): {
        for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
          for (int threadIdx = 0; threadIdx < num_threads_per_block; ++threadIdx)
            _GatherElementsKernel<int64_t>(
                blockId, 1, threadIdx,
                reinterpret_cast<const int64_t*>(input_data), axis_index_block_size, axis_input_block_size, axis_input_dim_value,
                input_batch_size, output_batch_size, indices_data, indices_size, index_element_size, reinterpret_cast<int64_t*>(output_data));
        }
        //_GatherElementsKernel<int64_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
        //    reinterpret_cast<const int64_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
        //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int64_t*>(output_data));
      } break;

      // should not reach here as we validate if the all relevant types are supported in the Compute method
      default:
        ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
    }
  }
}

}  // namespace cuda
}  // namespace onnxruntime
