// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_elements_impl.h"

#include <thread>

namespace onnxruntime {
namespace cuda {

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
template <typename T, typename Tin>
void _GatherElementsKernel(
    CUDA_LONG blockIdx,
    CUDA_LONG blockDim,
    CUDA_LONG threadIdx,
    const T* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_input_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const int64_t axis_index_block_size,
    const Tin* indices_data,
    const int64_t indices_size,
    T* output_data) {

  // Total threads we are trying to utilize is indices_size * index_block_size
  // So each thread will be hoping through the grid and handle its own offset there
  // which is threadIdx.x. 
  // Essentially, blockIdx.x * blockDim.x == outer_dims_prod
  // threadIdx.x is within the range of [0, indices_size * axis_index_block_size)
  //const auto input_batch = blockIdx.x * blockDim.x;
  //const auto start_item = input_batch + threadIdx.x;
  const int64_t input_batch = blockIdx * blockDim;
  const int64_t start_item = input_batch + threadIdx;
  const auto processing_step = indices_size * axis_index_block_size * axis_input_block_size;

  const auto items_to_process = outer_dims_prod * processing_step;
  for (auto item_num = start_item; start_item < items_to_process; item_num += processing_step) {
    // input_batch and output_batch indices are the same
    const auto input_batch_index = item_num / processing_step;
    const auto output_batch_index = input_batch_index;
    // output_block_index is the same as the index offset.
    const auto output_block_index = (item_num / axis_index_block_size / axis_input_block_size) % indices_size;
    // Substitute the input block_index. input_block_index is the block of data identified
    // by the index table
    const auto input_block_index = indices_data[output_block_index];
    const auto input_block_offset = item_num % axis_input_block_size;
    const auto output_block_offset = item_num % axis_index_block_size;

    const T* input_block = input_data + input_batch_index * input_batch_size + input_block_index * axis_input_block_size;
    T* output_block = output_data  + output_batch_index * output_batch_size + output_block_index * axis_index_block_size;
    output_block[output_block_offset] = input_block[input_block_offset];
  }
}

template<class T>
inline void StartThreads(int64_t outer_dims_prod, int64_t indices_size, int64_t axis_input_block_size) {
  const int blocksPerGrid = static_cast<int>(std::min<int64_t>(outer_dims_prod, GridDim::maxThreadBlocks));
}

template <typename Tin>
void GatherElementsImpl(
    const void* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_input_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const int64_t axis_index_block_size,
    const Tin* indices_data,
    const int64_t indices_size,
    void* output_data,
    size_t element_size) {

  if (indices_size > 0) {

    // Original input size is outer_dims_prod * axis_dim * axis_block_size
    // In the output we subst it to outer_dims_prod * indices_size * axis_block_size
    // Each thread processes a block of indices_size * axis_block_size and the number of 
    // blocks is outer_dims_prod
    //const int blocksPerGrid = static_cast<int>(std::min<int64_t>(outer_dims_prod, GridDim::maxThreadBlocks));
    //const auto num_threads_per_block = static_cast<int>(std::min<int64_t>((indices_size * axis_index_block_size), GridDim::maxThreadsPerBlock));

    const int blocksPerGrid = static_cast<int>(outer_dims_prod);
    const auto num_threads_per_block = static_cast<int>(indices_size * axis_index_block_size);

    std::vector<std::thread> threads;
    threads.reserve(blocksPerGrid * num_threads_per_block);

    switch (element_size) {
      {
        case sizeof(int8_t):
          auto int8_fn = [=](CUDA_LONG blockIdx,
                             CUDA_LONG blockDim,
                             CUDA_LONG threadIdx) {
            _GatherElementsKernel<int8_t, Tin>(
                blockIdx, blockDim, threadIdx,
                reinterpret_cast<const int8_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
                axis_index_block_size, indices_data, indices_size, reinterpret_cast<int8_t*>(output_data));
          };
          for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
            for (int tid = 0; tid < num_threads_per_block; ++tid)
              threads.push_back(std::thread(int8_fn, blockId, 1, tid));
          }
          //_GatherElementsKernel<int8_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
          //    reinterpret_cast<const int8_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
          //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int8_t*>(output_data));
      }
        break;

      case sizeof(int16_t): {
          auto int16_fn = [=](CUDA_LONG blockIdx,
                              CUDA_LONG blockDim,
                              CUDA_LONG threadIdx) {
            _GatherElementsKernel<int16_t, Tin>(
                blockIdx, blockDim, threadIdx,
                reinterpret_cast<const int16_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
                axis_index_block_size, indices_data, indices_size, reinterpret_cast<int16_t*>(output_data));
          };
        for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
            for (int tid = 0; tid < num_threads_per_block; ++tid)
            threads.push_back(std::thread(int16_fn, blockId, 1, tid));
        }
          //_GatherElementsKernel<int16_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
          //    reinterpret_cast<const int16_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
          //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int16_t*>(output_data));
        }
        break;

      case sizeof(int32_t): {
          auto int32_fn = [=](CUDA_LONG blockIdx,
                              CUDA_LONG blockDim,
                              CUDA_LONG threadIdx) {
            _GatherElementsKernel<int32_t, Tin>(
                blockIdx, blockDim, threadIdx,
                reinterpret_cast<const int32_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
                axis_index_block_size, indices_data, indices_size, reinterpret_cast<int32_t*>(output_data));
          };
        for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
            for (int tid = 0; tid < num_threads_per_block; ++tid)
              threads.push_back(std::thread(int32_fn, blockId, 1, tid));
        }
          //_GatherElementsKernel<int32_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
          //    reinterpret_cast<const int32_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
          //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int32_t*>(output_data));
        }
        break;

      case sizeof(int64_t): {
          auto int64_fn = [=](CUDA_LONG blockIdx,
                              CUDA_LONG blockDim,
                              CUDA_LONG threadIdx) {
            _GatherElementsKernel<int64_t, Tin>(
                blockIdx, blockDim, threadIdx,
                reinterpret_cast<const int64_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
                axis_index_block_size, indices_data, indices_size, reinterpret_cast<int64_t*>(output_data));
          };
        for (int blockId = 0; blockId < blocksPerGrid; ++blockId) {
            for (int tid = 0; tid < num_threads_per_block; ++tid)
              threads.push_back(std::thread(int64_fn, blockId, 1, tid));
        }
          //_GatherElementsKernel<int64_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
          //    reinterpret_cast<const int64_t*>(input_data), outer_dims_prod, axis_input_block_size, input_batch_size, output_batch_size,
          //    axis_index_block_size, indices_data, indices_size, reinterpret_cast<int64_t*>(output_data));
        }
        break;

      // should not reach here as we validate if the all relevant types are supported in the Compute method 
      default:
        ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
    }

    for (auto& th : threads) {
      th.join();
    }
  }
}

template void GatherElementsImpl<int32_t>(
    const void* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_input_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const int64_t axis_index_block_size,
    const int32_t* indices_data,
    const int64_t indices_size,
    void* output_data,
    size_t element_size);

template void GatherElementsImpl<int64_t>(
    const void* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_input_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const int64_t axis_index_block_size,
    const int64_t* indices_data,
    const int64_t indices_size,
    void* output_data,
    size_t element_size);

}  // namespace cuda
}  // namespace onnxruntime

