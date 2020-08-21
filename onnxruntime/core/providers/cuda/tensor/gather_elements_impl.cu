// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_elements_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
__global__ void _GatherElementsKernel(
    const T* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const Tin* indices_data,
    const int64_t indices_size,
    T* output_data) {

  // Total threads we are trying to utilize is indices_size * axis_block_size
  // So each thread will be hoping through the grid and handle its own offset there
  // which is threadIdx.x
  const auto start_item = blockIdx.x * blockDim.x + threadIdx.x;
  // Number of items to output
  const auto items_to_process = outer_dims_prod * indices_size * axis_block_size;
  for (auto item_num = start_item; start_item < items_to_process; item_num += blockIdx.x * blockDim.x) {
    // Batch index both for input and output although their sizes are different
    // the addressable number is the same as we use the same number of indices
    const auto batch_index = item_num / axis_block_size / indices_size;
    // output_batch_index is the same as indices and output have same dims
    const auto output_block_index = item_num / axis_block_size % indices_size;
    // Substitute the input block_index
    const auto  input_block_index = indices_data[output_block_index];
    const auto block_offset = item_num % axis_block_size;

    const T* input_block = input_data + batch_index * input_batch_size + input_block_index * axis_block_size;
    T* output_block = output_data + batch_index * output_batch_size + output_block_index * axis_block_size;
    output_block[block_offset] = input_block[block_offset];
  }
}

template <typename Tin>
void GatherElementsImpl(
    const void* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const Tin* indices_data,
    const int64_t indices_size,
    void* output_data,
    size_t element_size) {

  if (indices_size > 0) {

    // Original input size is outer_dims_prod * axis_dim * axis_block_size
    // In the output we subst it to outer_dims_prod * indices_size * axis_block_size
    // Each thread processes a block of indices_size * axis_block_size and the number of 
    // blocks is outer_dims_prod
    const int blocksPerGrid = static_cast<int>(outer_dims_prod);
    const auto num_threads_per_block = static_cast<int>(std::min<int64_t>((indices_size * axis_block_size), GridDim::maxThreadsPerBlock));

    switch (element_size) {
      case sizeof(int8_t):
        _GatherElementsKernel<int8_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
            reinterpret_cast<const int8_t*>(input_data), outer_dims_prod, axis_block_size, input_batch_size, output_batch_size,
            indices_data, indices_size, reinterpret_cast<int8_t*>(output_data));
        break;

      case sizeof(int16_t):
        _GatherElementsKernel<int16_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
            reinterpret_cast<const int16_t*>(input_data), outer_dims_prod, axis_block_size, input_batch_size, output_batch_size,
            indices_data, indices_size, reinterpret_cast<int16_t*>(output_data));
        break;

      case sizeof(int32_t):
        _GatherElementsKernel<int32_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
            reinterpret_cast<const int32_t*>(input_data), outer_dims_prod, axis_block_size, input_batch_size, output_batch_size,
            indices_data, indices_size, reinterpret_cast<int32_t*>(output_data));
        break;

      case sizeof(int64_t):
        _GatherElementsKernel<int64_t, Tin><<<blocksPerGrid, num_threads_per_block, 0>>>(
            reinterpret_cast<const int64_t*>(input_data), outer_dims_prod, axis_block_size, input_batch_size, output_batch_size,
            indices_data, indices_size, reinterpret_cast<int64_t*>(output_data));
        break;

      // should not reach here as we validate if the all relevant types are supported in the Compute method 
      default:
        ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
    }
  }
}

template void GatherElementsImpl<int32_t>(
    const void* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const int32_t* indices_data,
    const int64_t indices_size,
    void* output_data,
    size_t element_size);

template void GatherElementsImpl<int64_t>(
    const void* input_data,
    const int64_t outer_dims_prod,
    const int64_t axis_block_size,
    const int64_t input_batch_size,
    const int64_t output_batch_size,
    const int64_t* indices_data,
    const int64_t indices_size,
    void* output_data,
    size_t element_size);

}  // namespace cuda
}  // namespace onnxruntime

