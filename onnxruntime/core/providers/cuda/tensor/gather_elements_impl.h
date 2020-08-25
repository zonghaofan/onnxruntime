// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

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
    size_t element_size);

}  // namespace cuda
}  // namespace onnxruntime
