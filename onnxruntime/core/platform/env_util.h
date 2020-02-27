// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

#include "core/common/common.h"

namespace onnxruntime {
namespace detail {

/**
 * Loops over a sequence in batches and calls the processing function on each
 * batch.
 *
 * TProcessFn signature:
 * Status TProcessFn(
 *     // inputs
 *     size_t total_num_processed, TBatchElementCount num_to_process,
 *     // output
 *     TBatchElementCount& num_processed)
 *
 *
 * @param total_num_to_process The total number of elements to be processed.
 * @param max_num_per_batch The maximum number of elements to be processed in
 *                          one batch.
 * @param process_fn The function to process the elements.
 * @return The status of the operation.
 */
template <typename TProcessFn, typename TBatchElementCount>
Status ProcessInBatches(
    size_t total_num_to_process, TBatchElementCount max_num_per_batch, TProcessFn process_fn) {
  size_t total_num_processed = 0;
  while (total_num_processed < total_num_to_process) {
    const TBatchElementCount num_to_process = gsl::narrow_cast<TBatchElementCount>(std::min<size_t>(
        max_num_per_batch, total_num_to_process - total_num_processed));
    TBatchElementCount num_processed{};
    ORT_RETURN_IF_ERROR(process_fn(total_num_processed, num_to_process, num_processed));
    ORT_RETURN_IF(num_processed == 0, "Failed to process any elements.")
    total_num_processed += num_processed;
  }
  return Status::OK();
}

}  // namespace detail
}  // namespace onnxruntime
