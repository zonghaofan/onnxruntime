// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements.h"
#include "gather_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

Status GatherElements::ComputeInternal(OpKernelContext* context) const {
  // Process input data tensor
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();
  const auto& input_dims = input_shape.GetDims();
  const int64_t input_rank = static_cast<int64_t>(input_dims.size());

  // Process indices tensor
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto& indices_shape = indices_tensor->Shape();
  const int64_t indices_size = indices_shape.Size();

  // Handle negative axis if any
  const int64_t axis = static_cast<int64_t>(HandleNegativeAxis(axis_, input_rank));

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared checks)
  auto status = onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis);
  if (!status.IsOK())
    return status;

  // A number of axis_blocks
  const int64_t outer_dims_prod = input_shape.SizeToDimension(axis);
  // Input batch size addressable by axis
  const int64_t input_batch_size = input_shape.SizeFromDimension(axis);
  // Block size under the axis
  const int64_t axis_block_size = input_shape.SizeFromDimension(axis + 1);
  // Number of output blocks (output block is axis_block_size)
  const int64_t output_batch_size = indices_size * axis_block_size;

  // create output tensor
  auto* output_tensor = context->Output(0, indices_shape);

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  const size_t element_size = input_tensor->DataType()->Size();

  if (indices_tensor->IsDataType<int32_t>()) {
    const int32_t* indices_data = indices_tensor->template Data<int32_t>();
    GatherElementsImpl<int32_t>(
        input_tensor->DataRaw(),
        outer_dims_prod,
        axis_block_size,
        input_batch_size,
        output_batch_size,
        indices_data,
        indices_size,
        output_tensor->MutableDataRaw(),
        element_size);
    return Status::OK();
  } else if (indices_tensor->IsDataType<int64_t>()) {
    const int64_t* indices_data = indices_tensor->template Data<int64_t>();
    GatherElementsImpl<int64_t>(
        input_tensor->DataRaw(),
        outer_dims_prod,
        axis_block_size,
        input_batch_size,
        output_batch_size,
        indices_data,
        indices_size,
        output_tensor->MutableDataRaw(),
        element_size);
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherElements op: Type of 'indices' must be int32 or int64");
  }
}

}  // namespace cuda
}  // namespace onnxruntime
