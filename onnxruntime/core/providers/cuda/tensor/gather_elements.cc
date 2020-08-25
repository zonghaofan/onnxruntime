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

  // Handle negative axis if any
  const int64_t axis = static_cast<int64_t>(HandleNegativeAxis(axis_, input_rank));

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared checks)
  auto status = onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis);
  if (!status.IsOK())
    return status;

  const int64_t input_size = input_shape.Size();
  const int64_t indices_size = indices_shape.Size();
  // We iterate using index outer_dims. It is guaranteed not to exceed input_data dims.
  const int64_t input_outer_dims_prod = input_shape.SizeToDimension(axis);
  // Input batch size addressable by axis
  const int64_t input_batch_size = input_shape.SizeFromDimension(axis);
  const int64_t output_batch_size = indices_shape.SizeFromDimension(axis);
  const int64_t axis_input_dim_value = input_shape[axis];
  // Block size under the axis
  const int64_t axis_input_block_size = input_shape.SizeFromDimension(axis + 1);
  const int64_t axis_index_block_size = indices_shape.SizeFromDimension(axis + 1);

  // create output tensor
  auto* output_tensor = context->Output(0, indices_shape);

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  // CPU algo test ONLY!
  const void* input_data = input_tensor->DataRaw();
  void* output_data = output_tensor->MutableDataRaw();
  std::unique_ptr<char[]> input_cpu(new char[input_tensor->SizeInBytes()]);
  CUDA_RETURN_IF_ERROR(cudaMemcpy(input_cpu.get(), input_data, input_tensor->SizeInBytes(),
                                    cudaMemcpyDeviceToHost));
  std::unique_ptr<char[]> index_cpu(new char[indices_tensor->SizeInBytes()]);
  CUDA_RETURN_IF_ERROR(cudaMemcpy(index_cpu.get(), indices_tensor->DataRaw(), indices_tensor->SizeInBytes(),
                                  cudaMemcpyDeviceToHost));

    // Create output on CPU
  std::unique_ptr<char[]> output_cpu(new char[output_tensor->SizeInBytes()]);

  const size_t element_size = input_tensor->DataType()->Size();
  const size_t index_element_size = indices_tensor->DataType()->Size();

  if (indices_tensor->IsDataType<int32_t>() ||
      indices_tensor->IsDataType<int64_t>()) {
    GatherElementsImpl(
        input_cpu.get(),
        axis_index_block_size,
        axis_input_block_size,
        axis_input_dim_value,
        input_batch_size,
        output_batch_size,
        index_cpu.get(),
        indices_size,
        index_element_size,
        output_cpu.get(),
        element_size);

    // CPU algo test ONLY!
    // Copy data back to GPU for test
    CUDA_RETURN_IF_ERROR(cudaMemcpy(output_data, output_cpu.get(), output_tensor->SizeInBytes(),
                                      cudaMemcpyHostToDevice));
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherElements op: Type of 'indices' must be int32 or int64");
  }
}

}  // namespace cuda
}  // namespace onnxruntime
