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

  const int64_t indices_size = indices_shape.Size();
  // Input batch size addressable by axis
  const int64_t input_batch_size = input_shape.SizeFromDimension(axis);
  const int64_t output_batch_size = indices_shape.SizeFromDimension(axis);
  const int64_t axis_input_dim_value = input_shape[axis];
  // Block size under the axis
  const int64_t axis_index_block_size = indices_shape.SizeFromDimension(axis + 1);
  const int64_t axis_input_block_size = input_shape.SizeFromDimension(axis + 1);

  TensorPitches input_pitches(input_shape.GetDims());
  TensorPitches indices_pitches(indices_shape.GetDims());
  // We only pitches/strides after axis
  const int64_t pitches_size = input_pitches.size();
  const int32_t axis_one = static_cast<int32_t>(axis + 1);
  int64_t strides_size = 0;
  if (axis_one < pitches_size) {
    strides_size = pitches_size - axis_one;
  }

  TArray<int64_t> input_strides(static_cast<int32_t>(strides_size));
  TArray<fast_divmod> indices_strides(static_cast<int32_t>(strides_size));
  for (auto i = axis_one; i < pitches_size; ++i) {
    input_strides[i - axis_one] = input_pitches[i];
    indices_strides[i - axis_one] = fast_divmod(static_cast<int32_t>(indices_pitches[i]));
  }

  // create output tensor
  auto* output_tensor = context->Output(0, indices_shape);

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  // CPU algo test ONLY!
  const void* input_data = input_tensor->DataRaw();
  const void* indices_data = indices_tensor->DataRaw();
  void* output_data = output_tensor->MutableDataRaw();
  //std::unique_ptr<char[]> input_cpu(new char[input_tensor->SizeInBytes()]);
  //CUDA_RETURN_IF_ERROR(cudaMemcpy(input_cpu.get(), input_data, input_tensor->SizeInBytes(),
  //                                  cudaMemcpyDeviceToHost));
  //std::unique_ptr<char[]> index_cpu(new char[indices_tensor->SizeInBytes()]);
  //CUDA_RETURN_IF_ERROR(cudaMemcpy(index_cpu.get(), indices_tensor->DataRaw(), indices_tensor->SizeInBytes(),
  //                                cudaMemcpyDeviceToHost));
  //std::unique_ptr<char[]> output_cpu(new char[output_tensor->SizeInBytes()]);

  const size_t element_size = input_tensor->DataType()->Size();
  const size_t index_element_size = indices_tensor->DataType()->Size();

  if (indices_tensor->IsDataType<int32_t>() ||
      indices_tensor->IsDataType<int64_t>()) {
    switch (element_size) {
      case sizeof(int8_t):
      GatherElementsImpl<int8_t>(
          reinterpret_cast<const int8_t*>(input_data),
          input_strides,
          indices_strides,
          axis_index_block_size,
          axis_input_block_size,
          axis_input_dim_value,
          input_batch_size,
          output_batch_size,
          indices_data,
          indices_size,
          index_element_size,
          reinterpret_cast<int8_t*>(output_data));
        break;
      case sizeof(int16_t):
      GatherElementsImpl<int16_t>(
          reinterpret_cast<const int16_t*>(input_data),
          input_strides,
          indices_strides,
          axis_index_block_size,
          axis_input_block_size,
          axis_input_dim_value,
          input_batch_size,
          output_batch_size,
          indices_data,
          indices_size,
          index_element_size,
          reinterpret_cast<int16_t*>(output_data));
      break;

    case sizeof(int32_t):
        GatherElementsImpl<int32_t>(
            reinterpret_cast<const int32_t*>(input_data),
            input_strides,
            indices_strides,
            axis_index_block_size,
            axis_input_block_size,
            axis_input_dim_value,
            input_batch_size,
            output_batch_size,
            indices_data,
            indices_size,
            index_element_size,
            reinterpret_cast<int32_t*>(output_data));
        break;

    case sizeof(int64_t):
      GatherElementsImpl<int64_t>(
          reinterpret_cast<const int64_t*>(input_data),
          input_strides,
          indices_strides,
          axis_index_block_size,
          axis_input_block_size,
          axis_input_dim_value,
          input_batch_size,
          output_batch_size,
          indices_data,
          indices_size,
          index_element_size,
          reinterpret_cast<int64_t*>(output_data));
      break;

      // Should not reach here
      default:
        ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
    }

     //CPU algo test ONLY!
     //Copy data back to GPU for test
    //CUDA_RETURN_IF_ERROR(cudaMemcpy(output_data, output_cpu.get(), output_tensor->SizeInBytes(),
    //                                  cudaMemcpyHostToDevice));
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherElements op: Type of 'indices' must be int32 or int64");
  }
}

}  // namespace cuda
}  // namespace onnxruntime
