// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_flatbuffers_utils.h"
#include "core/framework/tensorprotoutils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::experimental;

namespace onnxruntime {
namespace experimental {
namespace utils {

static flatbuffers::Offset<fbs::Dimension> GetTensorDimensionOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const TensorShapeProto_Dimension& tensor_shape_dim) {
  auto denotation = builder.CreateString(tensor_shape_dim.denotation());
  flatbuffers::Offset<fbs::DimensionValue> dim_val;
  if (tensor_shape_dim.has_dim_param()) {
    dim_val = fbs::CreateDimensionValueDirect(builder, fbs::DimensionValueType_PARAM, 0, tensor_shape_dim.dim_param().c_str());
  } else if (tensor_shape_dim.has_dim_value()) {
    dim_val = fbs::CreateDimensionValueDirect(builder, fbs::DimensionValueType_VALUE, tensor_shape_dim.dim_value());
  } else {
    dim_val = fbs::CreateDimensionValueDirect(builder);
  }

  return fbs::CreateDimension(builder, dim_val, denotation);
}

static Status GetTensorShapeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      const TensorShapeProto& tensor_shape_proto,
                                      flatbuffers::Offset<fbs::Shape>& fbs_shape) {
  std::vector<flatbuffers::Offset<fbs::Dimension>> dim;
  dim.reserve(tensor_shape_proto.dim_size());
  for (const auto& d : tensor_shape_proto.dim()) {
    auto fbs_d = GetTensorDimensionOrtFormat(builder, d);
    dim.push_back(fbs_d);
  }
  fbs_shape = fbs::CreateShapeDirect(builder, &dim);
  return Status::OK();
}

static Status GetTensorTypeAndShapeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                             const TypeProto_Tensor& tensor_type_proto,
                                             flatbuffers::Offset<fbs::TensorTypeAndShape>& fbs_tensor_type) {
  flatbuffers::Offset<fbs::Shape> shape;
  ORT_RETURN_IF_ERROR(GetTensorShapeOrtFormat(builder, tensor_type_proto.shape(), shape));
  fbs_tensor_type = fbs::CreateTensorTypeAndShape(
      builder, static_cast<fbs::TensorDataType>(tensor_type_proto.elem_type()), shape);
  return Status::OK();
}

static Status GetTypeInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                   const TypeProto& type_proto,
                                   flatbuffers::Offset<fbs::TypeInfo>& fbs_type_info) {
  auto denotation = builder.CreateString(type_proto.denotation());
  auto value_type = fbs::TypeInfoValue_tensor_type;
  flatbuffers::Offset<void> value;
  if (type_proto.has_tensor_type()) {
    value_type = fbs::TypeInfoValue_tensor_type;
    flatbuffers::Offset<fbs::TensorTypeAndShape> tensor_type;
    ORT_RETURN_IF_ERROR(
        GetTensorTypeAndShapeOrtFormat(builder, type_proto.tensor_type(), tensor_type));
    value = tensor_type.Union();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "We only support tensor type for now");
  }

  fbs::TypeInfoBuilder tb(builder);
  tb.add_denotation(denotation);
  tb.add_value_type(value_type);
  tb.add_value(value);
  fbs_type_info = tb.Finish();
  return Status::OK();
}

Status GetValueInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                             const ValueInfoProto& value_info_proto,
                             flatbuffers::Offset<fbs::ValueInfo>& fbs_value_info) {
  auto name = builder.CreateString(value_info_proto.name());
  auto doc_string = builder.CreateString(value_info_proto.doc_string());
  flatbuffers::Offset<fbs::TypeInfo> type_info;
  if (value_info_proto.has_type()) {
    ORT_RETURN_IF_ERROR(
        GetTypeInfoOrtFormat(builder, value_info_proto.type(), type_info));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "value_info_proto has no type");
  }

  fbs::ValueInfoBuilder vb(builder);
  vb.add_name(name);
  vb.add_doc_string(doc_string);
  vb.add_type(type_info);
  fbs_value_info = vb.Finish();
  return Status::OK();
}

Status GetInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                               const TensorProto& initializer,
                               flatbuffers::Offset<fbs::Tensor>& fbs_tensor) {
  auto name = builder.CreateString(initializer.name());
  auto doc_string = builder.CreateString(initializer.doc_string());
  std::vector<int64_t> dims_data(initializer.dims().size());
  std::copy(initializer.dims().cbegin(), initializer.dims().cend(), dims_data.begin());
  auto dims = builder.CreateVector(dims_data);
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> string_data;
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> raw_data;

  auto src_type = initializer.data_type();
  bool has_string_data = src_type == ONNX_NAMESPACE::TensorProto_DataType_STRING;
  if (has_string_data) {
    std::vector<std::string> string_data_vec(initializer.string_data().size());
    std::copy(initializer.string_data().cbegin(), initializer.string_data().cend(), string_data_vec.begin());
    string_data = builder.CreateVectorOfStrings(string_data_vec);
  } else {
    std::unique_ptr<uint8_t[]> unpacked_tensor;
    size_t tensor_byte_size;
    ORT_RETURN_IF_ERROR(
        onnxruntime::utils::UnpackInitializerData(initializer, unpacked_tensor, tensor_byte_size));
    raw_data = builder.CreateVector(unpacked_tensor.get(), tensor_byte_size);
  }

  fbs::TensorBuilder tb(builder);
  tb.add_name(name);
  tb.add_doc_string(doc_string);
  tb.add_dims(dims);
  tb.add_data_type(static_cast<fbs::TensorDataType>(src_type));
  if (has_string_data)
    tb.add_string_data(string_data);
  else
    tb.add_raw_data(raw_data);
  fbs_tensor = tb.Finish();
  return Status::OK();
}

#define GET_FBS_ATTR(BUILDER, TYPE, DATA_NAME, DATA) \
  fbs::AttributeBuilder attr_builder(BUILDER);       \
  attr_builder.add_name(name);                       \
  attr_builder.add_doc_string(doc_string);           \
  attr_builder.add_type(TYPE);                       \
  attr_builder.add_##DATA_NAME(DATA);                \
  fbs_attr = attr_builder.Finish();                  \
  return Status::OK();

#define GET_DATA_VEC(TYPE, NAME, SRC_DATA) \
  std::vector<TYPE> NAME(SRC_DATA.size()); \
  std::copy(SRC_DATA.cbegin(), SRC_DATA.cend(), NAME.begin());

Status GetAttributeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                             const AttributeProto& attr_proto,
                             flatbuffers::Offset<fbs::Attribute>& fbs_attr,
                             const onnxruntime::Graph* graph) {
  auto name = builder.CreateString(attr_proto.name());
  auto doc_string = builder.CreateString(attr_proto.doc_string());
  auto type = static_cast<fbs::AttributeType>(attr_proto.type());
  switch (type) {
    case fbs::AttributeType_FLOAT: {
      GET_FBS_ATTR(builder, type, f, attr_proto.f());
    } break;
    case fbs::AttributeType_INT: {
      GET_FBS_ATTR(builder, type, i, attr_proto.i());
    } break;
    case fbs::AttributeType_STRING: {
      auto s = builder.CreateString(attr_proto.s());
      GET_FBS_ATTR(builder, type, s, s);
    } break;
    case fbs::AttributeType_TENSOR: {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          experimental::utils::GetInitializerOrtFormat(builder, attr_proto.t(), fbs_tensor));
      GET_FBS_ATTR(builder, type, t, fbs_tensor);
    } break;
    case fbs::AttributeType_GRAPH: {
      ORT_RETURN_IF_NOT(!graph, "GetAttributeOrtFormat, graph is null");
      flatbuffers::Offset<fbs::Graph> fbs_graph;
      ORT_RETURN_IF_ERROR(graph->SaveToOrtFormat(builder, fbs_graph));
      GET_FBS_ATTR(builder, type, g, fbs_graph);
    } break;
    case fbs::AttributeType_FLOATS: {
      GET_DATA_VEC(float, floats_vec_, attr_proto.floats());
      auto floats = builder.CreateVector(floats_vec_);
      GET_FBS_ATTR(builder, type, floats, floats);
    } break;
    case fbs::AttributeType_INTS: {
      GET_DATA_VEC(int64_t, ints_vec_, attr_proto.ints());
      auto ints = builder.CreateVector(ints_vec_);
      GET_FBS_ATTR(builder, type, ints, ints);
    } break;
    case fbs::AttributeType_STRINGS: {
      GET_DATA_VEC(std::string, strings_vec_, attr_proto.strings());
      auto strings = builder.CreateVectorOfStrings(strings_vec_);
      GET_FBS_ATTR(builder, type, strings, strings);
    } break;
    case fbs::AttributeType_TENSORS: {
      std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors_vec;
      fbs_tensors_vec.reserve(attr_proto.tensors().size());
      for (const auto& tensor : attr_proto.tensors()) {
        flatbuffers::Offset<fbs::Tensor> fbs_tensor;
        ORT_RETURN_IF_ERROR(
            experimental::utils::GetInitializerOrtFormat(builder, tensor, fbs_tensor));
        fbs_tensors_vec.push_back(fbs_tensor);
      }
      auto tensors = builder.CreateVector(fbs_tensors_vec);
      GET_FBS_ATTR(builder, type, tensors, tensors);
    } break;
    default:
      break;
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GetAttributeOrtFormat - Unsupported type: ", type);
}

#undef GET_FBS_ATTR
#undef GET_DATA_VEC

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime