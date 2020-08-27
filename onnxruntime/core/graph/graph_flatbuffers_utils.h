// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/graph/graph.h>

namespace onnxruntime {
namespace experimental {
namespace utils {
onnxruntime::common::Status GetValueInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                                  const ONNX_NAMESPACE::ValueInfoProto& value_info_proto,
                                                  flatbuffers::Offset<fbs::ValueInfo>& fbs_value_info);

onnxruntime::common::Status GetInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                                    const ONNX_NAMESPACE::TensorProto& initializer,
                                                    flatbuffers::Offset<fbs::Tensor>& fbs_tensor);

// Convert a given AttributeProto into fbs::Attribute
// Note, we current do not support graphs, and sparse_tensor(s)
//       If the attribute type is a graph, we need to use the supplied graph,
//       instead of the GraphProto in attr_proto
onnxruntime::common::Status GetAttributeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                                  const ONNX_NAMESPACE::AttributeProto& attr_proto,
                                                  flatbuffers::Offset<fbs::Attribute>& fbs_attr,
                                                  const onnxruntime::Graph* graph);
}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime