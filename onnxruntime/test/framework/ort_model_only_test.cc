// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {

// InferenceSession wrapper to expose loaded graph.
class InferenceSessionGetGraphWrapper : public InferenceSession {
 public:
  explicit InferenceSessionGetGraphWrapper(const SessionOptions& session_options,
                                           const Environment& env) : InferenceSession(session_options, env) {
  }

  const Graph& GetGraph() {
    return model_->MainGraph();
  }
};

namespace test {
#if !defined(ORT_MINIMAL_BUILD)
TEST(OrtModelOnlyTests, SerializeToOrtFormat) {
  const auto output_file = ORT_TSTR("ort_github_issue_4031.onnx.ort");
  SessionOptions so;
  so.session_logid = "SerializeToOrtFormat";
  so.optimized_model_filepath = output_file;
  // not strictly necessary - type should be inferred from the filename
  so.AddConfigEntry(ORT_SESSION_OPTIONS_CONFIG_SAVE_MODEL_FORMAT, "ORT");

  InferenceSessionGetGraphWrapper session_object{so, GetEnvironment()};

  // create .ort file during Initialize due to values in SessionOptions
  ASSERT_STATUS_OK(session_object.Load(ORT_TSTR("testdata/ort_github_issue_4031.onnx")));
  ASSERT_STATUS_OK(session_object.Initialize());

  // create inputs
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"state_var_out"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(feeds, output_names, &fetches));

  SessionOptions so2;
  so.session_logid = "LoadOrtFormat";
  // not strictly necessary - type should be inferred from the filename, but to be sure we're testing what we
  // think we're testing set it.
  so.AddConfigEntry(ORT_SESSION_OPTIONS_CONFIG_LOAD_MODEL_FORMAT, "ORT");

  // load serialized version
  InferenceSessionGetGraphWrapper session_object2{so2, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.Load(output_file));
  ASSERT_STATUS_OK(session_object2.Initialize());

  // compare contents on Graph instances
  const auto& graph = session_object.GetGraph();
  const auto& graph2 = session_object2.GetGraph();

  const auto& i1 = graph.GetAllInitializedTensors();
  const auto& i2 = graph2.GetAllInitializedTensors();
  ASSERT_EQ(i1.size(), i2.size());

  for (const auto& pair : i1) {
    auto iter = i2.find(pair.first);
    ASSERT_NE(iter, i2.cend());

    const TensorProto& left = *pair.second;
    const TensorProto& right = *iter->second;
    std::string left_data;
    std::string right_data;
    left.SerializeToString(&left_data);
    right.SerializeToString(&right_data);
    ASSERT_EQ(left_data, right_data);
  }

  // check all node args are fine
  for (const auto& input : graph.GetInputsIncludingInitializers()) {
    const auto& left = *graph.GetNodeArg(input->Name());
    const auto* right = graph2.GetNodeArg(input->Name());
    ASSERT_TRUE(right != nullptr);

    // serialize as strings so it's easy to compare all values at once
    std::string left_data;
    std::string right_data;
    left.ToProto().SerializeToString(&left_data);
    right->ToProto().SerializeToString(&right_data);
    ASSERT_EQ(left_data, right_data);
  }

  // check results match
  std::vector<OrtValue> fetches2;
  ASSERT_STATUS_OK(session_object2.Run(feeds, output_names, &fetches2));

  const auto& output = fetches[0].Get<Tensor>();
  ASSERT_TRUE(output.Shape().Size() == 1);
  ASSERT_TRUE(output.Data<float>()[0] == 125.f);

  const auto& output2 = fetches2[0].Get<Tensor>();
  ASSERT_TRUE(output2.Shape().Size() == 1);
  ASSERT_TRUE(output2.Data<float>()[0] == 125.f);
}
#endif

// FIXME: Need to save ORT format model and checkin to testdata
/*
// test that we can deserialize and run a model
TEST(OrtModelOnlyTests, LoadOrtFormatModel) {
  const auto output_file = ORT_TSTR("ort_github_issue_4031.onnx.ort");
  SessionOptions so;
  so.session_logid = "LoadOrtFormatModel";
  so.optimized_model_filepath = output_file;
  so.optimized_model_format = ORT;

  InferenceSessionGetGraphWrapper session_object2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.Load(output_file)); // infer type from filename
  ASSERT_STATUS_OK(session_object2.Initialize());

  const auto& graph2 = session_object2.GetGraph();
  std::cout << graph2.MaxNodeIndex() << " is max node index\n";

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"state_var_out"};
  std::vector<OrtValue> fetches;
  std::vector<OrtValue> fetches2;

  ASSERT_STATUS_OK(session_object2.Run(feeds, output_names, &fetches2));

  const auto& output2 = fetches2[0].Get<Tensor>();
  ASSERT_TRUE(output2.Shape().Size() == 1);
  ASSERT_TRUE(output2.Data<float>()[0] == 125.f);
}
*/
}  // namespace test
}  // namespace onnxruntime
