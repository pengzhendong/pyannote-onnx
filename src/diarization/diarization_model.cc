// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "diarization/diarization_model.h"

#include "glog/logging.h"

DiarizationModel::DiarizationModel(const std::string& model_path)
    : OnnxModel(model_path) {
  //   num_speakers_
  Ort::TypeInfo type_info = session_->GetOutputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType type = tensor_info.GetElementType();
  std::vector<int64_t> node_dims = tensor_info.GetShape();
  CHECK_EQ(node_dims.size(), 3);
  num_speakers_ = node_dims[2];
}

void DiarizationModel::Forward(const std::vector<float>& audio,
                               std::vector<float>* posterior) {
  // batch_size * num_channels (1 for mono) * num_samples
  const int64_t batch_size = 1;
  const int64_t num_channels = 1;
  int64_t input_node_dims[3] = {batch_size, num_channels, audio.size()};
  Ort::Value input_ort = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(audio.data()), audio.size(),
      input_node_dims, 3);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.emplace_back(std::move(input_ort));

  auto ort_outputs = session_->Run(
      Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
      ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

  const float* outputs = ort_outputs[0].GetTensorData<float>();
  auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int len = outputs_shape[1];
  posterior->assign(outputs, outputs + (batch_size * len * num_speakers_));
}
