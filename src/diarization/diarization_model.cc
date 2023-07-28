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

DiarizationModel::DiarizationModel(const std::string& model_path,
                                   float threshold, float max_dur)
    : OnnxModel(model_path), threshold_(threshold), max_dur_(max_dur) {
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
  int64_t input_node_dims[3] = {batch_size, num_channels,
                                static_cast<int64_t>(audio.size())};
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

float DiarizationModel::Diarization(const std::vector<float>& in_wav,
                                    std::vector<std::vector<float>>* start_pos,
                                    std::vector<std::vector<float>>* stop_pos) {
  std::vector<float> posterior;
  Forward(in_wav, &posterior);

  start_pos->resize(num_speakers_);
  stop_pos->resize(num_speakers_);
  int len = posterior.size() / num_speakers_;

  float cur_pos = 0;
  for (int i = 0; i < len; i++) {
    // Conv1d & MaxPool1d & SincNet:
    //   * https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    //   * https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    //   * https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/blocks/sincnet.py#L50-L71
    //            kernel_size  stride
    // FBank              251      10
    // MaxPool1d            3       3
    // Conv1d               5       1
    // MaxPool1d            3       3
    // Conv1d               5       1
    // MaxPool1d            3       3
    // (L_{in} - 721) / 270 = L_{out}
    cur_pos = round(1.0 * ((i * 270) + 721) / SAMPLE_RATE * 1000) / 1000.0;
    for (int j = 0; j < num_speakers_; j++) {
      float p = posterior[i * num_speakers_ + j];
      std::vector<float>& start = start_pos->at(j);
      std::vector<float>& stop = stop_pos->at(j);

      if (p > threshold_) {
        if (start.size() - stop.size() != 1) {
          start.emplace_back(cur_pos);
        }
      } else {
        if (start.size() - stop.size() == 1) {
          float break_pos = round((start.back() + max_dur_) * 1000) / 1000.0;
          if (cur_pos > break_pos) {
            stop.emplace_back(break_pos);
            start.emplace_back(break_pos);
          }
          stop.emplace_back(cur_pos);
        }
      }
    }
  }
  return cur_pos;
}
