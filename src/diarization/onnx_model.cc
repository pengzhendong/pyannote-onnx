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

#include <sstream>

#include "diarization/onnx_model.h"

#include "glog/logging.h"

Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions OnnxModel::session_options_ = Ort::SessionOptions();

void OnnxModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

static std::wstring ToWString(const std::string& str) {
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t* p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

OnnxModel::OnnxModel(const std::string& model_path) {
  InitEngineThreads(1);
#ifdef _MSC_VER
  session_ = std::make_shared<Ort::Session>(env_, ToWString(model_path).c_str(),
                                            session_options_);
#else
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
  return session;
#endif
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session_->GetInputCount();
  input_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    input_node_names_[i] = session_->GetInputName(i, allocator);
    LOG(INFO) << "Input names[" << i << "]: " << input_node_names_[i];
  }
  // Output info
  num_nodes = session_->GetOutputCount();
  output_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    output_node_names_[i] = session_->GetOutputName(i, allocator);
    LOG(INFO) << "Output names[" << i << "]: " << output_node_names_[i];
  }
}
