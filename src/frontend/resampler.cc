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

#include "frontend/resampler.h"

void Resampler::Resample(int in_sr, const std::vector<float>& in_wav,
                         int out_sr, std::vector<float>* out_wav) {
  float ratio = 1.0 * out_sr / in_sr;
  out_wav->resize(in_wav.size() * ratio);

  src_data_->src_ratio = ratio;
  src_data_->data_in = in_wav.data();
  src_data_->input_frames = in_wav.size();
  src_data_->data_out = out_wav->data();
  src_data_->output_frames = out_wav->size();

  int error = src_simple(src_data_.get(), converter_, 1);
  if (error != 0) {
    LOG(FATAL) << "src_simple error: " << src_strerror(error);
  }
  LOG(INFO) << "input_frames_used[" << src_data_->input_frames_used
            << "], output_frames_gen[" << src_data_->output_frames_gen << "]";
}
