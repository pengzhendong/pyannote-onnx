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

#ifndef FRONTEND_RESAMPLER_H_
#define FRONTEND_RESAMPLER_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "samplerate.h"

class Resampler {
 public:
  explicit Resampler(int converter = SRC_SINC_BEST_QUALITY)
      : converter_(converter) {
    src_data_ = std::make_shared<SRC_DATA>();
  }

  void Resample(int in_sr, const std::vector<float>& in_wav, int out_sr,
                std::vector<float>* out_wav);

 private:
  int converter_;
  std::shared_ptr<SRC_DATA> src_data_;
};

#endif  // FRONTEND_RESAMPLER_H_
