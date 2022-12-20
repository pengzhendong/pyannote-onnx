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

#ifndef FRONTEND_DENOISER_H_
#define FRONTEND_DENOISER_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "rnnoise/rnnoise.h"

#define FRAME_SIZE 480  // According to rnnoise/src/denoise.c

class Denoiser {
 public:
  explicit Denoiser() { st_.reset(rnnoise_create()); };

  void Denoise(const std::vector<float>& in_wav, std::vector<float>* out_wav);

 private:
  std::shared_ptr<DenoiseState> st_;
};

#endif  // FRONTEND_DENOISER_H_
