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

#include <iomanip>
#include <sstream>
#include <vector>

#include "gflags/gflags.h"

#include "diarization/diarization_model.h"
#include "frontend/resampler.h"
#include "frontend/wav.h"

DEFINE_string(wav_path, "", "wav path");
DEFINE_double(max_dur, 20, "max duration of one segment in seconds");
DEFINE_double(threshold, 0.3, "threshold of speaker diarization");
DEFINE_string(model_path, "", "speaker diarization model path");
DEFINE_string(output_dir, "", "output dir for segment wavs");

#define SAMPLE_RATE 16000

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wav::WavReader wav_reader(FLAGS_wav_path);
  int num_channels = wav_reader.num_channels();
  CHECK_EQ(num_channels, 1) << "Only support mono (1 channel) wav!";
  int bits_per_sample = wav_reader.bits_per_sample();
  int sample_rate = wav_reader.sample_rate();
  const float* audio = wav_reader.data();
  int num_samples = wav_reader.num_samples();
  std::vector<float> input_wav{audio, audio + num_samples};

  std::vector<float> resampled_wav;
  auto resampler = std::make_shared<Resampler>();

  // 0. Upsample to 48k for RnNoise
  if (sample_rate != 16000) {
    resampler->Resample(sample_rate, input_wav, 16000, &resampled_wav);
    input_wav = resampled_wav;
  }

  // 1. Speaker Diarization
  auto diarization = std::make_shared<DiarizationModel>(
      FLAGS_model_path, FLAGS_threshold, FLAGS_max_dur);

  std::vector<std::vector<float>> start_pos;
  std::vector<std::vector<float>> stop_pos;
  int num_speakers = diarization->num_speakers();
  float dur = diarization->Diarization(input_wav, &start_pos, &stop_pos);
  for (int i = 0; i < num_speakers; i++) {
    for (int j = 0; j < start_pos[i].size(); j++) {
      float start = start_pos[i][j];
      float stop = j < stop_pos[i].size() ? stop_pos[i][j] : dur;
      LOG(INFO) << "Speaker#" << i << " segments#" << j << " [" << start << ", "
                << stop << "]s.";
      // 4. Save segments from **original** wav to wavs
      if (!FLAGS_output_dir.empty()) {
        int start_sample = start * sample_rate;
        num_samples = stop * sample_rate - start_sample;
        wav::WavWriter writer(audio + start_sample, num_samples, 1, sample_rate,
                              bits_per_sample);
        std::stringstream wav_name;
        wav_name << std::fixed << std::setprecision(2) << "spk#" << i << "_seg#"
                 << j << "_" << start << "s_" << stop << "s";
        writer.Write(FLAGS_output_dir + "/" + wav_name.str() + ".wav");
      }
    }
  }
}
