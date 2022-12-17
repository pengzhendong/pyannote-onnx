#include <vector>

#include "gflags/gflags.h"

#include "diarization/diarization_model.h"
#include "wav.h"

DEFINE_string(wav_path, "", "wav path");
DEFINE_string(model_path, "", "speaker diarization model path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wav::WavReader wav_reader(FLAGS_wav_path);
  const float* audio = wav_reader.data();
  int sampling_rate = wav_reader.sample_rate();
  int num_samples = wav_reader.num_samples();
  std::vector<float> input_wav{audio, audio + num_samples};

  DiarizationModel diarization(FLAGS_model_path);
  int num_speakers = diarization.num_speakers();

  std::vector<float> posterior;
  diarization.Forward(input_wav, &posterior);
  int len = posterior.size() / num_speakers;

  float threshold = 0.3;
  std::vector<std::vector<float>> start_pos;
  std::vector<std::vector<float>> stop_pos;
  start_pos.resize(num_speakers);
  stop_pos.resize(num_speakers);
  for (int i = 0; i < len; i++) {
    float current_pos = 1.0 * ((i * 270) + 721) / sampling_rate;
    for (int j = 0; j < num_speakers; j++) {
      float p = posterior[i * num_speakers + j];

      if (p > threshold) {
        if (start_pos[j].size() - stop_pos[j].size() != 1) {
          LOG(INFO) << "Speaker#" << j << " start=" << current_pos << "s";
          start_pos[j].emplace_back(current_pos);
        }
      } else {
        if (start_pos[j].size() - stop_pos[j].size() == 1) {
          LOG(INFO) << "Speaker#" << j << " stop=" << current_pos << "s";
          stop_pos[j].emplace_back(current_pos);
        }
      }
    }
  }
}
