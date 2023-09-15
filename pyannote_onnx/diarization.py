# Copyright (c) 2023, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from importlib_resources import files
import librosa
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="speaker diarization")
    parser.add_argument("--wav", required=True, help="input wav path")
    parser.add_argument(
        "--onnx_model",
        default=files("pyannote_onnx").joinpath("pyannote.onnx"),
        help="pyannote onnx model path",
    )
    args = parser.parse_args()

    ort_sess = ort.InferenceSession(args.onnx_model)
    audio, sr = librosa.load(args.wav, sr=16000)
    outputs = ort_sess.run(None, {"input": audio[None, None, :]})[0][0]

    # Conv1d & MaxPool1d & SincNet:
    #   * https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    #   * https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    #   * https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/blocks/sincnet.py#L50-L71
    #            kernel_size  stride
    # Conv1d             251      10
    # MaxPool1d            3       3
    # Conv1d               5       1
    # MaxPool1d            3       3
    # Conv1d               5       1
    # MaxPool1d            3       3
    # (L_{in} - 721) / 270 = L_{out}
    x1 = np.arange(0, len(audio)) / sr
    x2 = [(i * 270 + 721) / sr for i in range(0, len(outputs))]

    _, axs = plt.subplots(2)
    axs[0].plot(x1, audio)
    axs[1].plot(x2, outputs)
    axs[1].set_xlabel("time (s)")
    plt.show()


if __name__ == "__main__":
    main()
