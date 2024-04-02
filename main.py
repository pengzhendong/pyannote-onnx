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

import click

import librosa
import matplotlib.pyplot as plt
import numpy as np

from pyannote_onnx import PyannoteONNX


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
def main(wav_path: str):
    model = PyannoteONNX()
    audio, sr = librosa.load(wav_path, sr=16000)
    outputs = model.run(None, {"input": audio[None, None, :]})[0][0]

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
