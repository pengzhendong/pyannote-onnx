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
@click.option("--plot/--no-plot", default=False, help="Plot the vad probabilities")
def main(wav_path: str, plot: bool):
    pyannote = PyannoteONNX()
    for turn in pyannote.itertracks(wav_path):
        print(turn)

    if plot:
        pyannote = PyannoteONNX(show_progress=True)
        wav, sr = librosa.load(wav_path, sr=pyannote.sample_rate)
        outputs = list(pyannote(wav))
        x1 = np.arange(0, len(wav)) / sr
        x2 = [(i * 270 + 721) / sr for i in range(0, len(outputs))]
        plt.plot(x1, wav)
        plt.plot(x2, outputs)
        plt.show()


if __name__ == "__main__":
    main()
