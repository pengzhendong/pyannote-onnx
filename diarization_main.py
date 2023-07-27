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

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description='speaker diarization')
    parser.add_argument('--onnx_model', required=True, help='onnx model')
    parser.add_argument('--wav', required=True, help='wav file')
    args = parser.parse_args()

    ort_sess = ort.InferenceSession(args.onnx_model)
    audio, _ = sf.read(args.wav, dtype='float32')
    outputs = ort_sess.run(None, {'input': audio[None, None, :]})[0][0]

    _, axs = plt.subplots(2)
    axs[0].plot(audio)
    axs[1].plot(outputs)
    plt.show()


if __name__ == '__main__':
    main()
