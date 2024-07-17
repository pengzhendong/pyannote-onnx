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

import os
from itertools import permutations
from pathlib import Path
from typing import Union

import librosa
import numpy as np
from tqdm import tqdm

from .inference_session import PickableInferenceSession


class PyannoteONNX:
    def __init__(self, show_progress: bool = False):
        # segmentation-3.0 classes:
        #   1. {no speech}
        #   2. {spk1}
        #   3. {spk2}
        #   4. {spk3}
        #   5. {spk1, spk2}
        #   6. {spk1, spk3}
        #   7. {spk2, spk3}
        # only keep the speaker classes
        #   1. {spk1}
        #   2. {spk2}
        #   3. {spk3}
        self.num_classes = 3
        self.sample_rate = 16000
        self.duration = 10 * self.sample_rate
        self.show_progress = show_progress
        onnx_model = f"{os.path.dirname(__file__)}/segmentation-3.0.onnx"
        self.session = PickableInferenceSession(onnx_model)

    @staticmethod
    def sample2frame(x):
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
        return (x - 721) // 270

    @staticmethod
    def frame2sample(x):
        return (x * 270) + 721

    @staticmethod
    def sliding_window(waveform, window_size, step_size):
        windows = []
        start = 0
        num_samples = len(waveform)
        while start <= num_samples - window_size:
            windows.append((start, start + window_size))
            yield window_size, waveform[start : start + window_size]
            start += step_size
        # last incomplete window
        if num_samples < window_size or (num_samples - window_size) % step_size > 0:
            last_window = waveform[start:]
            last_window_size = len(last_window)
            if last_window_size < window_size:
                last_window = np.pad(last_window, (0, window_size - last_window_size))
            yield last_window_size, last_window

    @staticmethod
    def reorder(x, y):
        perms = [np.array(perm).T for perm in permutations(y.T)]
        diffs = np.sum(
            np.abs(np.sum(np.array(perms)[:, : x.shape[0], :] - x, axis=1)), axis=1
        )
        return perms[np.argmin(diffs)]

    def __call__(self, x, step=5.0, return_chunk=False):
        step = int(step * self.sample_rate)
        # step: [0.5 * duration, 0.9 * duration]
        step = max(min(step, 0.9 * self.duration), self.duration // 2)
        # overlap: [0.1 * duration, 0.5 * duration]
        overlap = self.sample2frame(self.duration - step)
        overlap_chunk = np.zeros((overlap, self.num_classes))
        windows = list(self.sliding_window(x, self.duration, step))
        if self.show_progress:
            progress_bar = tqdm(
                total=len(windows),
                desc="Pyannote processing",
                unit="frames",
                bar_format="{l_bar}{bar}{r_bar} | {percentage:.2f}%",
            )
        for idx, (window_size, window) in enumerate(windows):
            if self.show_progress:
                progress_bar.update(1)
            ort_outs = self.session.run(None, {"input": window[None, None, :]})[0][0]
            ort_outs = np.exp(ort_outs[:, 1 : self.num_classes + 1])
            # https://herve.niderb.fr/fastpages/2022/10/23/One-speaker-segmentation-model-to-rule-them-all
            # reorder the speakers and aggregate
            ort_outs = self.reorder(overlap_chunk, ort_outs)
            if idx != 0:
                ort_outs[:overlap, :] = (ort_outs[:overlap, :] + overlap_chunk) / 2
            if idx != len(windows) - 1:
                overlap_chunk = ort_outs[-overlap:, :]
                ort_outs = ort_outs[:-overlap, :]
            else:
                # crop
                ort_outs = ort_outs[: self.sample2frame(window_size), :]

            if return_chunk:
                yield ort_outs
            else:
                for out in ort_outs:
                    yield out

    def itertracks(self, wav, onset: float = 0.5, offset: float = 0.5):
        if not isinstance(wav, np.ndarray):
            wav, _ = librosa.load(wav, sr=self.sample_rate, mono=True)

        current_samples = 721
        start = [0] * self.num_classes
        is_active = [False] * self.num_classes
        for speech_probs in self(wav):
            current_samples += 270
            for idx, prob in enumerate(speech_probs):
                if is_active[idx]:
                    if prob < offset:
                        yield {
                            "speaker": idx,
                            "start": round(start[idx] / self.sample_rate, 3),
                            "stop": round(current_samples / self.sample_rate, 3),
                        }
                        is_active[idx] = False
                else:
                    if prob > onset:
                        start[idx] = current_samples
                        is_active[idx] = True
        for idx in range(self.num_classes):
            if is_active[idx]:
                yield {
                    "speaker": idx,
                    "start": round(start[idx] / self.sample_rate, 3),
                    "stop": round(current_samples / self.sample_rate, 3),
                }
