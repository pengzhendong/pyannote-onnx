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
from functools import partial
from itertools import permutations
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
from numpy.linalg import norm

from .inference_session import PickableInferenceSession


class PyannoteONNX:
    def __init__(self):
        # segmentation-3.0 classes:
        #   1. {no speech}
        #   2. {spk1}
        #   3. {spk2}
        #   4. {spk3}
        #   5. {spk1, spk2}
        #   6. {spk1, spk3}
        #   7. {spk2, spk3}
        # only keep the first 4 classes
        #   1. {speech}
        #   2. {spk1}
        #   3. {spk2}
        #   4. {spk3}
        self.num_classes = 4
        self.sample_rate = 16000
        self.duration = 10 * self.sample_rate
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
        # overlap: [0.5 * duration, 0.9 * duration]
        step = max(min(step, 0.9 * self.duration // 10), self.duration // 2)
        overlap = self.sample2frame(self.duration - step)
        overlap_chunk = np.zeros((overlap, self.num_classes))
        windows = list(self.sliding_window(x, self.duration, step))
        for idx, (window_size, window) in enumerate(windows):
            ort_outs = np.exp(
                self.session.run(None, {"input": window[None, None, :]})[0][0]
            )
            # https://herve.niderb.fr/fastpages/2022/10/23/One-speaker-segmentation-model-to-rule-them-all
            # reorder the speakers and aggregate
            ort_outs = np.concatenate(
                (
                    1 - ort_outs[:, :1],  # speech probabilities
                    self.reorder(
                        overlap_chunk[:, 1 : self.num_classes],
                        ort_outs[:, 1 : self.num_classes],
                    ),  # speaker probabilities
                ),
                axis=1,
            )
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

    def process_segment(
        self,
        idx,
        segment,
        wav,
        sample_rate,
        save_path,
        flat_layout,
        speech_pad_samples,
        return_seconds,
    ):
        step = sample_rate / self.sample_rate
        if step != 1.0:
            segment["start"] = int(segment["start"] * step)
            segment["end"] = int(segment["end"] * step)

        segment["start"] = max(segment["start"] - speech_pad_samples, 0)
        segment["end"] = min(segment["end"] + speech_pad_samples, len(wav))
        if save_path:
            wav = wav[segment["start"] : segment["end"]]
            if flat_layout:
                sf.write(str(save_path) + f"_{idx:04d}.wav", wav, sample_rate)
            else:
                sf.write(str(Path(save_path) / f"{idx:04d}.wav"), wav, sample_rate)
        if return_seconds:
            segment["start"] = round(segment["start"] / sample_rate, 3)
            segment["end"] = round(segment["end"] / sample_rate, 3)
        return segment

    def get_speech_timestamps(
        self,
        wav_path: Union[str, Path],
        save_path: Union[str, Path] = None,
        flat_layout: bool = True,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
    ):
        """
        Splitting long audios into speech chunks using Pyannote ONNX

        Parameters
        ----------
        wav_path: wav path
        save_path: string or Path (default - None)
            whether the save speech segments
        flat_layout: bool (default - True)
            whether use the flat directory structure
        threshold: float (default - 0.5)
            Speech threshold. Pyannote audio outputs speech probabilities for each audio
            chunk, probabilities ABOVE this value are considered as SPEECH. It is
            better to tune this parameter for each dataset separately, but "lazy"
            0.5 is pretty good for most datasets.
        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out
        max_speech_duration_s: int (default - inf)
            Maximum duration of speech chunks in seconds
            Chunks longer than max_speech_duration_s will be split at the timestamp
            of the last silence that lasts more than 98ms (if any), to prevent
            agressive cutting. Otherwise, they will be split aggressively just
            before max_speech_duration_s.
        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before
            separating it.
        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds
            based on return_seconds)
        """
        wav, sr = librosa.load(wav_path, sr=self.sample_rate)
        speech_pad_samples = sr * speech_pad_ms // 1000

        original_wav = wav
        sample_rate = librosa.get_samplerate(wav_path)
        if sample_rate != self.sample_rate:
            # load the wav with original sample rate for saving
            original_wav, _ = librosa.load(wav_path, sr=sample_rate)
        fn = partial(
            self.process_segment,
            wav=original_wav,
            sample_rate=sample_rate,
            save_path=save_path,
            flat_layout=flat_layout,
            speech_pad_samples=speech_pad_samples,
            return_seconds=return_seconds,
        )

        if len(wav.shape) > 1:
            raise ValueError(
                "More than one dimension in audio."
                "Are you trying to process audio with 2 channels?"
            )
        if sr / len(wav) > 31.25:
            raise ValueError("Input audio is too short.")

        min_speech_samples = sr * min_speech_duration_ms // 1000
        max_speech_samples = sr * max_speech_duration_s - 2 * speech_pad_samples
        min_silence_samples = sr * min_silence_duration_ms // 1000
        min_silence_samples_at_max_speech = sr * 98 // 1000

        current_speech = {}
        neg_threshold = threshold - 0.15
        triggered = False
        # to save potential segment end (and tolerate some silence)
        temp_end = 0
        # to save potential segment limits in case of maximum segment size reached
        prev_end = 0
        next_start = 0

        idx = 0
        current_samples = 721
        for outupt in self(wav):
            speech_prob = outupt[0]
            current_samples += 270
            # current frame is speech
            if speech_prob >= threshold:
                if temp_end > 0 and next_start < prev_end:
                    next_start = current_samples
                temp_end = 0
                if not triggered:
                    triggered = True
                    current_speech["start"] = current_samples
                    continue
            # in speech, and speech duration is more than max speech duration
            if (
                triggered
                and current_samples - current_speech["start"] > max_speech_samples
            ):
                # prev_end larger than 0 means there is a short silence in the middle avoid aggressive cutting
                if prev_end > 0:
                    current_speech["end"] = prev_end
                    yield fn(idx, current_speech)
                    idx += 1
                    current_speech = {}
                    # previously reached silence (< neg_thres) and is still not speech (< thres)
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                else:
                    current_speech["end"] = current_samples
                    yield fn(idx, current_speech)
                    idx += 1
                    current_speech = {}
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = False
                    continue
            # in speech, and current frame is silence
            if triggered and speech_prob < neg_threshold:
                if temp_end == 0:
                    temp_end = current_samples
                # record the last silence before reaching max speech duration
                if current_samples - temp_end > min_silence_samples_at_max_speech:
                    prev_end = temp_end
                if current_samples - temp_end >= min_silence_samples:
                    current_speech["end"] = temp_end
                    # keep the speech segment if it is longer than min_speech_samples
                    if (
                        current_speech["end"] - current_speech["start"]
                        > min_speech_samples
                    ):
                        yield fn(idx, current_speech)
                        idx += 1
                    current_speech = {}
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = False

        num_samples = len(wav)
        # deal with the last speech segment
        if (
            current_speech
            and num_samples - current_speech["start"] > min_speech_samples
        ):
            current_speech["end"] = num_samples
            yield fn(idx, current_speech)

    def get_num_speakers(
        self,
        wav_path: Union[str, Path],
        threshold: float = 0.5,
        min_speech_duration_ms: float = 100,
    ):
        """
        Get the max number of speakers
        """
        wav, sr = librosa.load(wav_path, sr=self.sample_rate)
        if len(wav.shape) > 1:
            raise ValueError(
                "More than one dimension in audio."
                "Are you trying to process audio with 2 channels?"
            )
        if sr / len(wav) > 31.25:
            raise ValueError("Input audio is too short.")

        outputs = np.array(list(self(wav)))[:, 1 : self.num_classes]
        speech_frames = np.sum(outputs > threshold, axis=0)
        speech_duration_ms = self.frame2sample(speech_frames) * 1000 / sr
        num_speakers = np.sum(speech_duration_ms > min_speech_duration_ms)
        return num_speakers
