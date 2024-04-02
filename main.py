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

from pyannote_onnx import PyannoteONNX


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
def main(wav_path: str):
    vad = PyannoteONNX()
    segements = vad.get_speech_timestamps(wav_path, return_seconds=True)
    print(segements)


if __name__ == "__main__":
    main()
