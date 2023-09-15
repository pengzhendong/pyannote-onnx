# Copyright (c) 2023 Zhendong Peng (pzd17@tsinghua.org.cn)
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
from setuptools import setup, find_packages


with open("README.md", encoding="utf8") as fin:
    long_description = fin.read()

setup(
    name="pyannote-onnx",
    version=os.getenv("BUILD_VERSION") or "0.0.1",
    author="Zhendong Peng",
    author_email="pzd17@tsinghua.org.cn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Pyannote ONNX",
    url="https://github.com/pengzhendong/pyannote-onnx",
    packages=find_packages(),
    package_data={
        "pyannote_onnx": ["*.onnx"],
    },
    install_requires=["librosa", "matplotlib", "numpy", "onnxruntime"],
    entry_points={
        "console_scripts": [
            "diarization = pyannote_onnx.diarization:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
