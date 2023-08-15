# Copyright (c) 2022, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import torch
import onnxruntime as ort
from pyannote.audio import Model


def main():
    parser = argparse.ArgumentParser(description="export onnx model")
    parser.add_argument("--checkpoint", required=True, help="checkpoint")
    parser.add_argument("--onnx_model", required=True, help="onnx model")
    args = parser.parse_args()

    model = Model.from_pretrained(args.checkpoint)
    print(model)

    dummy_input = torch.zeros(3, 1, 32000)
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_model,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "B", 1: "C", 2: "T"},
        },
    )
    so = ort.SessionOptions()
    so.optimized_model_filepath = args.onnx_model
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    ort.InferenceSession(args.onnx_model, sess_options=so)


if __name__ == "__main__":
    main()
