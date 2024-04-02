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

import click
import torch
import onnxruntime as ort

try:
    from pyannote.audio import Model
except ImportError:
    print("Please install pyannote: https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip")


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True))
@click.argument("onnx_model", type=click.Path(exists=False, file_okay=True))
def main(checkpoint: str, onnx_model: str):
    model = Model.from_pretrained(checkpoint)
    print(model)

    dummy_input = torch.zeros(3, 1, 32000)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "B", 1: "C", 2: "T"},
        },
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.optimized_model_filepath = onnx_model
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    ort.InferenceSession(onnx_model, sess_options=opts)


if __name__ == "__main__":
    main()
