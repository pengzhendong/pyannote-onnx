# Speaker Diarization

> [pyannote-audio](https://github.com/pyannote/pyannote-audio) is an open-source toolkit written in Python for speaker diarization.

`pyannote-onnx` is used convert the [pretrained model](https://huggingface.co/pyannote/segmentation) defined in PyTorch into the ONNX format and then run it with [ONNX Runtime](https://github.com/microsoft/onnxruntime) (C++ or Python).

## Installation

> Only Python 3.8+ is supported.

``` bash
$ pip install -r requirements.txt
```

## Usage

1. Download the [pytorch model](https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin) from Hugging Face [pyannote/segmentation](https://huggingface.co/pyannote/segmentation/tree/main).
2. Export the pretrained model to ONNX model.

``` bash
$ python export_onnx.py \
  --checkpoint pytorch_model.bin \
  --onnx_model diarizatoin.onnx
```

3. Run the ONNX model with ONNX Runtime in C++ or Python.

### Python

TODO

### C++

- [RnNoise](https://github.com/werman/noise-suppression-for-voice)
- [libsamplerate](https://github.com/libsndfile/libsamplerate)

TODO
