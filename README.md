# Speaker Diarization

> [pyannote-audio](https://github.com/pyannote/pyannote-audio) is an open-source toolkit written in Python for speaker diarization.

`pyannote-onnx` is used to convert the [pretrained model](https://huggingface.co/pyannote/segmentation) defined in PyTorch into the ONNX format and then run it with [ONNX Runtime](https://github.com/microsoft/onnxruntime) (in C++ or Python).

## Installation

> Only Python 3.8+ is supported.

``` bash
$ pip install -r requirements.txt
```

## Usage

1. Download the [pretrained model](https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin) from Hugging Face [pyannote/segmentation](https://huggingface.co/pyannote/segmentation/tree/main).
2. Export the pretrained model to ONNX model.
3. Run the ONNX model with ONNX Runtime in C++ or Python.

``` bash
$ python export_onnx.py \
    --checkpoint data/pytorch_model.bin \
    --onnx_model pyannote.onnx
```

### Python Usage

``` bash
$ python pyannote_onnx/diarization.py \
    --onnx_model pyannote.onnx \
    --wav data/test_16k.wav
```

### C++ Usage

``` bash
$ cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
$ mkdir output
$ ./build/diarization_main \
    --model_path pyannote.onnx \
    --wav_path data/test_16k.wav \
    --output_dir output
```
