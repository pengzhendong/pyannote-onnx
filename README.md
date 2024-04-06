# Speaker Diarization

> [pyannote-audio](https://github.com/pyannote/pyannote-audio) is an open-source toolkit written in Python for speaker diarization.

`pyannote-onnx` is used to convert the [pretrained model](https://huggingface.co/pyannote/segmentation-3.0) defined in PyTorch into the ONNX format and then run it with [ONNX Runtime](https://github.com/microsoft/onnxruntime) (in C++ or Python).

> Only Python 3.8+ is supported.

## Usage

1. Download the [pretrained model](https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin) from Hugging Face [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0/tree/main).
2. Export the pretrained model to ONNX model.
3. Run the ONNX model with ONNX Runtime in C++ or Python.

```bash
$ pip install torch onnx https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
$ python export_onnx.py pytorch_model.bin segmentation-3.0.onnx

$ pip install -r requirements.txt
$ python main.py data/test_16k.wav
```
