add_library(diarization STATIC onnx_model.cc diarization_model.cc)
target_link_libraries(diarization PUBLIC glog onnxruntime)
