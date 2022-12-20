add_library(frontend STATIC denoiser.cc resampler.cc)
target_link_libraries(frontend PUBLIC glog samplerate rnnoise)
