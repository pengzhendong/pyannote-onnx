add_library(frontend STATIC resampler.cc)
target_link_libraries(frontend PUBLIC glog samplerate)
