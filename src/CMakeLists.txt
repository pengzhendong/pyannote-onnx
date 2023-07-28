cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speaker_diarization VERSION 0.1)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base
  "fc_base-${CMAKE_CXX_COMPILER_ID}"
  REALPATH BASE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
set(FETCHCONTENT_BASE_DIR ${fc_base})
option(BUILD_TESTING "whether to build unit test" OFF)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
else()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options(/W0 /wd4150 /wd4244 /wd4267)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(gflags)
set(WITH_GFLAGS OFF CACHE BOOL "whether build glog with gflags" FORCE)
include(samplerate) # Note: must include libsamplerate before glog
include(glog)
include(onnx)

add_subdirectory(frontend)
add_subdirectory(diarization)
add_executable(diarization_main bin/diarization_main.cc)
target_link_libraries(diarization_main PUBLIC frontend diarization gflags)
