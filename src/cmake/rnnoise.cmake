option(BUILD_FOR_RELEASE "Additional optimizations and steps may be taken for release" ON)
option(BUILD_TESTS "" OFF)
option(BUILD_VST_PLUGIN "If the VST2 plugin should be built" OFF)
option(BUILD_VST3_PLUGIN "If the VST3 plugin should be built" OFF)
option(BUILD_LV2_PLUGIN "If the LV2 plugin should be built" OFF)
option(BUILD_LADSPA_PLUGIN "If the LADSPA plugin should be built" OFF)
option(BUILD_AU_PLUGIN "If the AU plugin should be built (macOS only)" OFF)
option(BUILD_AUV3_PLUGIN "If the AUv3 plugin should be built (macOS only)" OFF)

FetchContent_Declare(rnnoise
  URL      https://github.com/werman/noise-suppression-for-voice/archive/refs/tags/v1.03.tar.gz
  URL_HASH SHA256=8c85cae3ebbb3a18facc38930a3b67ca90e3ad609526a0018c71690de35baf04
)
FetchContent_MakeAvailable(rnnoise)
include_directories(${rnnoise_SOURCE_DIR}/src/rnnoise/include)
link_directories(${CMAKE_BINARY_DIR}/lib)
