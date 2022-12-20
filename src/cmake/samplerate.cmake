FetchContent_Declare(samplerate
  URL      https://github.com/libsndfile/libsamplerate/archive/refs/tags/0.2.2.tar.gz
  URL_HASH SHA256=16e881487f184250deb4fcb60432d7556ab12cb58caea71ef23960aec6c0405a
)
FetchContent_MakeAvailable(samplerate)
include_directories(${libsamplerate_SOURCE_DIR}/include)
