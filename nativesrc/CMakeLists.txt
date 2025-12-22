cmake_minimum_required(VERSION 3.12)
project(PaddleSegInterence LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU selection is handled at runtime via the engine initialization parameters.
# Compile-time macro PADDLE_USE_GPU is no longer required and removed.

# User must set PADDLE_INFERENCE_DIR to point to Paddle Inference C++ SDK
set(PADDLE_DIR "D:/paddle/inference/paddle")

set(OpenCV_DIR "D:/opencv/build")
set(OpenCV_FOUND TRUE)

#find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PADDLE_DIR}/include)
include_directories(${OpenCV_DIR}/include)
link_directories(${PADDLE_DIR}/lib)

add_library(PaddleSegInterence SHARED
  src/paddleseg_api.cpp
)

target_link_libraries(PaddleSegInterence
  PRIVATE
  ${OpenCV_DIR}/x64/vc16/lib/opencv_world4120.lib
  ${PADDLE_DIR}/lib/paddle_inference.lib
  ${PADDLE_DIR}/lib/common.lib
)

if (WIN32)
  target_compile_definitions(PaddleSegInterence PRIVATE "_WINDOWS_EXPORT")
  if (MSVC)
    # Suppress noisy third-party warnings and enforce UTF-8 source charset
    target_compile_options(PaddleSegInterence PRIVATE /utf-8 /wd4251 /wd4244 /wd4819)
  endif()
endif()

# No compile-time PADDLE_USE_GPU definition; runtime init controls GPU usage.

set_target_properties(PaddleSegInterence PROPERTIES OUTPUT_NAME "PaddleSegInterence")

install(TARGETS PaddleSegInterence
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
