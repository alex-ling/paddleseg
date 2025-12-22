#pragma once

#include <stdint.h>

#ifdef _WIN32
#  ifdef _WINDOWS_EXPORT
#    define PADDLESEG_EXPORT __declspec(dllexport)
#  else
#    define PADDLESEG_EXPORT __declspec(dllimport)
#  endif
#else
#  define PADDLESEG_EXPORT
#endif

extern "C" {

// seg_init parameters:
// enable_use_gpu: 0 = CPU, 1 = enable GPU runtime
// cpu_thread_num: number of threads for CPU math library (if >0)
// enable_onednn: 0 = disabled, 1 = enable oneDNN (MKL/oneDNN) optimizations
// enable_onnxruntime: 0 = disabled, 1 = enable ONNXRuntime integration
PADDLESEG_EXPORT int seg_init(const char* model_dir, int enable_use_gpu, int gpu_mem_size, int cpu_thread_num, int enable_onednn, int enable_onnxruntime);

// rgba: pointer to input image pixels (RGBA, 8-bit per channel)
// width, height: input image size
// out_mask: pre-allocated buffer width*height bytes, will contain 0..255 mask
// returns 0 on success
PADDLESEG_EXPORT int seg_infer(const unsigned char* rgba, int width, int height, unsigned char* out_mask);

PADDLESEG_EXPORT void seg_release();

// Set a solid background color (R,G,B)
PADDLESEG_EXPORT int seg_set_background_rgb(unsigned char r, unsigned char g, unsigned char b);

// Set background from image file path (will be resized to input size on apply)
PADDLESEG_EXPORT int seg_set_background_image(const char* image_path);

// Apply the currently configured background to an RGBA input image.
// src_rgba: input RGBA pixels (R,G,B,A)
// out_rgba: output RGBA pixels (R,G,B,A) - must be preallocated width*height*4 bytes
PADDLESEG_EXPORT int seg_apply_background(const unsigned char* src_rgba, int width, int height, unsigned char* out_rgba);

}
