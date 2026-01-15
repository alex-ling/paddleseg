#include "../include/paddleseg_api.h"
#include <vector>
#include <memory>
#include <mutex>
#include <cstring>
#include <opencv2/opencv.hpp>

// Paddle Inference headers
#include <paddle_inference_api.h>

using namespace paddle_infer;

static std::shared_ptr<Predictor> g_predictor;
static std::mutex g_predictor_mutex;

// Background settings
static bool g_bg_is_color = true;
static cv::Scalar g_bg_color = cv::Scalar(0, 0, 0); // B,G,R
static cv::Mat g_bg_image;

// Set background to solid RGB color
int seg_set_background_rgb(unsigned char r, unsigned char g, unsigned char b) {
    g_bg_is_color = true;
    g_bg_color = cv::Scalar(b, g, r);
    g_bg_image.release();
    return 0;
}

// Set background from image path
int seg_set_background_image(const char* image_path) {
    if (!image_path) return -1;
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) return -2;
    g_bg_is_color = false;
    g_bg_image = img;
    return 0;
}

// Apply background using current mask (calls seg_infer internally)
int seg_apply_background(const unsigned char* src_rgba, int width, int height, unsigned char* out_rgba) {
    if (!src_rgba || !out_rgba) return -1;
    // compute mask
    std::vector<unsigned char> mask(width * height);
    int ret = seg_infer(src_rgba, width, height, mask.data());
    if (ret != 0) return ret;

    // prepare background BGR image of size width x height
    cv::Mat bg_resized;
    if (g_bg_is_color || g_bg_image.empty()) {
        // create uniform image
        bg_resized = cv::Mat(height, width, CV_8UC3, g_bg_color);
    } else {
        cv::resize(g_bg_image, bg_resized, cv::Size(width, height));
    }

    // composite
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            const unsigned char* p = src_rgba + (i * 4);
            unsigned char sr = p[0];
            unsigned char sg = p[1];
            unsigned char sb = p[2];
            unsigned char m = mask[i];
            float alpha = m / 255.0f;

            cv::Vec3b bgpix = bg_resized.at<cv::Vec3b>(y, x); // B,G,R
            unsigned char br = bgpix[2];
            unsigned char bg = bgpix[1];
            unsigned char bb = bgpix[0];

            unsigned char out_r = (unsigned char)(alpha * sr + (1.0f - alpha) * br + 0.5f);
            unsigned char out_g = (unsigned char)(alpha * sg + (1.0f - alpha) * bg + 0.5f);
            unsigned char out_b = (unsigned char)(alpha * sb + (1.0f - alpha) * bb + 0.5f);

            unsigned char* q = out_rgba + (i * 4);
            q[0] = out_r;
            q[1] = out_g;
            q[2] = out_b;
            q[3] = 255;
        }
    }

    return 0;
}

int seg_init(const char* model_dir, int enable_use_gpu, int gpu_mem_size, int cpu_thread_num, int enable_onednn, int enable_onnxruntime) {
    if (!model_dir) return -1;
    try {
        std::lock_guard<std::mutex> guard(g_predictor_mutex);
        Config config;
        std::string model_path = std::string(model_dir) + "/model.pdmodel";
        std::string params_path = std::string(model_dir) + "/model.pdiparams";
        config.SetModel(model_path, params_path);

        if (enable_use_gpu) {
            // enable GPU (gpu_mem_size MB, device 0). Adjust as needed.
            config.EnableUseGpu(gpu_mem_size, 0);
        } else {
            config.DisableGpu();
        }

        // set math library threads (applies to CPU/backends)
        if (cpu_thread_num > 0) {
            try {
                config.SetCpuMathLibraryNumThreads(cpu_thread_num);
            } catch (...) {
                // ignore if API not available for this Paddle build
            }
        }

        if (enable_use_gpu) {
            // When using GPU, enable IR optimizations and memory optimizations
            // to improve inference performance on GPU backends.
            try { config.SwitchIrOptim(true); } catch (...) {}
            try { config.EnableMemoryOptim(); } catch (...) {}
        } else {
            // CPU path: enable oneDNN and ONNXRuntime only for CPU builds/runtimes
            if (enable_onednn) {
                try { config.EnableoneDNN(); } catch (...) { try { config.EnableOneDNN(); } catch (...) {} }
            }
            if (enable_onnxruntime) {
                try { config.EnableONNXRuntime(); } catch (...) {}
            }
        }
        g_predictor = CreatePredictor(config);
        if (!g_predictor) return -2;
    } catch (...) {
        return -3;
    }
    return 0;
}

// simple helper: convert RGBA -> BGR float and normalize to 0..1
static inline int round_up(int v, int align) {
    return ((v + align - 1) / align) * align;
}

// Convert RGBA buffer to BGR float CHW with optional right/bottom padding to (pw,ph).
static void rgba_to_bgr_float_padded(const unsigned char* rgba, int w, int h, int pw, int ph, std::vector<float>& out) {
    out.resize(3 * pw * ph);
    // create cv mats to simplify color conversion and padding
    cv::Mat src_rgba(h, w, CV_8UC4, const_cast<unsigned char*>(rgba));
    cv::Mat bgr;
    cv::cvtColor(src_rgba, bgr, cv::COLOR_RGBA2BGR);

    if (pw != w || ph != h) {
        cv::Mat padded(ph, pw, bgr.type(), cv::Scalar::all(0));
        bgr.copyTo(padded(cv::Rect(0, 0, w, h)));
        bgr = padded;
    }

    // convert to float and normalize
    cv::Mat bgr_f;
    bgr.convertTo(bgr_f, CV_32F, 1.0f / 255.0f);

    // fill out in CHW order
    int area = pw * ph;
    for (int c = 0; c < 3; ++c) {
        float* dst = out.data() + c * area;
        for (int y = 0; y < ph; ++y) {
            const float* row = bgr_f.ptr<float>(y);
            for (int x = 0; x < pw; ++x) {
                int idx = y * pw + x;
                // BGR channels: row has 3 floats per pixel
                dst[idx] = row[x * 3 + c];
            }
        }
    }
}

int seg_infer(const unsigned char* rgba, int width, int height, unsigned char* out_mask) {
    if (!g_predictor) return -1;
    if (!rgba || !out_mask) return -2;

    // Basic pipeline: feed input as [1,3,H,W] floats. Many PaddleSeg models require specific resize and mean/std.
    // This implementation assumes model accepts native HxW; for best results, preexport or adjust preprocessing.

    // Many segmentation models require input sizes aligned to a multiple (e.g. 32).
    const int align = 32;
    int padded_w = round_up(width, align);
    int padded_h = round_up(height, align);

    std::vector<float> input_data;
    rgba_to_bgr_float_padded(rgba, width, height, padded_w, padded_h, input_data);

    // Get input names and tensor
    auto input_names = g_predictor->GetInputNames();
    if (input_names.empty()) return -3;
    auto input_t = g_predictor->GetInputHandle(input_names[0]);
    std::vector<int> shape = {1, 3, padded_h, padded_w};
    input_t->Reshape(shape);
    input_t->CopyFromCpu(input_data.data());

    // Run
    if (!g_predictor->Run()) return -4;

    // Get output
    auto output_names = g_predictor->GetOutputNames();
    if (output_names.empty()) return -5;
    auto out_t = g_predictor->GetOutputHandle(output_names[0]);
    std::vector<int> out_shape = out_t->shape();
    size_t total = 1;
    for (int s : out_shape) {
        total *= static_cast<size_t>(s);
    }
    std::vector<float> out_data(total);
    out_t->CopyToCpu(out_data.data());

    // Interpret output: if shape [1,1,H,W] -> use channel 0; if [1,2,H,W] -> use channel 1 as fg prob
    int C = (out_shape.size() >= 4) ? out_shape[1] : 1;
    int H = (out_shape.size() >= 4) ? out_shape[2] : padded_h;
    int W = (out_shape.size() >= 4) ? out_shape[3] : padded_w;

    // Build probability map (float) of size HxW. If output has channels, take channel 1 as foreground prob.
    cv::Mat prob_map(H, W, CV_32FC1);
    if (C == 1) {
        // straight copy
        memcpy(prob_map.data, out_data.data(), sizeof(float) * H * W);
    } else {
        // offset to channel 1
        size_t channel_size = static_cast<size_t>(H) * static_cast<size_t>(W);
        const float* ch1 = out_data.data() + channel_size; // channel 1
        memcpy(prob_map.data, ch1, sizeof(float) * channel_size);
    }

    // If model output was for padded size, crop back to original width/height region and then resize if necessary.
    cv::Mat prob_cropped = prob_map(cv::Rect(0, 0, std::min(W, padded_w), std::min(H, padded_h)));
    // Now resize/crop to the exact original image size
    cv::Mat prob_resized;
    if (prob_cropped.cols == width && prob_cropped.rows == height) {
        prob_resized = prob_cropped;
    } else {
        cv::resize(prob_cropped, prob_resized, cv::Size(width, height));
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float p = prob_resized.at<float>(y, x);
            out_mask[y * width + x] = (unsigned char)(std::min(std::max(p * 255.f, 0.f), 255.f));
        }
    }

    return 0;
}

void seg_release() {
    g_predictor.reset();
}
