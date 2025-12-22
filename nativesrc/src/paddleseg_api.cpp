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

        if (cpu_thread_num > 0) {
            // set math library threads (applies to CPU/backends)
            try {
                config.SetCpuMathLibraryNumThreads(cpu_thread_num);
            } catch (...) {
                // ignore if API not available for this Paddle build
            }
        }

        if (enable_onednn) {
            // enable oneDNN (may be named EnableoneDNN in some builds)
            try { config.EnableoneDNN(); } catch (...) { try { config.EnableOneDNN(); } catch (...) {} }
        }

        if (enable_onnxruntime) {
            try { config.EnableONNXRuntime(); } catch (...) {}
        }

        config.SwitchIrOptim();
        g_predictor = CreatePredictor(config);
        if (!g_predictor) return -2;
    } catch (...) {
        return -3;
    }
    return 0;
}

// simple helper: convert RGBA -> BGR float and normalize to 0..1
static void rgba_to_bgr_float(const unsigned char* rgba, int w, int h, std::vector<float>& out) {
    out.resize(3 * w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            const unsigned char* p = rgba + (i * 4);
            // input RGBA -> BGR
            out[0 * w * h + i] = p[2] / 255.f;
            out[1 * w * h + i] = p[1] / 255.f;
            out[2 * w * h + i] = p[0] / 255.f;
        }
    }
}

int seg_infer(const unsigned char* rgba, int width, int height, unsigned char* out_mask) {
    if (!g_predictor) return -1;
    if (!rgba || !out_mask) return -2;

    // Basic pipeline: feed input as [1,3,H,W] floats. Many PaddleSeg models require specific resize and mean/std.
    // This implementation assumes model accepts native HxW; for best results, preexport or adjust preprocessing.

    std::vector<float> input_data;
    rgba_to_bgr_float(rgba, width, height, input_data);

    // Get input names and tensor
    auto input_names = g_predictor->GetInputNames();
    if (input_names.empty()) return -3;
    auto input_t = g_predictor->GetInputHandle(input_names[0]);
    std::vector<int> shape = {1, 3, height, width};
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
    int H = (out_shape.size() >= 4) ? out_shape[2] : height;
    int W = (out_shape.size() >= 4) ? out_shape[3] : width;

    if (H != height || W != width) {
        // resize probability map to requested size
        cv::Mat prob_map;
        if (C == 1) {
            cv::Mat src(out_shape[2], out_shape[3], CV_32FC1, out_data.data());
            cv::resize(src, prob_map, cv::Size(width, height));
        } else {
            // take channel 1
            std::vector<float> ch(out_shape[2] * out_shape[3]);
            for (int y = 0; y < out_shape[2]; ++y) for (int x = 0; x < out_shape[3]; ++x) {
                ch[y * out_shape[3] + x] = out_data[(1 * out_shape[2] + y) * out_shape[3] + x];
            }
            cv::Mat src(out_shape[2], out_shape[3], CV_32FC1, ch.data());
            cv::resize(src, prob_map, cv::Size(width, height));
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float p = prob_map.at<float>(y, x);
                out_mask[y * width + x] = (unsigned char)(std::min(std::max(p * 255.f, 0.f), 255.f));
            }
        }
    } else {
        if (C == 1) {
            for (int i = 0; i < H*W; ++i) {
                float p = out_data[i];
                out_mask[i] = (unsigned char)(std::min(std::max(p * 255.f, 0.f), 255.f));
            }
        } else {
            // channel 1
            for (int i = 0; i < H*W; ++i) {
                float p = out_data[(1 * H + (i / W)) * W + (i % W)];
                out_mask[i] = (unsigned char)(std::min(std::max(p * 255.f, 0.f), 255.f));
            }
        }
    }

    return 0;
}

void seg_release() {
    g_predictor.reset();
}
