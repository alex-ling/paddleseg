# Paddle Segmentation Native + .NET Interop

Basic usage (Windows, from project root):

```powershell
cd .\publish\win-x64
# generate a mask (subcommand `mask`)
.\PaddleSegCli.exe mask -m <modelDir> -i <input.png> -o <output.png>

# apply a background (subcommand `apply`), use either --bg-color or --bg-image
.\PaddleSegCli.exe apply -m <modelDir> -i <input.png> -o <output.png> --bg-color 0,0,0
.\PaddleSegCli.exe apply -m <modelDir> -i <input.png> -o <output.png> --bg-image "C:\path\to\bg.jpg"
```

```powershell
dotnet build src/Acesoft.PaddleSeg/Acesoft.PaddleSeg.csproj -c Release
dotnet build src/Acesoft.PaddleSegCli/Acesoft.PaddleSegCli.csproj -c Release
```

## Run (.NET)

Copy the native `PaddleSegInterence.dll` (Windows) or `libPaddleSegInterence.so` (Linux) and its Paddle/OpenCV runtime deps next to the .NET binaries, or make them discoverable via PATH.

Example publish (Windows x64):

```powershell
# from repository root
dotnet publish src/Acesoft.PaddleSegCli/Acesoft.PaddleSegCli.csproj -c Release -r win-x64 -o publish/win-x64 --self-contained false
```

## Build native library (CMake)

The native interop library is in the `nativesrc` folder and must be built separately. The library no longer toggles GPU at compile-time; GPU vs CPU is chosen at runtime via the engine init parameters.

Windows (Visual Studio / MSVC):

```powershell
# Open "x64 Native Tools Command Prompt for VS" (or use an x64 developer PowerShell)
Import-Module "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell f92f0502
cmake -S nativesrc -B build_native -D PADDLE_DIR="D:/paddle/inference/paddle" -D OpenCV_DIR="D:/opencv/build" -A x64
cmake --build build_native --config Release

# After build, copy the produced PaddleSegInterence.dll to the published .NET folder:
copy build_native\Release\PaddleSegInterence.dll publish\win-x64\
```

Linux (GCC/Clang):

```bash
cmake -S nativesrc -B build_native -D PADDLE_DIR=/opt/paddle/inference -D OpenCV_DIR=/usr/local/opencv -DCMAKE_BUILD_TYPE=Release
cmake --build build_native --config Release -j$(nproc)

# Copy the shared object to the publish folder or add its directory to LD_LIBRARY_PATH
cp build_native/libPaddleSegInterence.so publish/linux-x64/
```

Notes:

- Set `PADDLE_DIR` to your Paddle Inference SDK root (the folder that contains `include/` and `lib/`).
- Set `OpenCV_DIR` to your OpenCV build folder.
- If you use non-standard locations for CUDA/CUDNN/TensorRT, ensure those are discoverable by the linker and available at runtime.
- The produced library names are `PaddleSegInterence.dll` (Windows) and `libPaddleSegInterence.so` (Linux). Place the appropriate binary next to the published `PaddleSegCli.exe` (or add its folder to `PATH` / `LD_LIBRARY_PATH`).


## Command-line Usage

After publishing, the CLI is available in `publish/win-x64` (Windows) or the corresponding publish folder for your runtime. The executable name is `PaddleSegCli.exe`.

Basic usage (run from the publish folder):

```powershell
.\PaddleSegCli.exe mask -m <modelDir> -i <input.png> -o <output.png> [--enable-gpu --gpu-mem-size N --cpu-threads N --enable-onednn --enable-onnxruntime]
.\PaddleSegCli.exe apply -m <modelDir> -i <input.png> -o <output.png> --bg-color R,G,B
.\PaddleSegCli.exe apply -m <modelDir> -i <input.png> -o <output.png> --bg-image <path-to-image>
```

Examples:

- Solid color background (black):

```powershell
.\PaddleSegCli.exe apply -m "models/pp_matting" -i "in.png" -o "out.png" --bg-color 0,0,0
```

- Image background:

```powershell
.\PaddleSegCli.exe apply -m "models/pp_matting" -i "in.png" -o "out.png" --bg-image "C:\path\to\bg.jpg"
```

Notes:

- `modelDir` must contain `model.pdmodel` and `model.pdiparams` exported from Paddle.
- If you built a CPU-only native binary or you don't have GPU Paddle runtime DLLs available, ensure you copied the CPU Paddle runtime DLLs into the publish folder (use `copy-dlls.ps1`).
- On Linux adjust the executable path (for example `./publish/linux-x64/PaddleSegCli`) and ensure shared objects are discoverable via `LD_LIBRARY_PATH`.



## Native API

- `seg_init(const char* model_dir, int enable_use_gpu, int gpu_mem_size, int cpu_thread_num, int enable_onednn, int enable_onnxruntime)`

Initialization parameters (via `seg_init`):

- `enable_use_gpu` (int, 0/1) — enable GPU runtime when set to 1. Default: 0 (CPU).
- `gpu_mem_size` (int) — GPU memory to reserve in MB when enabling GPU. Default: 100.
- `cpu_thread_num` (int) — number of threads for CPU math library when applicable. Default: 20 (recommended; set 0 to use runtime default).
- `enable_onednn` (int, 0/1) — enable oneDNN optimizations (if Paddle build supports it). Default: 1 (enabled).
- `enable_onnxruntime` (int, 0/1) — enable ONNXRuntime integration (if available). Default: 0 (disabled).

When using the .NET `PaddleEngine` via JSON config, the equivalent keys are:

- `enableUseGpu` (bool)
- `gpuMemSize` (int)
- `cpuThreadNum` (int)
- `enableOneDnn` (bool)
- `enableOnnxRuntime` (bool)

- `seg_infer(const unsigned char* rgba, int width, int height, unsigned char* out_mask)`
- `seg_set_background_rgb(unsigned char r, unsigned char g, unsigned char b)`
- `seg_set_background_image(const char* image_path)`
- `seg_apply_background(const unsigned char* src_rgba, int width, int height, unsigned char* out_rgba)`
- `seg_release()`

## Preprocessing Notes

The default pipeline converts RGBA→BGR and normalizes to [0,1]. Many PaddleSeg models require resize and mean/std normalization. Adjust `rgba_to_bgr_float` and input reshape in [nativesrc/src/paddleseg_api.cpp](nativesrc/src/paddleseg_api.cpp) to match your export.

## GPU/CPU

Current code enables GPU via `config.EnableUseGpu(100, 0)`. For CPU-only, replace with `config.DisableGpu()` and ensure CPU runtime libraries are on PATH.

## Troubleshooting

- CMake not found: Install CMake or start from a VS Developer PowerShell where CMake is on PATH.
- MSBuild errors about `VCTargetsPath`: Install Visual Studio C++ Build Tools (Desktop development with C++ workload) and retry configure/build.
- Native DLL not loading in .NET: Ensure Paddle/OpenCV dependencies are next to the .NET exe or on PATH.

## Deploying Native DLLs

- Copy `build_native/Release/PaddleSegInterence.dll` (Windows) or `build_native/Release/libPaddleSegInterence.so` (Linux) plus required Paddle Inference and OpenCV runtime shared libraries into the same folder as your .NET executable, or add their folders to `PATH`.
- For CPU-only builds, ensure the CPU runtime DLLs from Paddle are present; for GPU builds, also copy CUDA/TensorRT-related Paddle DLLs as required by your SDK package.

## License

This repo contains integration scaffolding and expects you to use your own exported Paddle models; follow Paddle licenses for the SDK and model assets.
