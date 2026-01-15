using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Text.Json.Nodes;

namespace Acesoft.PaddleSeg;

public abstract class PaddleEngine
{
    // Allow derived classes to construct without immediately initializing native runtime.
    protected PaddleEngine() { }

    // Shared initialization logic extracted so derived classes can call it when ready.
    protected void InitializeNative(string modelDir, JsonObject modelConfig)
    {
        bool enableUseGpu = false; // default false
        int cpuThreadNum = 20;
        int gpuMemSize = 100;
        bool enableOneDnn = false; // default false: disable oneDNN optimizations by default
        bool enableOnnxRuntime = false;

        if (modelConfig != null)
        {
            try { enableUseGpu = modelConfig["enableUseGpu"]?.GetValue<bool>() ?? enableUseGpu; } catch { }
            try { cpuThreadNum = modelConfig["cpuThreadNum"]?.GetValue<int>() ?? cpuThreadNum; } catch { }
            try { gpuMemSize = modelConfig["gpuMemSize"]?.GetValue<int>() ?? gpuMemSize; } catch { }
            try { enableOneDnn = modelConfig["enableOneDnn"]?.GetValue<bool>() ?? enableOneDnn; } catch { }
            try { enableOnnxRuntime = modelConfig["enableOnnxRuntime"]?.GetValue<bool>() ?? enableOnnxRuntime; } catch { }
        }

        var ret = Interop.Initialize(modelDir, enableUseGpu, gpuMemSize, cpuThreadNum, enableOneDnn, enableOnnxRuntime);
        if (ret != 0)
        {
            throw new InvalidOperationException($"Failed to initialize PaddleSeg engine. Error code: {ret}");
        }
    }

    protected PaddleEngine(string modelDir, JsonObject modelConfig)
    {
        InitializeNative(modelDir, modelConfig);
    }

    // Helper: load image file into RGBA byte array (R,G,B,A)
    public byte[] LoadRgba(string path, out int width, out int height)
    {
        using var img = Image.Load<Rgba32>(path);
        width = img.Width;
        height = img.Height;
        int count = width * height;
        var pixels = new Rgba32[count];
        img.CopyPixelDataTo(pixels);
        byte[] data = new byte[count * 4];
        for (int i = 0; i < count; i++)
        {
            data[i * 4 + 0] = pixels[i].R;
            data[i * 4 + 1] = pixels[i].G;
            data[i * 4 + 2] = pixels[i].B;
            data[i * 4 + 3] = pixels[i].A;
        }
        return data;
    }

    // Helper: save RGBA byte array to PNG
    public void SaveRgba(string path, byte[] rgba, int width, int height)
    {
        int count = width * height;
        var pixels = new Rgba32[count];
        for (int i = 0; i < count; i++)
        {
            int idx = i * 4;
            pixels[i] = new Rgba32(rgba[idx + 0], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]);
        }
        using var img = Image.LoadPixelData<Rgba32>(pixels, width, height);
        img.SaveAsPng(path);
    }

    ~PaddleEngine()
    {
        Interop.Release();
    }
}
