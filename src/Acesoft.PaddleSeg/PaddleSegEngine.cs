using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Acesoft.PaddleSeg;

public class PaddleSegEngine : PaddleEngine
{
    private bool _initialized;
    private string? _modelDir;
    private JsonObject? _modelConfig;

    public PaddleSegEngine() : base() { }

    public void Initialize(string modelDir, JsonObject? modelConfig)
    {
        if (_initialized) return;
        _modelDir = modelDir;
        _modelConfig = modelConfig ?? new JsonObject();
        InitializeNative(modelDir, _modelConfig);
        _initialized = true;
    }

    public void Release()
    {
        if (!_initialized) return;
        Interop.Release();
        _initialized = false;
    }

    public byte[] ChangeBackColor(string imageSrc, byte r, byte g, byte b)
    {
        if (!_initialized) throw new InvalidOperationException("Engine not initialized. Call Initialize() first.");
        var inRgba = LoadRgba(imageSrc, out int width, out int height);

        int rc = Interop.SetBackgroundRgb(r, g, b);
        if (rc != 0)
        {
            throw new InvalidOperationException($"SetBackgroundRgb failed with code {rc}");
        }

        IntPtr inPtr = Marshal.AllocHGlobal(inRgba.Length);
        IntPtr outPtr = Marshal.AllocHGlobal(inRgba.Length);
        try
        {
            Marshal.Copy(inRgba, 0, inPtr, inRgba.Length);
            rc = Interop.ApplyBackground(inPtr, width, height, outPtr);
            if (rc != 0)
            {
                throw new InvalidOperationException($"ApplyBackground failed with code {rc}");
            }

            var outRgba = new byte[inRgba.Length];
            Marshal.Copy(outPtr, outRgba, 0, outRgba.Length);
            return outRgba;
        }
        finally
        {
            if (inPtr != IntPtr.Zero) Marshal.FreeHGlobal(inPtr);
            if (outPtr != IntPtr.Zero) Marshal.FreeHGlobal(outPtr);
        }
    }

    public byte[] ChangeBackImage(string imageSrc, string imageBgPath)
    {
        if (!_initialized) throw new InvalidOperationException("Engine not initialized. Call Initialize() first.");
        if (!File.Exists(imageBgPath)) throw new FileNotFoundException("Background image not found", imageBgPath);

        var inRgba = LoadRgba(imageSrc, out int width, out int height);

        int rc = Interop.SetBackgroundImage(imageBgPath);
        if (rc != 0)
        {
            throw new InvalidOperationException($"SetBackgroundImage failed with code {rc}");
        }

        IntPtr inPtr = Marshal.AllocHGlobal(inRgba.Length);
        IntPtr outPtr = Marshal.AllocHGlobal(inRgba.Length);
        try
        {
            Marshal.Copy(inRgba, 0, inPtr, inRgba.Length);
            rc = Interop.ApplyBackground(inPtr, width, height, outPtr);
            if (rc != 0)
            {
                throw new InvalidOperationException($"ApplyBackground failed with code {rc}");
            }

            var outRgba = new byte[inRgba.Length];
            Marshal.Copy(outPtr, outRgba, 0, outRgba.Length);
            return outRgba;
        }
        finally
        {
            if (inPtr != IntPtr.Zero) Marshal.FreeHGlobal(inPtr);
            if (outPtr != IntPtr.Zero) Marshal.FreeHGlobal(outPtr);
        }
    }

    public byte[] SegmentImage(string imagePath)
    {
        if (!_initialized) throw new InvalidOperationException("Engine not initialized. Call Initialize() first.");
        var inRgba = LoadRgba(imagePath, out int width, out int height);

        IntPtr inPtr = Marshal.AllocHGlobal(inRgba.Length);
        IntPtr maskPtr = Marshal.AllocHGlobal(width * height);
        try
        {
            Marshal.Copy(inRgba, 0, inPtr, inRgba.Length);
            int rc = Interop.Infer(inPtr, width, height, maskPtr);
            if (rc != 0)
            {
                throw new InvalidOperationException($"Infer failed with code {rc}");
            }

            var mask = new byte[width * height];
            Marshal.Copy(maskPtr, mask, 0, mask.Length);

            // convert mask to RGBA
            var rgba = new byte[mask.Length * 4];
            for (int i = 0; i < mask.Length; i++)
            {
                byte v = mask[i];
                rgba[i * 4 + 0] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = 255;
            }
            return rgba;
        }
        finally
        {
            if (inPtr != IntPtr.Zero) Marshal.FreeHGlobal(inPtr);
            if (maskPtr != IntPtr.Zero) Marshal.FreeHGlobal(maskPtr);
        }
    }
}
