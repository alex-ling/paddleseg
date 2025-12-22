using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.InteropServices;

namespace Acesoft.PaddleSeg;

internal static class Interop
{
    // select native library name per OS (matches CMake target PaddleSegInterence)
    private static readonly string DLL = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "PaddleSegInterence.dll" : "libPaddleSegInterence.so";

    [DllImport("__Internal", EntryPoint = "seg_init", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
    private static extern int seg_init([MarshalAs(UnmanagedType.LPStr)] string model_dir, int enableUseGpu, int gpuMemSize, int cpuThreadNum, int enableOneDnn, int enableOnnxRuntime);

    [DllImport("__Internal", EntryPoint = "seg_infer", CallingConvention = CallingConvention.Cdecl)]
    private static extern int seg_infer(IntPtr rgba, int width, int height, IntPtr out_mask);

    [DllImport("__Internal", EntryPoint = "seg_set_background_rgb", CallingConvention = CallingConvention.Cdecl)]
    private static extern int seg_set_background_rgb(byte r, byte g, byte b);

    [DllImport("__Internal", EntryPoint = "seg_set_background_image", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
    private static extern int seg_set_background_image([MarshalAs(UnmanagedType.LPStr)] string image_path);

    [DllImport("__Internal", EntryPoint = "seg_apply_background", CallingConvention = CallingConvention.Cdecl)]
    private static extern int seg_apply_background(IntPtr src_rgba, int width, int height, IntPtr out_rgba);

    [DllImport("__Internal", EntryPoint = "seg_release", CallingConvention = CallingConvention.Cdecl)]
    private static extern void seg_release();

    static Interop()
    {
        NativeLibrary.SetDllImportResolver(typeof(Interop).Assembly, Resolver);
    }

    private static IntPtr Resolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "__Internal")
        {
            // Try default search first
            if (NativeLibrary.TryLoad(DLL, assembly, searchPath, out IntPtr handle))
            {
                return handle;
            }
        }
        return IntPtr.Zero;
    }

    // Public wrappers
    public static int Initialize(string modelDir, bool enableUseGpu = false, int gpuMemSize = 100, int cpuThreadNum = 20, bool enableOneDnn = false, bool enableOnnxRuntime = false)
        => seg_init(modelDir, enableUseGpu ? 1 : 0, gpuMemSize, cpuThreadNum, enableOneDnn ? 1 : 0, enableOnnxRuntime ? 1 : 0);
    public static int Infer(IntPtr rgba, int width, int height, IntPtr outMask) => seg_infer(rgba, width, height, outMask);
    public static int SetBackgroundRgb(byte r, byte g, byte b) => seg_set_background_rgb(r, g, b);
    public static int SetBackgroundImage(string path) => seg_set_background_image(path);
    public static int ApplyBackground(IntPtr srcRgba, int width, int height, IntPtr outRgba) => seg_apply_background(srcRgba, width, height, outRgba);
    public static void Release() => seg_release();
}
