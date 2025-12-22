using System.CommandLine;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Acesoft.PaddleSeg;

class Program {
    static int Main(string[] args) {
        // Create options using the System.CommandLine 2.0.1 constructor overload (aliases passed in ctor)
        var modelOption = new Option<string>("--model", "-m") {
            Description = "Path to the segmentation model directory"
        };

        // Init flags
        var enableGpuOption = new Option<bool>("--enable-gpu") { Description = "Enable GPU runtime (default: false)" };
        var gpuMemOption = new Option<int>("--gpu-mem-size") { Description = "GPU memory (MB) to reserve when enabling GPU (default: 100)" };
        var cpuThreadsOption = new Option<int>("--cpu-threads") { Description = "CPU math library thread count (default: 20)" };
        var enableOneDnnOption = new Option<bool>("--enable-onednn") { Description = "Enable oneDNN optimizations (default: true)" };
        var enableOnnxOption = new Option<bool>("--enable-onnxruntime") { Description = "Enable ONNXRuntime integration (default: false)" };

        var inputOption = new Option<string>("--input", "-i") {
            Description = "Path to the input image file"
        };

        var outputOption = new Option<string>("--output", "-o") {
            Description = "Path to save the output image"
        };

        var bgColorOption = new Option<string>("--bg-color", "-c") {
            Description = "Background color in r,g,b format (e.g., 255,255,255)"
        };

        var bgImageOption = new Option<string>("--bg-image", "-b") {
            Description = "Path to background image file"
        };

        // Create subcommands and register options (use Options collection per 2.0.1)
        var maskCommand = new Command("mask", "Generate segmentation mask from input image");
        maskCommand.Options.Add(modelOption);
        maskCommand.Options.Add(inputOption);
        maskCommand.Options.Add(outputOption);
        maskCommand.Options.Add(enableGpuOption);
        maskCommand.Options.Add(gpuMemOption);
        maskCommand.Options.Add(cpuThreadsOption);
        maskCommand.Options.Add(enableOneDnnOption);
        maskCommand.Options.Add(enableOnnxOption);

        var applyCommand = new Command("apply", "Apply background to segmentation result");
        applyCommand.Options.Add(modelOption);
        applyCommand.Options.Add(inputOption);
        applyCommand.Options.Add(outputOption);
        applyCommand.Options.Add(bgColorOption);
        applyCommand.Options.Add(bgImageOption);
        applyCommand.Options.Add(enableGpuOption);
        applyCommand.Options.Add(gpuMemOption);
        applyCommand.Options.Add(cpuThreadsOption);
        applyCommand.Options.Add(enableOneDnnOption);
        applyCommand.Options.Add(enableOnnxOption);

        // Create root command and register subcommands
        var rootCommand = new RootCommand("Image segmentation tool");
        rootCommand.Subcommands.Add(maskCommand);
        rootCommand.Subcommands.Add(applyCommand);

        // Bind actions using SetAction(parseResult => ...)
            maskCommand.SetAction(parseResult => {
            var model = parseResult.GetValue(modelOption);
            var input = parseResult.GetValue(inputOption);
            var output = parseResult.GetValue(outputOption);
            var enableGpu = parseResult.GetValue(enableGpuOption);
            var gpuMem = parseResult.GetValue(gpuMemOption);
            var cpuThreads = parseResult.GetValue(cpuThreadsOption);
            var enableOneDnn = parseResult.GetValue(enableOneDnnOption);
            var enableOnnx = parseResult.GetValue(enableOnnxOption);

            if (string.IsNullOrWhiteSpace(model)) { Console.Error.WriteLine("--model is required"); return 2; }
            if (string.IsNullOrWhiteSpace(input)) { Console.Error.WriteLine("--input is required"); return 2; }
            if (string.IsNullOrWhiteSpace(output)) { Console.Error.WriteLine("--output is required"); return 2; }

            return GenerateMask(model, input, output, enableGpu, gpuMem, cpuThreads, enableOneDnn, enableOnnx);
        });

        applyCommand.SetAction(parseResult => {
            var model = parseResult.GetValue(modelOption);
            var input = parseResult.GetValue(inputOption);
            var output = parseResult.GetValue(outputOption);
            var bgColor = parseResult.GetValue(bgColorOption);
            var bgImage = parseResult.GetValue(bgImageOption);
            var enableGpu = parseResult.GetValue(enableGpuOption);
            var gpuMem = parseResult.GetValue(gpuMemOption);
            var cpuThreads = parseResult.GetValue(cpuThreadsOption);
            var enableOneDnn = parseResult.GetValue(enableOneDnnOption);
            var enableOnnx = parseResult.GetValue(enableOnnxOption);

            if (string.IsNullOrWhiteSpace(model)) { Console.Error.WriteLine("--model is required"); return 2; }
            if (string.IsNullOrWhiteSpace(input)) { Console.Error.WriteLine("--input is required"); return 2; }
            if (string.IsNullOrWhiteSpace(output)) { Console.Error.WriteLine("--output is required"); return 2; }

            return ApplyBackground(model, input, output, bgColor, bgImage, enableGpu, gpuMem, cpuThreads, enableOneDnn, enableOnnx);
        });

        // Parse and invoke (2.0.1 style)
        return rootCommand.Parse(args).Invoke();
    }

    private static int GenerateMask(string model, string input, string output, bool enableGpu, int gpuMem, int cpuThreads, bool enableOneDnn, bool enableOnnx) {
        try {
            Console.WriteLine($"Generating mask: {input} -> {output}");
            Console.WriteLine($"Using model: {model}");

            var config = new JsonObject
            {
                ["type"] = "paddleseg",
                ["enableUseGpu"] = enableGpu,
                ["gpuMemSize"] = gpuMem,
                ["cpuThreadNum"] = cpuThreads,
                ["enableOneDnn"] = enableOneDnn,
                ["enableOnnxRuntime"] = enableOnnx
            };

            var engine = new PaddleSegEngine();
            try
            {
                engine.Initialize(model, config);
                var rgba = engine.SegmentImage(input);
                engine.LoadRgba(input, out int width, out int height);
                engine.SaveRgba(output, rgba, width, height);
            }
            finally
            {
                engine.Release();
            }

            Console.WriteLine($"Successfully saved mask: {output}");
            return 0;
        } catch (Exception ex) {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            return 5;
        }
    }

    private static int ApplyBackground(string model, string input, string output, string? bgColor, string? bgImage, bool enableGpu, int gpuMem, int cpuThreads, bool enableOneDnn, bool enableOnnx) {
        try {
            Console.WriteLine($"Applying background: {input} -> {output}");
            Console.WriteLine($"Using model: {model}");
            var config = new JsonObject
            {
                ["type"] = "paddleseg",
                ["enableUseGpu"] = enableGpu,
                ["gpuMemSize"] = gpuMem,
                ["cpuThreadNum"] = cpuThreads,
                ["enableOneDnn"] = enableOneDnn,
                ["enableOnnxRuntime"] = enableOnnx
            };
            var engine = new PaddleSegEngine();
            try
            {
                engine.Initialize(model, config);
                byte[] outRgba;
                if (!string.IsNullOrEmpty(bgColor))
                {
                    var parts = bgColor.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 3 &&
                        byte.TryParse(parts[0], out byte r) &&
                        byte.TryParse(parts[1], out byte g) &&
                        byte.TryParse(parts[2], out byte b))
                    {
                        outRgba = engine.ChangeBackColor(input, r, g, b);
                        Console.WriteLine($"Using background color: {r},{g},{b}");
                    }
                    else
                    {
                        Console.WriteLine("Invalid --bg-color format. Use r,g,b (e.g., 255,255,255)");
                        return 6;
                    }
                }
                else if (!string.IsNullOrEmpty(bgImage))
                {
                    if (File.Exists(bgImage))
                    {
                        outRgba = engine.ChangeBackImage(input, bgImage);
                        Console.WriteLine($"Using background image: {bgImage}");
                    }
                    else
                    {
                        Console.WriteLine($"Error: Background image not found: {bgImage}");
                        return 7;
                    }
                }
                else
                {
                    Console.WriteLine("Warning: No background specified. Using default background.");
                    outRgba = engine.ChangeBackColor(input, 0, 0, 0);
                }

                engine.LoadRgba(input, out int width, out int height);
                engine.SaveRgba(output, outRgba, width, height);

                Console.WriteLine($"Successfully saved result: {output}");
                return 0;
            }
            finally
            {
                engine.Release();
            }
        } catch (Exception ex) {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            return 5;
        }
    }

    // no-op cleanup: engine and Interop handle native resources; explicit release called after operations
}