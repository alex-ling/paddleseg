<#
copy-dlls.ps1
Usage: run from an elevated PowerShell or Developer PowerShell.
It will copy Paddle Inference and OpenCV runtime DLLs into the .NET publish folder.
Adjust the variables below if your SDK / OpenCV paths differ.
#>

param(
  [string]$PublishDir = "",
  [string]$PaddleLibDir = "D:\paddle\inference\paddle\lib",
  [string]$OpenCvBinDir = "D:\opencv\build\x64\vc16\bin",
  [string]$NativeBuildReleaseDir = "",
  [switch]$CopyNativeInterop,
  [switch]$VerboseCopy
)

if (-not $PublishDir) { $PublishDir = Join-Path $PSScriptRoot "publish\win-x64" }
if (-not $NativeBuildReleaseDir) { $NativeBuildReleaseDir = Join-Path $PSScriptRoot "build_native\Release" }

Write-Host "PublishDir: $PublishDir"
Write-Host "PaddleLibDir: $PaddleLibDir"
Write-Host "OpenCvBinDir: $OpenCvBinDir"
Write-Host "NativeBuildReleaseDir: $NativeBuildReleaseDir"

# Derive third-party dirs relative to PaddleLibDir (two levels up)
try {
  $paddleLibResolved = Resolve-Path -Path $PaddleLibDir -ErrorAction Stop
  $paddleLibItem = Get-Item $paddleLibResolved
  $paddleRoot = $paddleLibItem.Parent.Parent.FullName
  $PaddleThirdPartyMkl = Join-Path $paddleRoot "third_party\install\mklml\lib"
  $PaddleThirdPartyOneDNN = Join-Path $paddleRoot "third_party\install\onednn\lib"
} catch {
  Write-Warning "Could not resolve third-party dirs from PaddleLibDir; falling back to defaults if present in params."
  $PaddleThirdPartyMkl = ""
  $PaddleThirdPartyOneDNN = ""
}

Write-Host "PaddleThirdPartyMkl: $PaddleThirdPartyMkl"
Write-Host "PaddleThirdPartyOneDNN: $PaddleThirdPartyOneDNN"

# Ensure publish directory exists
if (-not (Test-Path $PublishDir)) { New-Item -ItemType Directory -Path $PublishDir -Force | Out-Null }

# List of common DLLs to prefer-copy first
$preferred = @(
  'paddle_inference.dll',
  'common.dll',
  'opencv_world4120.dll'
)

foreach ($dll in $preferred) {
  $found = $null
  $p1 = Join-Path $PaddleLibDir $dll
  $p2 = Join-Path $OpenCvBinDir $dll
  if (Test-Path $p1) { $found = $p1 }
  elseif (Test-Path $p2) { $found = $p2 }

  if ($found) {
    Copy-Item -Path $found -Destination $PublishDir -Force -ErrorAction SilentlyContinue
    if ($VerboseCopy) { Write-Host "Copied $found -> $PublishDir" }
  } else {
    Write-Warning "$dll not found in PaddleLibDir or OpenCvBinDir"
  }
}

# Copy all DLLs from Paddle lib and OpenCV bin (non-recursive)
if (Test-Path $PaddleLibDir) {
  Get-ChildItem -Path $PaddleLibDir -Filter *.dll -File -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $PublishDir -Force -ErrorAction SilentlyContinue
    if ($VerboseCopy) { Write-Host "Copied $_.FullName -> $PublishDir" }
  }
} else {
  Write-Warning "PaddleLibDir not found: $PaddleLibDir"
}

# Copy third-party Paddle DLLs (mklml, onednn)
if (Test-Path $PaddleThirdPartyMkl) {
  Get-ChildItem -Path $PaddleThirdPartyMkl -Filter *.dll -File -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $PublishDir -Force
    if ($VerboseCopy) { Write-Host "Copied $_.FullName -> $PublishDir" }
  }
} else {
  Write-Warning "PaddleThirdPartyMkl not found: $PaddleThirdPartyMkl"
}

if (Test-Path $PaddleThirdPartyOneDNN) {
  Get-ChildItem -Path $PaddleThirdPartyOneDNN -Filter *.dll -File -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $PublishDir -Force -ErrorAction SilentlyContinue
    if ($VerboseCopy) { Write-Host "Copied $_.FullName -> $PublishDir" }
  }
} else {
  Write-Warning "PaddleThirdPartyOneDNN not found: $PaddleThirdPartyOneDNN"
}

if (Test-Path $OpenCvBinDir) {
  Get-ChildItem -Path $OpenCvBinDir -Filter *.dll -File -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $PublishDir -Force -ErrorAction SilentlyContinue
    if ($VerboseCopy) { Write-Host "Copied $_.FullName -> $PublishDir" }
  }
} else {
  Write-Warning "OpenCvBinDir not found: $OpenCvBinDir"
}

# Optionally copy the native interop DLL produced by the native build
if ($CopyNativeInterop) {
  $interopName = "PaddleSegInterence.dll"
  $interopPath = Join-Path $NativeBuildReleaseDir $interopName
  if (Test-Path $interopPath) {
    Copy-Item -Path $interopPath -Destination $PublishDir -Force -ErrorAction SilentlyContinue
    Write-Host "Copied native interop $interopPath -> $PublishDir"
  } else {
    Write-Warning "Native interop not found at $interopPath"
  }
}

Write-Host "DLL copy finished. Verify the publish folder now contains Paddle/OpenCV runtime DLLs."