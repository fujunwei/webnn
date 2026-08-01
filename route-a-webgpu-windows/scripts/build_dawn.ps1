# Build webgpu-dawn-binaries monolithic Dawn DLL matching the local Chrome version.
#
# Output layout (in $Dest):
#   include/  — dawn/, webgpu/ headers (webgpu.h, webgpu_cpp.h, etc.)
#   lib/      — webgpu_dawn.dll + webgpu_dawn.lib (import lib)
#
# Prereqs:
#   - Visual Studio 2022 (or newer) with C++ workload
#   - CMake >= 3.20 in PATH
#   - Python 3.x in PATH
#   - Git in PATH
#
# Notes:
#   - Building Dawn from scratch takes ~30-60 minutes.
#   - $ChromiumVersion should match the Chrome build the DLL will be loaded into.

param(
    [string]$SrcDir           = "C:\Users\junweifu\workspace\webnn\_webgpu_dawn_src",
    [string]$Dest             = "C:\Users\junweifu\workspace\webnn\_dawn_prebuilt_win",
    [string]$ChromiumVersion  = "147.0.7714.2"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SrcDir)) {
    Write-Host "Cloning webgpu-dawn-binaries into $SrcDir..."
    git clone https://github.com/jspanchu/webgpu-dawn-binaries.git $SrcDir
}

Set-Content -Path (Join-Path $SrcDir "chromium_version.txt") -Value $ChromiumVersion -NoNewline
Write-Host "Set chromium_version.txt = $ChromiumVersion"

$BuildDir = Join-Path $SrcDir "out\latest"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Push-Location $BuildDir
try {
    Write-Host "Configuring CMake (monolithic shared DLL)..."
    # DAWN_BUILD_MONOLITHIC_LIBRARY=SHARED produces a single webgpu_dawn.dll
    # exporting all wgpu* C symbols (300+).
    #
    # Notes:
    #  - CMAKE_BUILD_TYPE is IGNORED by VS multi-config generator; we must pass
    #    --config Release at build time (below).
    #  - Use -U (unset) on both cache vars first so that re-running this script
    #    doesn't inherit a stale STATIC value from a prior configure attempt.
    cmake ..\.. `
        -UDAWN_BUILD_MONOLITHIC_LIBRARY `
        -UCMAKE_BUILD_TYPE `
        -DCMAKE_BUILD_TYPE=Release `
        -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    Write-Host "Building Dawn Release (this takes 30-60 min)..."
    cmake --build . --config Release
    if ($LASTEXITCODE -ne 0) { throw "cmake --build failed" }
} finally {
    Pop-Location
}

# Layout output tree
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "include") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "lib")     | Out-Null

# Copy headers from Dawn source + generated
$srcHdrRoot = Join-Path $BuildDir "_deps\dawn-src\include"
$genHdrRoot = Join-Path $BuildDir "_deps\dawn-build\gen\include"
foreach ($sub in @("dawn", "webgpu")) {
    foreach ($root in @($srcHdrRoot, $genHdrRoot)) {
        $src = Join-Path $root $sub
        if (Test-Path $src) {
            Copy-Item -Recurse -Force $src (Join-Path $Dest "include")
        }
    }
}

# Find + copy DLL and import lib.
# CRITICAL: with the VS multi-config generator, both Debug\ and Release\ subdirs
# may contain a webgpu_dawn.dll (if the developer ever built Debug). We MUST
# pick the Release build — Debug is ~5x larger and ~10x slower at runtime.
# Restrict search to explicit Release paths.
$candidateDirs = @(
    (Join-Path $BuildDir "_deps\dawn-build\Release"),
    (Join-Path $BuildDir "bin\Release")
)
$dll = $null
$lib = $null
foreach ($d in $candidateDirs) {
    if (-not $dll -and (Test-Path (Join-Path $d "webgpu_dawn.dll"))) {
        $dll = Get-Item (Join-Path $d "webgpu_dawn.dll")
    }
}
# Import lib is usually placed under <BuildDir>\lib\Release\
$libCandidates = @(
    (Join-Path $BuildDir "lib\Release\webgpu_dawn.lib"),
    (Join-Path $BuildDir "_deps\dawn-build\Release\webgpu_dawn.lib")
)
foreach ($p in $libCandidates) {
    if (-not $lib -and (Test-Path $p)) { $lib = Get-Item $p }
}
if (-not $dll) { throw "Release webgpu_dawn.dll not found under $BuildDir\_deps\dawn-build\Release or $BuildDir\bin\Release" }
if (-not $lib) { throw "Release webgpu_dawn.lib not found under $BuildDir\lib\Release" }

Copy-Item -Force $dll.FullName (Join-Path $Dest "lib\webgpu_dawn.dll")
Copy-Item -Force $lib.FullName (Join-Path $Dest "lib\webgpu_dawn.lib")

Write-Host ""
Write-Host "OK. Prebuilt Dawn tree at $Dest"
Write-Host "  DLL: $((Get-Item (Join-Path $Dest 'lib\webgpu_dawn.dll')).Length / 1MB) MB"
Write-Host "  LIB: $((Get-Item (Join-Path $Dest 'lib\webgpu_dawn.lib')).Length / 1MB) MB"
Write-Host ""
Write-Host "Set DAWN_PREBUILT_DIR=$Dest before running the litert Bazel build."
