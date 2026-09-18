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
    [string]$SrcDir           = "C:\Users\awx_localadmin\workspace\webnn\_webgpu_dawn_src",
    [string]$Dest             = "C:\Users\awx_localadmin\workspace\webnn\_dawn_prebuilt_win",
    [string]$ChromiumVersion  = "155.0.8054.0",
    # Directory containing a python3.exe (depot_tools' bootstrapped CPython works).
    # A `python.exe` alias is created next to it if missing, because Dawn's CMake
    # scripts and some generators invoke bare `python`.
    [string]$PythonDir        = "",
    [string]$DepotTools       = "C:\Users\junwei\workspace\depot_tools"
)

$ErrorActionPreference = "Stop"

# Refresh PATH from the registry (Machine + User) so freshly installed tools
# (cmake, python, git) are found even when the invoking shell has a stale env.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

# ---- Python ----
# Windows ships a `python.exe` App Execution Alias that only opens the Store, and
# depot_tools bootstraps CPython as `python3.exe` with no `python.exe` next to it.
# Dawn's CMake needs a real interpreter under both names, so resolve one and put
# it first on PATH.
if (-not $PythonDir) {
    $PythonDir = Get-ChildItem $DepotTools -Filter "bootstrap-*_bin" -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "python3\bin" } |
        Where-Object { Test-Path (Join-Path $_ "python3.exe") } |
        Select-Object -First 1
}
if ($PythonDir) {
    if (-not (Test-Path (Join-Path $PythonDir "python.exe"))) {
        Copy-Item (Join-Path $PythonDir "python3.exe") (Join-Path $PythonDir "python.exe")
        Write-Host "Created python.exe alias in $PythonDir"
    }
    $env:Path = "$PythonDir;$env:Path"
}
Write-Host "python: $((Get-Command python -ErrorAction SilentlyContinue).Source)"

if (-not (Test-Path $SrcDir)) {
    Write-Host "Cloning webgpu-dawn-binaries into $SrcDir..."
    # git/cmake write progress to stderr. Under $ErrorActionPreference='Stop'
    # PowerShell promotes native-command stderr to a terminating NativeCommandError
    # even on exit code 0, so drop to 'Continue' and gate on $LASTEXITCODE instead.
    $ErrorActionPreference = "Continue"
    git clone https://github.com/jspanchu/webgpu-dawn-binaries.git $SrcDir
    $rc = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($rc -ne 0) { throw "git clone failed (exit $rc)" }
}

# webgpu-dawn-binaries' CMakeLists derives the Dawn ref from the BUILD field of
# chromium_version.txt: `GIT_TAG chromium/<BUILD>`. Dawn only creates a
# `chromium/<N>` branch once that Chromium milestone is cut, so a bleeding-edge
# Chromium main checkout is routinely 1-2 builds ahead of the newest Dawn branch
# and the clone dies with `fatal: invalid reference: chromium/<BUILD>`.
# Fall back to the highest published branch <= our BUILD.
$verParts = $ChromiumVersion.Split(".")
$wantBuild = [int]$verParts[2]
$ErrorActionPreference = "Continue"
$branches = git ls-remote --heads https://dawn.googlesource.com/dawn "refs/heads/chromium/*" 2>$null |
    ForEach-Object { ($_ -split "\s+")[1] -replace "refs/heads/chromium/", "" } |
    Where-Object { $_ -match "^\d+$" } | ForEach-Object { [int]$_ }
$ErrorActionPreference = "Stop"
if ($branches) {
    $usable = $branches | Where-Object { $_ -le $wantBuild } | Sort-Object | Select-Object -Last 1
    if (-not $usable) { throw "No dawn chromium/<N> branch at or below $wantBuild" }
    if ($usable -ne $wantBuild) {
        Write-Host "Dawn has no chromium/$wantBuild branch; using chromium/$usable (newest <= $wantBuild)."
        $verParts[2] = "$usable"
        $ChromiumVersion = $verParts -join "."
    }
} else {
    Write-Host "WARNING: could not list dawn branches; using $ChromiumVersion as-is."
}

Set-Content -Path (Join-Path $SrcDir "chromium_version.txt") -Value $ChromiumVersion -NoNewline
Write-Host "Set chromium_version.txt = $ChromiumVersion"

$BuildDir = Join-Path $SrcDir "out\latest"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Push-Location $BuildDir
try {
    # See the note above: cmake streams status to stderr, so keep native-command
    # stderr non-terminating and rely on $LASTEXITCODE for failure detection.
    $ErrorActionPreference = "Continue"
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
    $ErrorActionPreference = "Stop"
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

# Under -define=ml_drift_use_dawn_proc=true (README.md SS1.4b), the accelerator
# links only this thin proc-table trampoline, not lib\webgpu_dawn.dll above. It
# can't be the plain generated src/dawn/dawn_proc.cpp, though: that file pulls
# in Dawn's internal (non-public) headers src/dawn/common/Compiler.h,
# src/utils/assert.h and src/utils/log.h, none of which ship in this include
# tree. vendor/dawn_proc.cpp is a one-time hand-patched copy (see its own
# header comment) with those three replaced by trivial local equivalents; copy
# it in on every run so regenerating the prebuilt tree on a new machine can't
# silently lose it.
$vendoredProcCpp = Join-Path $PSScriptRoot "..\vendor\dawn_proc.cpp"
if (-not (Test-Path $vendoredProcCpp)) {
    throw "Missing vendored file: $vendoredProcCpp (see README.md section 1.4b)."
}
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "src") | Out-Null
Copy-Item -Force $vendoredProcCpp (Join-Path $Dest "src\dawn_proc.cpp")
Write-Host "  src\dawn_proc.cpp: vendored from $vendoredProcCpp"
Write-Host ""
Write-Host "Set DAWN_PREBUILT_DIR=$Dest before running the litert Bazel build."
