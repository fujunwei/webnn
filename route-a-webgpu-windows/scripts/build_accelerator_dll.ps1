# Build the Chromium-libc++-ABI-compatible LiteRtWebGpuAccelerator.dll.
#
# Prereqs (see README §Setup):
#   1. Chromium checkout built once in Release
#   2. libc++.lib packaged via scripts/build_libcxx_lib.ps1
#   3. Dawn prebuilt tree via scripts/build_dawn.ps1
#   4. All patches applied to $LITERT and $ML_DRIFT (see README §Apply patches)
#   5. First-time-only: run scripts/patch_bazel_toolchain.py after the FIRST
#      failed build (which will have created the local_config_cc/BUILD file).

param(
    [ValidateSet("dbg","opt")] [string]$Mode = "dbg",
    [string]$ChromiumSrc,   # default: %WORKSPACE_ROOT%\chromium\src  (or $env:CHROMIUM_SRC)
    [string]$WebnnDir,      # default: dir containing this bundle           (or $env:WEB_NN)
    [string]$MlDrift,       # default: %CHROMIUM_SRC%\third_party\ml-drift   (or $env:ML_DRIFT_DIR)
    # Git-for-Windows bash.exe used as Bazel's --shell_executable. Auto-detected
    # when empty (machine-wide install, then per-user under %LOCALAPPDATA%).
    [string]$Bash        = ""
)

. "$PSScriptRoot\common.ps1"

$ChromiumSrc = Resolve-BundlePath $ChromiumSrc "CHROMIUM_SRC" $DefaultChromiumSrc
$WebnnDir    = Resolve-BundlePath $WebnnDir "WEB_NN" $WebnnRoot
$MlDrift     = Resolve-BundlePath $MlDrift "ML_DRIFT_DIR" $DefaultMlDrift

$ErrorActionPreference = "Continue"

# Bazel fetches external repos (FP16, XNNPACK, farmhash, ...) with its own Java
# HTTP client, which reads HTTP_PROXY/HTTPS_PROXY and ignores the WinINET proxy
# that git and browsers use. On a corporate network without these set, every
# http_archive dies with "Connect timed out". Seed them from the WinINET
# settings when the caller has not already provided them.
if (-not $env:HTTPS_PROXY) {
    $ie = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction SilentlyContinue
    if ($ie.ProxyEnable -eq 1 -and $ie.ProxyServer) {
        # ProxyServer may be "host:port" or "http=host:port;https=host:port".
        $p = $ie.ProxyServer
        if ($p -match "https?=([^;]+)") { $p = $Matches[1] }
        if ($p -notmatch "^https?://") { $p = "http://$p" }
        $env:HTTP_PROXY = $p
        $env:HTTPS_PROXY = $p
        Write-Host "Using proxy from WinINET settings: $p"
    }
}
if ($env:HTTPS_PROXY -and -not $env:NO_PROXY) {
    $env:NO_PROXY = "localhost,127.0.0.1"
}

$LITERT = Join-Path $ChromiumSrc "third_party\litert\src"
if (-not (Test-Path $LITERT)) { throw "litert src not found at $LITERT" }

# Shell INCLUDE must contain libc++ header dirs so windows_cc_configure.bzl
# picks them up during Bazel's cc_toolchain auto-detection. (Any MSVC INCLUDE
# paths from vcvars are appended by Bazel automatically.)
$env:INCLUDE = @(
    (Join-Path $ChromiumSrc "third_party\libc++\src\include"),
    (Join-Path $ChromiumSrc "third_party\libc++abi\src\include"),
    (Join-Path $ChromiumSrc "buildtools\third_party\libc++")
) -join ";"
$env:BAZEL_LLVM    = (Join-Path $ChromiumSrc "third_party\llvm-build\Release+Asserts").Replace("\","/")
$env:USE_CLANG_CL  = "1"
$env:DAWN_PREBUILT_DIR = (Join-Path $WebnnDir "_dawn_prebuilt_win")

# Some litert/ml_drift genrules run through a shell. Bazel's default on Windows
# is a bash that may not exist, so point it at Git-for-Windows' bash. Git may be
# installed machine-wide or per-user (depot_tools bootstraps the latter), so
# probe instead of hardcoding.
if (-not $Bash) {
    $Bash = @(
        "C:\Program Files\Git\bin\bash.exe",
        "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe",
        "C:\Program Files (x86)\Git\bin\bash.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Bash) {
    throw "bash.exe not found. Install Git for Windows or pass -Bash <path to bash.exe>."
}
Write-Host "bash: $Bash"

Set-Location $LITERT
Write-Host "Building //litert/runtime/accelerators/gpu:libLiteRtWebGpuAccelerator (-c $Mode)"
Write-Host "  Chromium src: $ChromiumSrc"
Write-Host "  ml-drift:     $MlDrift"
Write-Host "  DAWN dir:     $env:DAWN_PREBUILT_DIR"
Write-Host ""

$logPath = Join-Path $WebnnDir "build_crcxx_$Mode.log"
# NOTE: ml_drift is wired via local_repository in WORKSPACE (with repo_mapping
# @fp16 -> @FP16). Do NOT pass --override_repository here: it would replace the
# local_repository definition and drop the repo_mapping, breaking @fp16 deps.
bazel build --config=windows --config=crcxx_win --check_visibility=false `
    "--shell_executable=$($Bash.Replace('\','/'))" `
    -c $Mode `
    //litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dll 2>&1 |
    Tee-Object -FilePath $logPath

$rc = $LASTEXITCODE
Write-Host "Bazel exit code: $rc  (log: $logPath)"
if ($rc -ne 0) {
    Write-Host "Hint: if this is the first-ever build attempt, run scripts\patch_bazel_toolchain.py"
    Write-Host "      then re-run this script. See README §Setup for details."
}
exit $rc
