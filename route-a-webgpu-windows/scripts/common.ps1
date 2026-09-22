# Shared path resolution for the route-a-webgpu-windows bundle.
#
# Every directory used by these scripts resolves in this order:
#   1. explicit script parameter  (e.g. -ChromiumSrc D:\chromium\src)
#   2. environment variable       (see table below)
#   3. derived default            (current user profile + this bundle's own location)
#
# Because the defaults are derived from %WORKSPACE_ROOT% (falling back to
# %USERPROFILE%\workspace) and from this file's location (the bundle always
# lives at <WEB_NN>\route-a-webgpu-windows), the same bundle checkout works
# on any machine/account without edits.
#
# Environment variable overrides (only needed for non-standard layouts):
#   WORKSPACE_ROOT      dir containing chromium/, depot_tools/ and webnn/
#                       (default: %USERPROFILE%\workspace)
#   CHROMIUM_SRC        Chromium checkout      (default: %WORKSPACE_ROOT%\chromium\src)
#   WEB_NN              dir containing this bundle (default: two levels above this file)
#   ML_DRIFT_DIR        ml-drift checkout      (default: %CHROMIUM_SRC%\third_party\ml-drift)
#   DEPOT_TOOLS         depot_tools dir        (default: %WORKSPACE_ROOT%\depot_tools)
#   DAWN_PREBUILT_DIR   prebuilt Dawn tree     (default: %WEB_NN%\_dawn_prebuilt_win)
#   DAWN_SRC_DIR        webgpu-dawn-binaries   (default: %WEB_NN%\_webgpu_dawn_src)
#   CR_LIBCXX_DEST      packed libc++.lib dir  (default: %WEB_NN%\_cr_libcxx_link_win)
#
# Usage: dot-source AFTER the param() block (param() must be the first
# statement, so the dot-source can't precede it). Everything defined here
# lands in the caller's scope.

# Refresh PATH from the registry (Machine + User) so freshly installed tools
# (cmake, python, git, bazel, ninja) are found even when the invoking shell
# has a stale env.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

$BundleDir = Split-Path $PSScriptRoot -Parent   # ...\route-a-webgpu-windows
$WebnnRoot = Split-Path $BundleDir -Parent      # ...\webnn

# Workspace root: $env:WORKSPACE_ROOT when set (webnn, chromium and
# depot_tools are siblings under it), otherwise %USERPROFILE%\workspace.
$WorkspaceRoot = [Environment]::GetEnvironmentVariable("WORKSPACE_ROOT")
if (-not $WorkspaceRoot) {
  $WorkspaceRoot = Join-Path $env:USERPROFILE "workspace"
}

$DefaultChromiumSrc  = Join-Path $WorkspaceRoot "chromium\src"
$DefaultMlDrift      = Join-Path $DefaultChromiumSrc "third_party\ml-drift"
$DefaultLitert       = Join-Path $DefaultChromiumSrc "third_party\litert\src"
$DefaultDepotTools   = Join-Path $WorkspaceRoot "depot_tools"
$DefaultDawnPrebuilt = Join-Path $WebnnRoot "_dawn_prebuilt_win"
$DefaultDawnSrc      = Join-Path $WebnnRoot "_webgpu_dawn_src"
$DefaultCrLibcxxDest = Join-Path $WebnnRoot "_cr_libcxx_link_win"

function Resolve-BundlePath {
    param(
        [string]$ParamValue,   # caller's parameter value ("" if not passed)
        [string]$EnvVarName,   # environment variable that may override
        [string]$Default       # derived default
    )
    if ($ParamValue) { return $ParamValue }
    $fromEnv = [Environment]::GetEnvironmentVariable($EnvVarName)
    if ($fromEnv) { return $fromEnv }
    return $Default
}
