# Verify the prebuilt Dawn tree matches the Dawn version Chromium was built with.
#
# Why this matters (README section 1.4b): under the Dawn Proc route the
# accelerator embeds the 20-byte SHA1 from <prebuilt>\include\dawn\dawn_version.h
# at compile time, and dawnProcSetProcs() byte-compares it against the proc
# table Chromium passes at runtime (which carries Chromium's own Dawn build
# hash). A mismatch aborts the GPU process. So after updating Chromium you must
# re-run build_dawn.ps1 if -- and only if -- this script reports a mismatch.
#
# Usage:
#   .\check_dawn_version.ps1                 # compares against out\Release
#   .\check_dawn_version.ps1 -ChromeOutDir <chromium src>\out\upstream_bots_debug

param(
    [string]$ChromiumSrc,   # default: %WORKSPACE_ROOT%\chromium\src (or $env:CHROMIUM_SRC)
    [string]$ChromeOutDir   # default: %CHROMIUM_SRC%\out\Release
)

. "$PSScriptRoot\common.ps1"

$ChromiumSrc = Resolve-BundlePath $ChromiumSrc "CHROMIUM_SRC" $DefaultChromiumSrc
if (-not $ChromeOutDir) { $ChromeOutDir = Join-Path $ChromiumSrc "out\Release" }

function Get-DawnHash {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $line = Select-String -Path $Path -Pattern "kDawnVersion\s*=" | Select-Object -First 1
    if (-not $line) { return $null }
    $bytes = [regex]::Matches($line.Line, "0x[0-9a-fA-F]{2}") |
        ForEach-Object { $_.Value.Substring(2).ToLower() }
    if ($bytes.Count -ne 20) { return $null }
    return ($bytes -join "")
}

$prebuiltHdr = Join-Path $DefaultDawnPrebuilt "include\dawn\dawn_version.h"
$chromiumHdr = Join-Path $ChromeOutDir "gen\third_party\dawn\include\dawn\dawn_version.h"

$prebuiltHash = Get-DawnHash $prebuiltHdr
$chromiumHash = Get-DawnHash $chromiumHdr

# Secondary signal: HEAD of the Dawn source checkout inside the Chromium tree.
# (The generated header above only reflects what was actually built, which is
# what matters; the checkout hash is shown for context and used as fallback.)
$chromiumHead = git -C (Join-Path $ChromiumSrc "third_party\dawn") rev-parse HEAD 2>$null
if ($chromiumHead) { $chromiumHead = $chromiumHead.Trim().ToLower() }

Write-Host "Prebuilt Dawn  (accelerator compile-time): $prebuiltHash   ($prebuiltHdr)"
if ($chromiumHash) {
    Write-Host "Chromium Dawn  (runtime proc table):      $chromiumHash   ($chromiumHdr)"
} else {
    Write-Warning "Chromium's generated dawn_version.h not found at $chromiumHdr"
    Write-Warning "(Chromium out dir not built yet, or a different out dir -- pass -ChromeOutDir.)"
}
if ($chromiumHead) { Write-Host "Chromium third_party/dawn git HEAD:       $chromiumHead" }

Write-Host ""
if ($chromiumHash -and $prebuiltHash -eq $chromiumHash) {
    Write-Host "RESULT: MATCH -- no need to re-run build_dawn.ps1." -ForegroundColor Green
    if ($chromiumHead -and $chromiumHead -ne $chromiumHash) {
        Write-Warning "  (The checkout has a newer Dawn roll than what out\Release was built with."
        Write-Warning "   After the next chrome rebuild, re-run this script to confirm.)"
    }
    exit 0
} elseif ($chromiumHead -and $prebuiltHash -eq $chromiumHead) {
    Write-Host "RESULT: prebuilt Dawn already matches the current checkout --" -ForegroundColor Green
    Write-Host "no need to re-run build_dawn.ps1." -ForegroundColor Green
    if ($chromiumHash) {
        Write-Warning "BUT the chrome binaries in $ChromeOutDir were built with an OLDER Dawn"
        Write-Warning "(built hash $chromiumHash != checkout $chromiumHead). The runtime version"
        Write-Warning "check in dawnProcSetProcs() would abort the GPU process. Rebuild chrome"
        Write-Warning "(autoninja -C $ChromeOutDir chrome), then re-run this script."
        exit 1
    }
    Write-Host "Rebuild chrome, then re-run this script to confirm the built hash."
    exit 0
} else {
    Write-Host "RESULT: MISMATCH -- re-run scripts\build_dawn.ps1 (it now reads the"
    Write-Host "version from chrome\VERSION automatically), then rebuild the accelerator." -ForegroundColor Red
    exit 1
}
