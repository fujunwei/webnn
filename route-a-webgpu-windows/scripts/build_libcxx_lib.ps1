# Pack Chromium's libc++ ThinLTO .obj files into a static libc++.lib.
#
# Chromium builds libc++ as LLVM bitcode objects (ThinLTO). MSVC's lib.exe cannot
# handle these (LNK1107). We use LLD's /lib mode which understands bitcode.
#
# Prereqs:
#   - Chromium checkout has been built at least once in Release mode
#     (out/Release/obj/buildtools/third_party/libc++/libc++/*.obj must exist)
#   - Chromium's clang toolchain has been fetched
#     (third_party/llvm-build/Release+Asserts/bin/lld-link.exe must exist)
#
# Output: $DEST\libc++.lib (~3 MB)

param(
    [string]$ChromiumSrc = "C:\Users\junweifu\workspace\chromium\src",
    [string]$Dest        = "C:\Users\junweifu\workspace\webnn\_cr_libcxx_link_win"
)

$ErrorActionPreference = "Stop"

$LLD    = Join-Path $ChromiumSrc "third_party\llvm-build\Release+Asserts\bin\lld-link.exe"
$OBJDIR = Join-Path $ChromiumSrc "out\Release\obj\buildtools\third_party\libc++\libc++"

if (-not (Test-Path $LLD))    { throw "lld-link.exe not found at $LLD (fetch Chromium clang toolchain first)" }
if (-not (Test-Path $OBJDIR)) { throw "libc++ .obj dir not found at $OBJDIR (build Chromium Release first: 'autoninja -C out/Release chrome')" }

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# Write response file with all .obj paths (avoids Windows command-line length limit)
$rsp  = Join-Path $Dest "objs.rsp"
$objs = Get-ChildItem "$OBJDIR\*.obj" | ForEach-Object { '"' + $_.FullName + '"' }
if ($objs.Count -eq 0) { throw "No .obj files in $OBJDIR" }
$objs | Out-File -FilePath $rsp -Encoding ASCII -NoNewline

$out = Join-Path $Dest "libc++.lib"
& $LLD /lib /NOLOGO "/OUT:$out" "@$rsp"
if ($LASTEXITCODE -ne 0) { throw "lld-link /lib failed: exit $LASTEXITCODE" }

$size = (Get-Item $out).Length
Write-Host "OK: $out ($([math]::Round($size/1MB,2)) MB, from $($objs.Count) .obj files)"

# Sanity check: should contain __Cr namespace symbols
$bytes = [System.IO.File]::ReadAllBytes($out)
$text  = [System.Text.Encoding]::ASCII.GetString($bytes)
$crCount = ($text -split "`0" | Where-Object { $_ -match "__Cr@std" }).Count
Write-Host "  __Cr@std string occurrences: $crCount (must be > 0)"
if ($crCount -eq 0) { throw "libc++.lib does not contain __Cr symbols (check __config_site defines _LIBCPP_ABI_NAMESPACE=__Cr)" }
