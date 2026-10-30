<#
.SYNOPSIS
  Sync C:\Users\junwei\workspace\chromium to C:\Users\webnn\workspace\chromium on 10.239.182.81.

.DESCRIPTION
  Uses robocopy over the remote's C$ admin share (SMB). The remote account needs
  admin rights (or the target folder must be shared separately). Robocopy only
  copies changed files, so repeat runs are incremental.

  WARNING: -Mirror deletes files on the destination that no longer exist in the
  source. Use -DryRun first.

.EXAMPLE
  .\sync-to-remote.ps1 -DryRun
  .\sync-to-remote.ps1
  .\sync-to-remote.ps1 -IncludeOut -Mirror
#>
param(
    [string]$Source     = 'C:\Users\junwei\workspace\chromium',
    [string]$RemoteHost = '10.239.182.81',
    [string]$RemoteUser = 'webnn',
    [string]$ShareName  = 'Users',      # remote share of C:\Users
    [string]$RemotePath = 'webnn\workspace\chromium',   # relative to the share
    [string]$SubPath    = '',   # sync only this subdirectory (relative to Source), e.g. src\third_party\litert
    [switch]$Mirror,     # delete extra files on destination (robocopy /PURGE)
    [switch]$IncludeOut,  # also copy build output dirs (src\out*), skipped by default
    [switch]$DryRun       # list only, copy nothing
)

$ErrorActionPreference = 'Stop'

$share = "\\$RemoteHost\$ShareName"
$dest  = Join-Path $share $RemotePath
$srcDir = $Source
if ($SubPath) {
    $srcDir = Join-Path $Source $SubPath
    $dest   = Join-Path $dest $SubPath
    if (-not (Test-Path $srcDir -PathType Container)) { throw "Source subdirectory not found: $srcDir" }
}

# Connect to the share with explicit credentials (prompts for password).
$cred = Get-Credential -UserName $RemoteUser -Message "Password for $RemoteUser@$RemoteHost"
$plain = $cred.GetNetworkCredential().Password
# Run a native command without its stderr turning into a terminating error.
function Invoke-Quiet([scriptblock]$Block) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Block 2>&1 | Out-Null } finally { $ErrorActionPreference = $old }
}

Invoke-Quiet { net use $share /delete /y }
$old = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$out = net use $share $plain "/user:$($cred.UserName)" 2>&1 | Out-String
$ErrorActionPreference = $old
if ($LASTEXITCODE -ne 0) { throw "Cannot connect to $share`n$out" }

try {
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }

    $roboArgs = @(
        $srcDir, $dest,
        '/E',                 # include subdirectories
        '/MT:32',             # multithreaded
        '/R:2', '/W:2',       # retry twice, 2s apart
        '/FFT',               # tolerate coarse timestamps
        '/XJ',                # don't follow junctions
        '/NP', '/NDL'         # quieter output
    )
    if (-not $IncludeOut) {
        # /XD does not accept wildcards in full paths, so expand src\out and src\out_* here.
        $outDirs = Get-ChildItem (Join-Path $Source 'src') -Directory |
            Where-Object { $_.Name -eq 'out' -or $_.Name -like 'out_*' } |
            ForEach-Object FullName
        if ($outDirs) { $roboArgs += @('/XD') + $outDirs }
    }
    if ($Mirror)          { $roboArgs += '/PURGE' }
    if ($DryRun)          { $roboArgs += '/L' }

    robocopy @roboArgs
    # robocopy exit codes 0-7 are success, >=8 means failures.
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
    Write-Host "Done (robocopy exit code $LASTEXITCODE)."
}
finally {
    Invoke-Quiet { net use $share /delete /y }
}
