# Like chrome-peak-memory.ps1, but also reports COMMIT (private bytes), which is what PartitionAlloc / OOM care about.
# chrome-peak-memory.ps1 only reports the working set (resident pages), which never counts memory that was
# committed but not touched (e.g. TFLite arenas) and does count file-mapped pages (e.g. the weights file).
# Usage: run while the SD page is open / right after the image is generated.
#   .\chrome-peak-commit.ps1
param(
  [string]$PathPattern = '*chromium\src\out\Release*'
)

$rows = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" |
  Where-Object { $_.ExecutablePath -like $PathPattern } |
  ForEach-Object {
    $type = if ($_.CommandLine -match '--type=([\w-]+)') { $matches[1] } else { 'browser' }
    $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    if ($p) {
      [pscustomobject]@{
        Type         = $type
        PID          = $_.ProcessId
        PeakCommitMB = [math]::Round($p.PeakPagedMemorySize64 / 1MB, 1)
        CurCommitMB  = [math]::Round($p.PrivateMemorySize64 / 1MB, 1)
        PeakWS_MB    = [math]::Round($p.PeakWorkingSet64 / 1MB, 1)
        CurWS_MB     = [math]::Round($p.WorkingSet64 / 1MB, 1)
      }
    }
  }

if (-not $rows) { Write-Host "No chrome.exe matching '$PathPattern'."; return }

$rows | Sort-Object PeakCommitMB -Descending | Format-Table -AutoSize

$topRenderer = $rows | Where-Object { $_.Type -eq 'renderer' } | Sort-Object PeakCommitMB -Descending | Select-Object -First 1
$gpu = $rows | Where-Object { $_.Type -eq 'gpu-process' } | Measure-Object PeakCommitMB -Sum
"Renderer peak COMMIT (largest)  : {0} MB (PID {1})  <- PartitionAlloc pool limit is 16384 MB per pool" -f $topRenderer.PeakCommitMB, $topRenderer.PID
"Renderer peak working set       : {0} MB" -f $topRenderer.PeakWS_MB
"GPU process peak COMMIT (sum)   : {0} MB" -f [math]::Round($gpu.Sum, 1)
