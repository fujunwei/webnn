# Lists peak/current working set of every chrome.exe process (by type) from the custom Chromium build.
# Usage: run while the SD page is open / right after the image is generated.
#   .\chrome-peak-memory.ps1
#   .\chrome-peak-memory.ps1 -PathPattern '*chromium\src\out\Release*'
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
        Type   = $type
        PID    = $_.ProcessId
        PeakMB = [math]::Round($p.PeakWorkingSet64 / 1MB, 1)
        CurMB  = [math]::Round($p.WorkingSet64 / 1MB, 1)
      }
    }
  }

if (-not $rows) {
  Write-Host "No chrome.exe matching '$PathPattern'. Actual paths:"
  Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Select-Object -ExpandProperty ExecutablePath -Unique
  return
}

$rows | Sort-Object PeakMB -Descending | Format-Table -AutoSize

function Sum($types, $prop) {
  [math]::Round(($rows | Where-Object { $_.Type -in $types } | Measure-Object $prop -Sum).Sum, 1)
}
$topRenderer = $rows | Where-Object { $_.Type -eq 'renderer' } | Sort-Object PeakMB -Descending | Select-Object -First 1

"GPU process peak (sum)          : {0} MB" -f (Sum @('gpu-process') 'PeakMB')
"Renderer peak (largest)         : {0} MB (PID {1})" -f $topRenderer.PeakMB, $topRenderer.PID
"Renderer peak (all, sum)        : {0} MB" -f (Sum @('renderer') 'PeakMB')
"GPU + largest renderer peak     : {0} MB" -f ((Sum @('gpu-process') 'PeakMB') + $topRenderer.PeakMB)
"All chrome processes peak (sum) : {0} MB" -f (Sum ($rows.Type | Select-Object -Unique) 'PeakMB')
