# Decode a JPEG, resize to 224x224 (stretch, bicubic), and dump raw 24bpp
# pixels (GDI+ 24bppRgb stores B,G,R byte order) to a tightly packed .bin.
param(
  [string]$In = "C:\Users\junweifu\workspace\webnn\segment_anythings\sam_native_runner\tiger.jpg",
  [string]$Out = "C:\Users\junweifu\workspace\tflite-dump-model\tiger_224_bgr.bin",
  [int]$Size = 224
)
Add-Type -AssemblyName System.Drawing

$src = [System.Drawing.Image]::FromFile($In)
$bmp = New-Object System.Drawing.Bitmap($Size, $Size)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.DrawImage($src, 0, 0, $Size, $Size)
$g.Dispose()
$src.Dispose()

$rect = New-Object System.Drawing.Rectangle(0, 0, $Size, $Size)
$data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                      [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$stride = $data.Stride
$padded = New-Object byte[] ($stride * $Size)
[System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $padded, 0, $padded.Length)
$bmp.UnlockBits($data)

$rowBytes = $Size * 3
$raw = New-Object byte[] ($rowBytes * $Size)
for ($y = 0; $y -lt $Size; $y++) {
  [System.Array]::Copy($padded, $y * $stride, $raw, $y * $rowBytes, $rowBytes)
}
$bmp.Dispose()

[System.IO.File]::WriteAllBytes($Out, $raw)
Write-Output ("wrote " + $raw.Length + " bytes -> " + $Out)
