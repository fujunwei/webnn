# Decode an image, resize longest side to 1024 (bilinear), pad right/bottom
# with black to 1024x1024, and dump raw 24bpp pixels (GDI+ 24bppRgb = B,G,R).
# Matches the original SAM ResizeLongestSide + F.pad(0, padw, 0, padh) flow.
param(
  [string]$In = "C:\Users\junweifu\workspace\tflite-dump-model\EgyptianCat.png",
  [string]$Out = "C:\Users\junweifu\workspace\webnn\segment_anythings\sam_native_runner\segment_anythings_empty_result\sam_enc_1024_bgr.bin",
  [int]$Size = 1024
)
Add-Type -AssemblyName System.Drawing

$src = [System.Drawing.Image]::FromFile($In)
$w = $src.Width
$h = $src.Height
$scale = $Size / [Math]::Max($w, $h)
$sw = [Math]::Round($w * $scale)
$sh = [Math]::Round($h * $scale)

$bmp = New-Object System.Drawing.Bitmap($Size, $Size)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::Black)   # pad value 0
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBilinear
$g.DrawImage($src, 0, 0, $sw, $sh)        # top-left, pad right/bottom
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

[System.IO.Directory]::CreateDirectory((Split-Path $Out)) | Out-Null
[System.IO.File]::WriteAllBytes($Out, $raw)
Write-Output ("src={0}x{1} scale={2} -> {3}x{4} @(0,0); wrote {5} bytes -> {6}" -f $w,$h,$scale,$sw,$sh,$raw.Length,$Out)
