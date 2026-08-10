/**
 * Segment Anything Demo — LiteRT.js + WebNN
 * Uses MobileSAM TFLite models accelerated via WebNN GPU delegate.
 *
 * Models: qualcomm/MobileSam (HuggingFace)
 *   - encoder.tflite: 26.6 MB, input [1,1024,1024,3], output [1,64,64,256]
 *   - decoder.tflite: 23.7 MB, inputs [1,64,64,256][1,1,2][1,1], outputs [1,256,256,1][1,1]
 */

import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';

// ─── DOM refs ───────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fileInput = document.getElementById('file-input');
const btnClear = document.getElementById('btn-clear');
const btnCut = document.getElementById('btn-cut');
const progressOverlay = document.getElementById('progress-overlay');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const logArea = document.getElementById('log-area');
const gpuBadge = document.getElementById('gpu-badge');
const webnnBadge = document.getElementById('webnn-badge');
const statusText = document.getElementById('status-text');
const encTimeEl = document.getElementById('enc-time');
const decTimeEl = document.getElementById('dec-time');
const backendInfo = document.getElementById('backend-info');

// ─── State ──────────────────────────────────────────────────────────────────
const IMG_SIZE = 1024;
const MAX_DISPLAY = 680; // max display size

let encoderModel = null;   // compiled encoder
let decoderModel = null;   // compiled decoder
let imageEmbeddings = null; // Float32Array of encoder output
let originalImage = null;   // HTML ImageElement
let displayWidth, displayHeight; // canvas display size
let points = [];            // [{x, y, label}] in 1024x1024 space
let maxPoints = 1;          // max points supported by decoder (1=MobileSAM, 2=full SAM)
let isProcessing = false;
let preprocessInfo = null;  // stored from encoder step for coordinate mapping

// ─── Logging ────────────────────────────────────────────────────────────────
function log(msg) {
  const time = new Date().toLocaleTimeString();
  logArea.textContent += `[${time}] ${msg}\n`;
  logArea.scrollTop = logArea.scrollHeight;
  console.log(`[SAM-LiteRT] ${msg}`);
}

// ─── Progress ───────────────────────────────────────────────────────────────
function showProgress(text, percent) {
  progressOverlay.classList.remove('hidden');
  progressText.textContent = text;
  progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

function hideProgress() {
  progressOverlay.classList.add('hidden');
}

// ─── Check WebNN support ────────────────────────────────────────────────────
async function checkWebNN() {
  try {
    if (typeof MLContext === 'undefined' || typeof navigator.ml === 'undefined') {
      return false;
    }
    const context = await navigator.ml.createContext({ deviceType: 'gpu' });
    return !!context;
  } catch (e) {
    log(`WebNN check failed: ${e.message}`);
    return false;
  }
}

// ─── OPFS cache ─────────────────────────────────────────────────────────────
async function getOPFS(name) {
  try {
    const root = await navigator.storage.getDirectory();
    const fileHandle = await root.getFileHandle(name, { create: false });
    const file = await fileHandle.getFile();
    const buffer = await file.arrayBuffer();
    log(`Loaded ${name} from OPFS cache (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
    return new Uint8Array(buffer);
  } catch {
    return null;
  }
}

async function putOPFS(name, buffer) {
  try {
    const root = await navigator.storage.getDirectory();
    const fileHandle = await root.getFileHandle(name, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(buffer);
    await writable.close();
    log(`Cached ${name} to OPFS (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
  } catch (e) {
    log(`OPFS write failed for ${name}: ${e.message}`);
  }
}

// ─── Model downloading ──────────────────────────────────────────────────────
async function fetchWithProgress(url, label) {
  // Try OPFS cache first
  const fileName = url.split('/').pop();
  const cached = await getOPFS(`mobile_sam_${fileName}`);
  if (cached) {
    return cached;
  }

  log(`Downloading ${label} from ${url}...`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);

  const contentLength = parseInt(resp.headers.get('Content-Length') || '0');
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (contentLength > 0) {
      const pct = (received / contentLength) * 100;
      showProgress(`Downloading ${label}...`, pct);
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  // Cache to OPFS
  await putOPFS(`mobile_sam_${fileName}`, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  log(`Downloaded ${label}: ${(received / 1024 / 1024).toFixed(1)} MB`);
  return buffer; // return Uint8Array (not ArrayBuffer)
}

// ─── LiteRT.js initialization ───────────────────────────────────────────────
async function initLiteRT() {
  showProgress('Loading LiteRT.js runtime...', 5);

  // Load wasm with JSPI enabled (required for WebNN)
  log('Loading LiteRT wasm with JSPI...');
  showProgress('Loading LiteRT.js WASM (JSPI)...', 10);

  await loadLiteRt('https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/', { jspi: true });
  log('LiteRT.js runtime initialized');
}

// ─── Model loading ──────────────────────────────────────────────────────────
async function loadModels() {
  // URL params:
  //   ?models=<base_url>  — model directory (default: ./models/)
  //   ?encoder=<name>     — encoder filename (default: encoder.tflite)
  //   ?decoder=<name>     — decoder filename (default: decoder.tflite)
  const urlParams = new URLSearchParams(window.location.search);
  const modelBase = urlParams.get('models') || './models/';
  const encoderName = urlParams.get('encoder') || 'encoder.tflite';
  const decoderName = urlParams.get('decoder') || 'decoder.tflite';

  log(`Model base: ${modelBase}`);
  log(`Encoder: ${encoderName}, Decoder: ${decoderName}`);

  // Pass URL directly to loadAndCompile — it handles download + compilation
  showProgress('Loading & compiling encoder...', 70);
  log('Compiling encoder with WebNN delegate...');
  const encStart = performance.now();
  encoderModel = await loadAndCompile(modelBase + encoderName, {
    accelerator: 'webnn',
    webNNOptions: { devicePreference: 'gpu' },
  });
  const encTime = performance.now() - encStart;
  log(`Encoder compiled in ${encTime.toFixed(0)} ms`);

  showProgress('Loading & compiling decoder...', 85);
  log('Compiling decoder with WebNN delegate...');
  const decStart = performance.now();
  decoderModel = await loadAndCompile(modelBase + decoderName, {
    accelerator: 'webnn',
    webNNOptions: { devicePreference: 'gpu' },
  });
  const decTime = performance.now() - decStart;
  log(`Decoder compiled in ${decTime.toFixed(0)} ms`);

  // Auto-detect max points from decoder input shape
  // Full SAM decoder: point_labels [1, 2] → 2 points
  // MobileSAM decoder:  point_labels [1, 1] → 1 point
  try {
    const details = decoderModel.getInputDetails?.() || decoderModel.inputs || [];
    for (const inp of details) {
      if (inp.name === 'point_labels' && inp.shape) {
        maxPoints = inp.shape[1] || 1;
        log(`Detected max points: ${maxPoints}`);
        break;
      }
    }
  } catch (e) {
    log(`Could not detect max points, defaulting to 1`);
  }

  hideProgress();
}

// ─── Image preprocessing ────────────────────────────────────────────────────
function preprocessImage(img) {
  // Draw image to a 1024x1024 canvas (resize keeping aspect ratio, pad)
  const offCanvas = document.createElement('canvas');
  offCanvas.width = IMG_SIZE;
  offCanvas.height = IMG_SIZE;
  const offCtx = offCanvas.getContext('2d');

  // Fill with padding (gray 0.5)
  offCtx.fillStyle = '#808080';
  offCtx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);

  // Compute scaled size preserving aspect ratio
  const scale = IMG_SIZE / Math.max(img.width, img.height);
  const sw = Math.round(img.width * scale);
  const sh = Math.round(img.height * scale);
  const dx = Math.floor((IMG_SIZE - sw) / 2);
  const dy = Math.floor((IMG_SIZE - sh) / 2);

  offCtx.drawImage(img, dx, dy, sw, sh);

  const imageData = offCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const pixels = imageData.data; // RGBA, uint8

  // Convert to NHWC float32 [1, 1024, 1024, 3], values in [0, 1]
  const floatData = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    floatData[i * 3] = pixels[i * 4] / 255.0;       // R
    floatData[i * 3 + 1] = pixels[i * 4 + 1] / 255.0; // G
    floatData[i * 3 + 2] = pixels[i * 4 + 2] / 255.0; // B
  }

  // Store scale info for coordinate mapping
  return {
    tensor: new Tensor(floatData, [1, IMG_SIZE, IMG_SIZE, 3]),
    scale,
    offsetX: dx,
    offsetY: dy,
    scaledWidth: sw,
    scaledHeight: sh,
  };
}

// ─── Display image on canvas ────────────────────────────────────────────────
function displayImage(img) {
  // Scale for display
  const scale = MAX_DISPLAY / Math.max(img.width, img.height);
  displayWidth = Math.round(img.width * scale);
  displayHeight = Math.round(img.height * scale);

  canvas.width = displayWidth;
  canvas.height = displayHeight;
  ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
  originalImage = img;
}

// ─── Run encoder ────────────────────────────────────────────────────────────
async function runEncoder(img) {
  showProgress('Preprocessing image...', 95);
  log('Preprocessing image for encoder...');
  const preprocessed = preprocessImage(img);

  showProgress('Running encoder...', 97);
  log('Running encoder inference...');
  const start = performance.now();

  const outputs = await encoderModel.run({ image: preprocessed.tensor });
  const embeddings = outputs.image_embeddings;

  // Get output data (move from GPU to CPU)
  const embeddingData = await embeddings.data();
  // .data() may return TypedArray or ArrayBuffer; normalize
  if (ArrayBuffer.isView(embeddingData)) {
    imageEmbeddings = new Float32Array(embeddingData.buffer, embeddingData.byteOffset, embeddingData.byteLength / 4);
  } else {
    imageEmbeddings = new Float32Array(embeddingData);
  }

  const elapsed = performance.now() - start;
  encTimeEl.textContent = `${elapsed.toFixed(0)} ms`;
  log(`Encoder inference: ${elapsed.toFixed(1)} ms`);

  // Clean up
  preprocessed.tensor.delete();
  embeddings.delete();

  hideProgress();

  // Store preprocess info for coordinate mapping
  preprocessInfo = {
    scale: preprocessed.scale,
    offsetX: preprocessed.offsetX,
    offsetY: preprocessed.offsetY,
    scaledWidth: preprocessed.scaledWidth,
    scaledHeight: preprocessed.scaledHeight,
  };
}

// ─── Coordinate conversion ─────────────────────────────────────────────────
function canvasToImageSpace(canvasX, canvasY) {
  // canvas → original image → 1024x1024 model space
  const imgX = canvasX / displayWidth * originalImage.width;
  const imgY = canvasY / displayHeight * originalImage.height;

  if (preprocessInfo) {
    return {
      x: imgX * preprocessInfo.scale + preprocessInfo.offsetX,
      y: imgY * preprocessInfo.scale + preprocessInfo.offsetY,
    };
  }

  // Fallback: recompute
  const scale = IMG_SIZE / Math.max(originalImage.width, originalImage.height);
  const sw = originalImage.width * scale;
  const sh = originalImage.height * scale;
  const dx = (IMG_SIZE - sw) / 2;
  const dy = (IMG_SIZE - sh) / 2;
  return { x: imgX * scale + dx, y: imgY * scale + dy };
}

// ─── Run decoder ────────────────────────────────────────────────────────────
async function runDecoder(activePoints) {
  if (!imageEmbeddings) return null;

  const start = performance.now();

  // Build decoder inputs
  // image_embeddings: [1, 64, 64, 256]
  const embTensor = new Tensor(imageEmbeddings, [1, 64, 64, 256]);

  // Take the most recent N points (up to maxPoints), pad if fewer
  const recent = activePoints.slice(-maxPoints);
  const numPts = recent.length;

  // point_coords: [1, maxPoints, 2] — pad with zeros for unused slots
  const coordsData = new Float32Array(maxPoints * 2);
  const labelsData = new Float32Array(maxPoints);
  for (let i = 0; i < maxPoints; i++) {
    if (i < numPts) {
      coordsData[i * 2] = recent[i].x;
      coordsData[i * 2 + 1] = recent[i].y;
      labelsData[i] = recent[i].label;
    } else {
      // Pad: duplicate last point's coords with label=0 (ignored)
      coordsData[i * 2] = recent[numPts - 1].x;
      coordsData[i * 2 + 1] = recent[numPts - 1].y;
      labelsData[i] = 0;
    }
  }
  const coordsTensor = new Tensor(coordsData, [1, maxPoints, 2]);
  const labelsTensor = new Tensor(labelsData, [1, maxPoints]);

  // Run decoder
  const outputs = await decoderModel.run({
    image_embeddings: embTensor,
    point_coords: coordsTensor,
    point_labels: labelsTensor,
  });

  const masks = outputs.masks;
  const scores = outputs.scores;

  // Get mask data [1, 256, 256, 1]
  const maskRaw = await masks.data();
  const maskArray = ArrayBuffer.isView(maskRaw)
    ? new Float32Array(maskRaw.buffer, maskRaw.byteOffset, maskRaw.byteLength / 4)
    : new Float32Array(maskRaw);

  // Get score
  const scoreRaw = await scores.data();
  const score = ArrayBuffer.isView(scoreRaw)
    ? new Float32Array(scoreRaw.buffer, scoreRaw.byteOffset, scoreRaw.byteLength / 4)[0]
    : new Float32Array(scoreRaw)[0];

  const elapsed = performance.now() - start;
  decTimeEl.textContent = `${elapsed.toFixed(0)} ms`;
  log(`Decoder: ${elapsed.toFixed(1)} ms, score: ${score.toFixed(4)}, points: ${numPts}/${maxPoints}`);

  // Clean up
  embTensor.delete();
  coordsTensor.delete();
  labelsTensor.delete();
  masks.delete();
  scores.delete();

  return {
    mask: maskArray, // [1, 256, 256, 1]
    score,
    timeMs: elapsed,
  };
}

// ─── Render mask overlay ────────────────────────────────────────────────────
function renderMask(result) {
  if (!result || !originalImage) return;

  // Redraw original image
  const scale = MAX_DISPLAY / Math.max(originalImage.width, originalImage.height);
  displayWidth = Math.round(originalImage.width * scale);
  displayHeight = Math.round(originalImage.height * scale);
  canvas.width = displayWidth;
  canvas.height = displayHeight;
  ctx.drawImage(originalImage, 0, 0, displayWidth, displayHeight);

  // Get image pixel data
  const imageData = ctx.getImageData(0, 0, displayWidth, displayHeight);

  // Resize 256x256 mask to display size
  const offCanvas = document.createElement('canvas');
  offCanvas.width = 256;
  offCanvas.height = 256;
  const offCtx = offCanvas.getContext('2d');
  const maskImageData = offCtx.createImageData(256, 256);
  for (let i = 0; i < 256 * 256; i++) {
    const val = result.mask[i]; // sigmoid already applied by model
    const v = Math.round(Math.max(0, Math.min(255, val * 255)));
    maskImageData.data[i * 4] = v;
    maskImageData.data[i * 4 + 1] = v;
    maskImageData.data[i * 4 + 2] = v;
    maskImageData.data[i * 4 + 3] = 255;
  }
  offCtx.putImageData(maskImageData, 0, 0);

  // Draw mask scaled to display size
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = displayWidth;
  tempCanvas.height = displayHeight;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(offCanvas, 0, 0, displayWidth, displayHeight);
  const scaledMask = tempCtx.getImageData(0, 0, displayWidth, displayHeight);

  // Apply green overlay where mask > 0.5
  for (let i = 0; i < displayWidth * displayHeight; i++) {
    const maskVal = scaledMask.data[i * 4] / 255;
    if (maskVal > 0.5) {
      // 50% green overlay
      imageData.data[i * 4] = Math.round(imageData.data[i * 4] * 0.4 + 0x4c * 0.6);
      imageData.data[i * 4 + 1] = Math.round(imageData.data[i * 4 + 1] * 0.4 + 0xaf * 0.6);
      imageData.data[i * 4 + 2] = Math.round(imageData.data[i * 4 + 2] * 0.4 + 0x50 * 0.6);
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // Draw point markers
  for (const pt of points) {
    const cx = pt.x / IMG_SIZE * displayWidth;
    const cy = pt.y / IMG_SIZE * displayHeight;
    ctx.fillStyle = pt.label === 1 ? '#4caf50' : '#f44336';
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// ─── Handle image upload ────────────────────────────────────────────────────
async function handleImage(file) {
  if (isProcessing) return;
  isProcessing = true;
  points = [];
  imageEmbeddings = null;
  btnClear.disabled = true;
  btnCut.disabled = true;

  const reader = new FileReader();
  const img = await new Promise((resolve, reject) => {
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  displayImage(img);

  try {
    await runEncoder(img);
  } catch (e) {
    log(`Encoder error: ${e.message}`);
    console.error(e);
  }

  btnClear.disabled = false;
  btnCut.disabled = false;
  isProcessing = false;
}

// ─── Handle click ───────────────────────────────────────────────────────────
async function handleClick(e) {
  if (isProcessing || !imageEmbeddings) return;

  const rect = canvas.getBoundingClientRect();
  const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
  const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

  const imgCoords = canvasToImageSpace(canvasX, canvasY);
  points.push({ x: imgCoords.x, y: imgCoords.y, label: 1 });

  isProcessing = true;
  try {
    const result = await runDecoder(points);
    renderMask(result);
    btnCut.disabled = false;
  } catch (e) {
    log(`Decoder error: ${e.message}`);
    console.error(e);
  }
  isProcessing = false;
}

// ─── Handle right-click (negative point) ────────────────────────────────────
async function handleRightClick(e) {
  e.preventDefault();
  if (isProcessing || !imageEmbeddings) return;

  const rect = canvas.getBoundingClientRect();
  const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
  const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

  const imgCoords = canvasToImageSpace(canvasX, canvasY);
  points.push({ x: imgCoords.x, y: imgCoords.y, label: 0 });

  isProcessing = true;
  try {
    const result = await runDecoder(points);
    renderMask(result);
  } catch (e) {
    log(`Decoder error: ${e.message}`);
    console.error(e);
  }
  isProcessing = false;
}

// ─── Clear all points ───────────────────────────────────────────────────────
async function clearPoints() {
  points = [];
  if (originalImage) {
    const scale = MAX_DISPLAY / Math.max(originalImage.width, originalImage.height);
    displayWidth = Math.round(originalImage.width * scale);
    displayHeight = Math.round(originalImage.height * scale);
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    ctx.drawImage(originalImage, 0, 0, displayWidth, displayHeight);
    btnCut.disabled = true;
  }
}

// ─── Cut out masked region ──────────────────────────────────────────────────
async function handleCut() {
  if (!originalImage || points.length === 0 || !imageEmbeddings) return;

  isProcessing = true;
  try {
    // Run decoder for the last point to get current mask
    const result = await runDecoder(points);

    // Create output canvas at original image resolution
    const outCanvas = document.createElement('canvas');
    outCanvas.width = originalImage.width;
    outCanvas.height = originalImage.height;
    const outCtx = outCanvas.getContext('2d');

    // Draw original image
    outCtx.drawImage(originalImage, 0, 0);

    // Resize mask to original image size
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = 256;
    maskCanvas.height = 256;
    const maskCtx = maskCanvas.getContext('2d');
    const maskImageData = maskCtx.createImageData(256, 256);
    for (let i = 0; i < 256 * 256; i++) {
      const val = result.mask[i];
      const v = Math.round(Math.max(0, Math.min(255, val * 255)));
      maskImageData.data[i * 4] = v;
      maskImageData.data[i * 4 + 1] = v;
      maskImageData.data[i * 4 + 2] = v;
      maskImageData.data[i * 4 + 3] = 255;
    }
    maskCtx.putImageData(maskImageData, 0, 0);

    const scaledMaskCanvas = document.createElement('canvas');
    scaledMaskCanvas.width = originalImage.width;
    scaledMaskCanvas.height = originalImage.height;
    const scaledMaskCtx = scaledMaskCanvas.getContext('2d');
    scaledMaskCtx.drawImage(maskCanvas, 0, 0, originalImage.width, originalImage.height);
    const scaledMaskData = scaledMaskCtx.getImageData(0, 0, originalImage.width, originalImage.height);

    // Apply alpha based on mask
    const outData = outCtx.getImageData(0, 0, originalImage.width, originalImage.height);
    for (let i = 0; i < originalImage.width * originalImage.height; i++) {
      const maskVal = scaledMaskData.data[i * 4] / 255;
      outData.data[i * 4 + 3] = maskVal > 0.5 ? 255 : 0;
    }
    outCtx.putImageData(outData, 0, 0);

    // Download
    outCanvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'segment_anything_cutout.png';
      a.click();
      URL.revokeObjectURL(url);
      log('Cut-out downloaded');
    }, 'image/png');
  } catch (e) {
    log(`Cut error: ${e.message}`);
    console.error(e);
  }
  isProcessing = false;
}

// ─── Device info ────────────────────────────────────────────────────────────
function updateDeviceInfo() {
  // Try to get GPU info
  const gl = document.createElement('canvas').getContext('webgl2') ||
             document.createElement('canvas').getContext('webgl');
  if (gl) {
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      backendInfo.textContent = renderer;
      log(`GPU: ${renderer}`);
    } else {
      backendInfo.textContent = 'WebGL (no debug info)';
    }
  } else {
    backendInfo.textContent = 'No WebGL';
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────
async function main() {
  log('=== Segment Anything — LiteRT.js + WebNN ===');
  log('Checking WebNN support...');

  const webnnSupported = await checkWebNN();
  if (!webnnSupported) {
    log('ERROR: WebNN is not available. Please enable:');
    log('  1. chrome://flags/#web-machine-learning-neural-network → Enabled');
    log('  2. Use a Chromium-based browser (Chrome 121+, Edge 121+)');
    statusText.textContent = 'WebNN not available';
    webnnBadge.className = 'badge badge-error';
    webnnBadge.textContent = 'WebNN ❌';
    showProgress('WebNN not available. See console for setup instructions.', 0);
    return;
  }

  log('WebNN is available ✓');
  webnnBadge.className = 'badge badge-webnn';
  webnnBadge.textContent = 'WebNN ✓';
  statusText.textContent = 'WebNN supported';

  updateDeviceInfo();

  try {
    // Initialize LiteRT.js
    await initLiteRT();

    // Load models
    await loadModels();
    log('Models loaded successfully ✓');
    statusText.textContent = 'Ready — click on image to segment';
    gpuBadge.className = 'badge badge-gpu';
    gpuBadge.textContent = 'GPU ✓';

    // Enable UI
    btnClear.disabled = false;
    canvas.style.cursor = 'crosshair';

    // Load default image if provided
    const defaultImg = new Image();
    defaultImg.crossOrigin = 'anonymous';
    defaultImg.onload = async () => {
      displayImage(defaultImg);
      await runEncoder(defaultImg);
      btnClear.disabled = false;
      btnCut.disabled = true;
    };
    defaultImg.onerror = () => {
      log('No default image. Upload an image to start.');
      hideProgress();
      statusText.textContent = 'Upload an image to begin';
    };
    defaultImg.src = 'https://raw.githubusercontent.com/microsoft/webnn-developer-preview/main/demos/segment-anything/EgyptianCat.png';

  } catch (e) {
    log(`FATAL: ${e.message}`);
    console.error(e);
    statusText.textContent = `Error: ${e.message}`;
    webnnBadge.className = 'badge badge-error';
    webnnBadge.textContent = 'ERROR';
    hideProgress();
  }
}

// ─── Event listeners ────────────────────────────────────────────────────────
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleImage(file);
});

canvas.addEventListener('click', handleClick);
canvas.addEventListener('contextmenu', handleRightClick);
btnClear.addEventListener('click', clearPoints);
btnCut.addEventListener('click', handleCut);

// ─── Start ───────────────────────────────────────────────────────────────────
main().catch((e) => {
  log(`Startup error: ${e.message}`);
  console.error(e);
  hideProgress();
});
