import { inflateSync } from "node:zlib";

export async function canvasVisualStats(page, selector, options = {}) {
  const locator = page.locator(selector).first();
  const timeout = options.timeoutMs ?? 15000;
  await locator.waitFor({ timeout });
  const buffer = options.usePageClip
    ? await page.screenshot({
        animations: "disabled",
        clip: await canvasScreenshotClip(locator),
        timeout,
      })
    : await locator.screenshot({ animations: "disabled", timeout });
  return visualStatsFromPng(buffer);
}

async function canvasScreenshotClip(locator) {
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error("cannot capture canvas visual stats: missing bounding box");
  }
  return {
    x: Math.max(0, Math.floor(box.x)),
    y: Math.max(0, Math.floor(box.y)),
    width: Math.max(1, Math.floor(box.width)),
    height: Math.max(1, Math.floor(box.height)),
  };
}

export function visualStatsFromPng(buffer) {
  const image = decodePng(buffer);
  const totalPixels = image.width * image.height;
  let nonBackgroundPixels = 0;
  let lumaSum = 0;
  let chromaSum = 0;
  let checksum = 2166136261;

  for (let pixelIndex = 0; pixelIndex < totalPixels; pixelIndex += 1) {
    const pixel = pngPixelAt(image, pixelIndex);
    checksum = checksumByte(checksum, pixel.red);
    checksum = checksumByte(checksum, pixel.green);
    checksum = checksumByte(checksum, pixel.blue);
    checksum = checksumByte(checksum, pixel.alpha);
    if (
      pixel.alpha <= 0 ||
      Math.abs(pixel.red - 16) + Math.abs(pixel.green - 19) + Math.abs(pixel.blue - 22) <= 10
    ) {
      continue;
    }
    const luma = (0.2126 * pixel.red + 0.7152 * pixel.green + 0.0722 * pixel.blue) / 255;
    const chroma = (Math.max(pixel.red, pixel.green, pixel.blue) - Math.min(pixel.red, pixel.green, pixel.blue)) / 255;
    nonBackgroundPixels += 1;
    lumaSum += luma;
    chromaSum += chroma;
  }

  return {
    width: image.width,
    height: image.height,
    pixels: totalPixels,
    nonBackgroundPixels,
    coverage: roundMetric(totalPixels > 0 ? nonBackgroundPixels / totalPixels : 0),
    lumaMean: roundMetric(nonBackgroundPixels > 0 ? lumaSum / nonBackgroundPixels : 0),
    chromaMean: roundMetric(nonBackgroundPixels > 0 ? chromaSum / nonBackgroundPixels : 0),
    checksum: checksum.toString(16).padStart(8, "0"),
  };
}

export function validateCanvasVisualStats(assetId, label, stats) {
  if (
    !stats ||
    stats.width <= 0 ||
    stats.height <= 0 ||
    stats.pixels <= 0 ||
    stats.nonBackgroundPixels <= 0 ||
    stats.coverage <= 0 ||
    !/^[0-9a-f]{8}$/.test(stats.checksum)
  ) {
    throw new Error(
      `${assetId} ${label} canvas visual stats are invalid: ${JSON.stringify(stats)}`,
    );
  }
}

export function compareVisualStats(referenceStats, candidateStats) {
  return {
    coverageRatio: roundMetric(candidateStats.coverage / Math.max(referenceStats.coverage, 0.000001)),
    lumaDelta: roundMetric(Math.abs(candidateStats.lumaMean - referenceStats.lumaMean)),
    chromaDelta: roundMetric(Math.abs(candidateStats.chromaMean - referenceStats.chromaMean)),
  };
}

export function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function decodePng(buffer) {
  const source = Buffer.from(buffer);
  const signature = "89504e470d0a1a0a";
  if (source.subarray(0, 8).toString("hex") !== signature) {
    throw new Error("unsupported screenshot format: expected PNG signature");
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatChunks = [];
  while (offset < source.length) {
    const length = source.readUInt32BE(offset);
    const type = source.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (type === "IHDR") {
      width = source.readUInt32BE(dataStart);
      height = source.readUInt32BE(dataStart + 4);
      bitDepth = source[dataStart + 8];
      colorType = source[dataStart + 9];
    } else if (type === "IDAT") {
      idatChunks.push(source.subarray(dataStart, dataEnd));
    } else if (type === "IEND") {
      break;
    }
    offset = dataEnd + 4;
  }

  if (width <= 0 || height <= 0 || bitDepth !== 8) {
    throw new Error(`unsupported PNG dimensions or bit depth: ${width}x${height}:${bitDepth}`);
  }
  const bytesPerPixel = pngBytesPerPixel(colorType);
  const inflated = inflateSync(Buffer.concat(idatChunks));
  const stride = width * bytesPerPixel;
  const data = new Uint8Array(height * stride);
  let sourceOffset = 0;
  let targetOffset = 0;
  let previous = new Uint8Array(stride);
  for (let y = 0; y < height; y += 1) {
    const filter = inflated[sourceOffset];
    sourceOffset += 1;
    const row = inflated.subarray(sourceOffset, sourceOffset + stride);
    sourceOffset += stride;
    const output = data.subarray(targetOffset, targetOffset + stride);
    unfilterPngRow({ filter, row, output, previous, bytesPerPixel });
    previous = output;
    targetOffset += stride;
  }
  return { width, height, colorType, bytesPerPixel, data };
}

function pngBytesPerPixel(colorType) {
  if (colorType === 0) return 1;
  if (colorType === 2) return 3;
  if (colorType === 4) return 2;
  if (colorType === 6) return 4;
  throw new Error(`unsupported PNG color type: ${colorType}`);
}

function pngPixelAt(image, pixelIndex) {
  const offset = pixelIndex * image.bytesPerPixel;
  if (image.colorType === 0) {
    const value = image.data[offset];
    return { red: value, green: value, blue: value, alpha: 255 };
  }
  if (image.colorType === 2) {
    return {
      red: image.data[offset],
      green: image.data[offset + 1],
      blue: image.data[offset + 2],
      alpha: 255,
    };
  }
  if (image.colorType === 4) {
    const value = image.data[offset];
    return { red: value, green: value, blue: value, alpha: image.data[offset + 1] };
  }
  return {
    red: image.data[offset],
    green: image.data[offset + 1],
    blue: image.data[offset + 2],
    alpha: image.data[offset + 3],
  };
}

function unfilterPngRow({ filter, row, output, previous, bytesPerPixel }) {
  for (let index = 0; index < row.length; index += 1) {
    const left = index >= bytesPerPixel ? output[index - bytesPerPixel] : 0;
    const up = previous[index] ?? 0;
    const upLeft = index >= bytesPerPixel ? previous[index - bytesPerPixel] : 0;
    let predictor = 0;
    if (filter === 1) predictor = left;
    else if (filter === 2) predictor = up;
    else if (filter === 3) predictor = Math.floor((left + up) / 2);
    else if (filter === 4) predictor = paethPredictor(left, up, upLeft);
    else if (filter !== 0) throw new Error(`unsupported PNG filter: ${filter}`);
    output[index] = (row[index] + predictor) & 0xff;
  }
}

function paethPredictor(left, up, upLeft) {
  const estimate = left + up - upLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const upLeftDistance = Math.abs(estimate - upLeft);
  if (leftDistance <= upDistance && leftDistance <= upLeftDistance) return left;
  if (upDistance <= upLeftDistance) return up;
  return upLeft;
}

function checksumByte(checksum, value) {
  const next = checksum ^ (value & 0xff);
  return Math.imul(next, 16777619) >>> 0;
}
