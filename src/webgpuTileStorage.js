export const WEBGPU_TILE_STORAGE_LAYOUT_VERSION = "webgpu-tile-storage-v1";

const STORAGE_BUFFER_DEFINITIONS = Object.freeze([
  {
    key: "positionRadius",
    label: "position-radius",
    elementType: "float32",
    role: "gaussian-center-radius",
  },
  {
    key: "colorOpacity",
    label: "color-opacity",
    elementType: "float32",
    role: "gaussian-color-opacity",
  },
  {
    key: "scaleRotation",
    label: "scale-rotation",
    elementType: "float32",
    role: "gaussian-scale-rotation",
  },
  {
    key: "objectIndices",
    label: "object-indices",
    elementType: "uint32",
    role: "gaussian-object-index",
  },
  {
    key: "objectState",
    label: "object-state",
    elementType: "uint32",
    role: "object-visibility-state",
  },
  {
    key: "tileCounts",
    label: "tile-counts",
    elementType: "uint32",
    role: "tile-reference-counts",
  },
  {
    key: "tileAccumulation",
    label: "tile-accumulation",
    elementType: "float32",
    role: "tile-weighted-accumulation",
  },
  {
    key: "tileResolvedRgba",
    label: "tile-resolved-rgba",
    elementType: "float32",
    role: "tile-resolve-output",
  },
  {
    key: "pixelResolvedRgba",
    label: "pixel-resolved-rgba",
    elementType: "float32",
    role: "viewport-pixel-resolve-output",
    optional: true,
  },
  {
    key: "tileEntries",
    label: "tile-entries",
    elementType: "uint32",
    role: "tile-entry-indices",
    optional: true,
  },
]);

export function describeWebGpuTileStorage(tileSmoke) {
  const descriptors = storageBufferDescriptors(tileSmoke);
  let totalByteLength = 0;
  let checksum = 2166136261;

  for (const descriptor of descriptors) {
    totalByteLength += descriptor.allocatedByteLength;
    checksum = checksumString(checksum, descriptor.key);
    checksum = checksumString(checksum, descriptor.elementType);
    checksum = checksumUint32(checksum, descriptor.elementCount);
    checksum = checksumUint32(checksum, descriptor.byteLength);
    checksum = checksumTypedArray(checksum, descriptor.source);
  }

  return {
    layoutVersion: WEBGPU_TILE_STORAGE_LAYOUT_VERSION,
    bufferCount: descriptors.length,
    totalByteLength,
    checksum: checksum.toString(16).padStart(8, "0"),
    tileEntriesIncluded: descriptors.some((descriptor) => descriptor.key === "tileEntries"),
    pixelOutputIncluded: descriptors.some((descriptor) => descriptor.key === "pixelResolvedRgba"),
    descriptors: descriptors.map(({ source, ...descriptor }) => descriptor),
  };
}

export function createWebGpuTileStorageBuffers(device, tileSmoke) {
  const descriptors = storageBufferDescriptors(tileSmoke);
  const description = describeWebGpuTileStorage(tileSmoke);
  const usage = storageBufferUsage();
  const buffers = descriptors.map((descriptor) => {
    const buffer = device.createBuffer({
      label: `objgauss-${descriptor.label}`,
      size: descriptor.allocatedByteLength,
      usage,
    });
    if (descriptor.source.byteLength > 0) {
      device.queue.writeBuffer(buffer, 0, descriptor.source);
    }
    return {
      key: descriptor.key,
      label: descriptor.label,
      byteLength: descriptor.byteLength,
      allocatedByteLength: descriptor.allocatedByteLength,
      buffer,
    };
  });
  const bufferMap = new Map(buffers.map((entry) => [entry.key, entry]));

  return {
    ...description,
    buffers,
    getBuffer(key) {
      const entry = bufferMap.get(key);
      if (!entry) throw new Error(`missing WebGPU storage buffer entry: ${key}`);
      return entry;
    },
    destroy() {
      for (const entry of buffers) {
        entry.buffer?.destroy?.();
      }
    },
  };
}

function storageBufferDescriptors(tileSmoke) {
  const sourceBuffers = tileSmoke?.buffers ?? {};
  const descriptors = [];
  for (const definition of STORAGE_BUFFER_DEFINITIONS) {
    const source = sourceBuffers[definition.key];
    if (definition.optional && !source) continue;
    if (!isTypedArray(source)) {
      throw new Error(`missing WebGPU tile storage buffer: ${definition.key}`);
    }
    descriptors.push({
      key: definition.key,
      label: definition.label,
      role: definition.role,
      elementType: definition.elementType,
      elementCount: source.length,
      byteLength: source.byteLength,
      allocatedByteLength: alignedByteLength(source.byteLength),
      usage: "storage|copy-dst|copy-src",
      source,
    });
  }
  return descriptors;
}

function storageBufferUsage() {
  const usage = globalThis.GPUBufferUsage ?? {
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    STORAGE: 0x0080,
  };
  return usage.STORAGE | usage.COPY_DST | usage.COPY_SRC;
}

function alignedByteLength(byteLength) {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function isTypedArray(value) {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function checksumTypedArray(checksum, source) {
  const bytes = new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  let next = checksum;
  for (const byte of bytes) {
    next ^= byte;
    next = Math.imul(next, 16777619) >>> 0;
  }
  return next;
}

function checksumString(checksum, value) {
  let next = checksum;
  for (let index = 0; index < value.length; index += 1) {
    next ^= value.charCodeAt(index) & 0xff;
    next = Math.imul(next, 16777619) >>> 0;
  }
  return next;
}

function checksumUint32(checksum, value) {
  let next = checksum;
  next ^= value & 0xff;
  next = Math.imul(next, 16777619) >>> 0;
  next ^= (value >>> 8) & 0xff;
  next = Math.imul(next, 16777619) >>> 0;
  next ^= (value >>> 16) & 0xff;
  next = Math.imul(next, 16777619) >>> 0;
  next ^= (value >>> 24) & 0xff;
  return Math.imul(next, 16777619) >>> 0;
}
