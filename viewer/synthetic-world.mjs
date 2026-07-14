import { SPLAT_RECORD_BYTES } from "./splat-format.mjs";

export const SYNTHETIC_WORLD_SEED = 0x4f_47_57_31;

function createRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function byte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function terrainHeight(x, z) {
  return -2.45 + x * z * 0.0018 + x * x * 0.0009 - z * z * 0.00045;
}

/**
 * Build a deterministic viewer-only Gaussian world in the strict 32-byte
 * antimatter15 layout. It is a visual fixture, not an episode or model output.
 */
export function createSyntheticWorldSplat() {
  const random = createRandom(SYNTHETIC_WORLD_SEED);
  const records = [];
  const add = ({
    position,
    scale,
    color,
    alpha = 220,
    quaternion = [255, 128, 128, 128],
  }) => {
    records.push({ position, scale, color, alpha, quaternion });
  };

  // A broad Gaussian ground surface gives the camera environmental context.
  const groundExtent = 15;
  const groundStep = 0.52;
  for (let x = -groundExtent; x <= groundExtent; x += groundStep) {
    for (let z = -groundExtent; z <= groundExtent; z += groundStep) {
      const jitterX = (random() - 0.5) * 0.12;
      const jitterZ = (random() - 0.5) * 0.12;
      const shade = Math.round(random() * 16);
      add({
        position: [x + jitterX, terrainHeight(x, z), z + jitterZ],
        scale: [0.35, 0.055, 0.35],
        color: [24 + shade, 61 + shade, 50 + Math.round(shade * 0.7)],
        alpha: 205,
      });
    }
  }

  // A warm winding route makes the scene read as navigable space.
  for (let z = -13.5; z <= 13.5; z += 0.3) {
    const centerX = z * 0.12 - z * z * z * 0.00075;
    for (let lane = -0.9; lane <= 0.9; lane += 0.3) {
      const x = centerX + lane;
      add({
        position: [x, terrainHeight(x, z) + 0.09, z],
        scale: [0.2, 0.045, 0.23],
        color: [194 + Math.round(random() * 24), 119 + Math.round(random() * 18), 58],
        alpha: 235,
      });
    }
  }

  // A shallow blue channel provides a second long-range depth cue.
  for (let z = -14; z <= 14; z += 0.34) {
    const centerX = -9.2 + z * 0.045;
    for (let lane = -0.65; lane <= 0.65; lane += 0.26) {
      const x = centerX + lane;
      add({
        position: [x, terrainHeight(x, z) + 0.075, z],
        scale: [0.2, 0.035, 0.28],
        color: [39, 119 + Math.round(random() * 20), 157 + Math.round(random() * 28)],
        alpha: 205,
      });
    }
  }

  const addBox = ({ centerX, centerZ, width, height, depth, color }) => {
    const baseY = terrainHeight(centerX, centerZ) + 0.12;
    const spacing = 0.38;
    const halfWidth = width / 2;
    const halfDepth = depth / 2;
    const wallColor = () => color.map((channel) => byte(channel + (random() - 0.5) * 18));

    for (let y = spacing / 2; y <= height; y += spacing) {
      for (let z = -halfDepth; z <= halfDepth; z += spacing) {
        for (const side of [-1, 1]) {
          add({
            position: [centerX + side * halfWidth, baseY + y, centerZ + z],
            scale: [0.075, 0.24, 0.24],
            color: wallColor(),
            alpha: 235,
          });
        }
      }
      for (let x = -halfWidth; x <= halfWidth; x += spacing) {
        for (const side of [-1, 1]) {
          const isWindow = Math.round((x + halfWidth) / spacing) % 4 === 1
            && Math.round(y / spacing) % 4 === 2;
          add({
            position: [centerX + x, baseY + y, centerZ + side * halfDepth],
            scale: [0.24, 0.24, 0.075],
            color: isWindow ? [105, 222, 211] : wallColor(),
            alpha: isWindow ? 250 : 235,
          });
        }
      }
    }

    for (let x = -halfWidth; x <= halfWidth; x += spacing) {
      for (let z = -halfDepth; z <= halfDepth; z += spacing) {
        add({
          position: [centerX + x, baseY + height, centerZ + z],
          scale: [0.24, 0.075, 0.24],
          color: color.map((channel) => byte(channel * 0.76)),
          alpha: 240,
        });
      }
    }
  };

  [
    { centerX: -5.8, centerZ: -5.5, width: 3.8, height: 4.4, depth: 3.6, color: [162, 93, 76] },
    { centerX: 6.3, centerZ: -6.4, width: 3.2, height: 6.1, depth: 3.2, color: [88, 105, 151] },
    { centerX: -5.7, centerZ: 6.8, width: 4.5, height: 3.8, depth: 3.4, color: [146, 117, 68] },
    { centerX: 7.2, centerZ: 5.8, width: 4.7, height: 5.0, depth: 3.8, color: [104, 77, 132] },
    { centerX: 10.8, centerZ: 0.4, width: 2.8, height: 3.4, depth: 3.2, color: [78, 126, 113] },
  ].forEach(addBox);

  const addTree = (x, z, height) => {
    const baseY = terrainHeight(x, z) + 0.08;
    for (let y = 0.2; y < height; y += 0.34) {
      add({
        position: [x, baseY + y, z],
        scale: [0.13, 0.28, 0.13],
        color: [87 + Math.round(random() * 14), 58 + Math.round(random() * 10), 38],
        alpha: 240,
      });
    }
    for (let index = 0; index < 42; index += 1) {
      const offsetX = (random() - 0.5) * 2.2;
      const offsetY = (random() - 0.5) * 1.7;
      const offsetZ = (random() - 0.5) * 2.2;
      const size = 0.32 + random() * 0.28;
      add({
        position: [x + offsetX, baseY + height + offsetY, z + offsetZ],
        scale: [size, size * (0.7 + random() * 0.6), size],
        color: [34 + Math.round(random() * 22), 104 + Math.round(random() * 48), 64 + Math.round(random() * 20)],
        alpha: 185 + Math.round(random() * 48),
      });
    }
  };

  [
    [-12.0, -9.5, 2.8], [-10.5, -5.2, 3.3], [-12.4, 2.1, 3.0], [-10.8, 8.7, 3.6],
    [-2.0, -9.8, 3.2], [2.7, -10.5, 2.8], [10.8, -9.4, 3.5], [12.0, 9.8, 3.2],
    [3.0, 10.2, 3.4], [-1.8, 11.4, 2.9], [-10.8, 12.1, 3.5], [12.4, -3.2, 3.1],
  ].forEach(([x, z, height]) => addTree(x, z, height));

  // A luminous portal anchors the far end of the route.
  const portalZ = 12.2;
  const portalX = portalZ * 0.12 - portalZ * portalZ * portalZ * 0.00075;
  const portalBaseY = terrainHeight(portalX, portalZ) + 0.2;
  for (let y = 0; y <= 4.4; y += 0.24) {
    for (const side of [-1, 1]) {
      add({
        position: [portalX + side * 1.45, portalBaseY + y, portalZ],
        scale: [0.14, 0.26, 0.14],
        color: side < 0 ? [92, 230, 216] : [194, 105, 232],
        alpha: 235,
      });
    }
  }
  for (let x = -1.45; x <= 1.45; x += 0.22) {
    add({
      position: [portalX + x, portalBaseY + 4.4, portalZ],
      scale: [0.25, 0.14, 0.14],
      color: [121 + Math.round((x + 1.45) * 18), 179, 227],
      alpha: 240,
    });
  }

  // Sparse translucent particles make parallax visible above the environment.
  for (let index = 0; index < 420; index += 1) {
    add({
      position: [(random() - 0.5) * 28, -0.3 + random() * 8, (random() - 0.5) * 28],
      scale: [0.045 + random() * 0.07, 0.045 + random() * 0.1, 0.045 + random() * 0.07],
      color: random() > 0.5 ? [104, 231, 214] : [201, 255, 103],
      alpha: 45 + Math.round(random() * 75),
    });
  }

  const buffer = new ArrayBuffer(records.length * SPLAT_RECORD_BYTES);
  const view = new DataView(buffer);
  records.forEach((record, index) => {
    const offset = index * SPLAT_RECORD_BYTES;
    record.position.forEach((value, component) => view.setFloat32(offset + component * 4, value, true));
    record.scale.forEach((value, component) => view.setFloat32(offset + 12 + component * 4, value, true));
    record.color.forEach((value, component) => view.setUint8(offset + 24 + component, byte(value)));
    view.setUint8(offset + 27, byte(record.alpha));
    record.quaternion.forEach((value, component) => view.setUint8(offset + 28 + component, byte(value)));
  });
  return buffer;
}
