import { colorForObject } from "./palette.js";

function boxCloud({
  objectId,
  center,
  size,
  count,
  jitter = 0.025,
  originalColor,
}) {
  const points = [];
  const [cx, cy, cz] = center;
  const [sx, sy, sz] = size;

  for (let index = 0; index < count; index += 1) {
    const face = index % 6;
    const u = seeded(index * 13 + objectId * 37) - 0.5;
    const v = seeded(index * 29 + objectId * 19) - 0.5;
    const w = seeded(index * 41 + objectId * 11) - 0.5;
    let x = u * sx;
    let y = v * sy;
    let z = w * sz;

    if (face === 0) x = sx / 2;
    if (face === 1) x = -sx / 2;
    if (face === 2) y = sy / 2;
    if (face === 3) y = -sy / 2;
    if (face === 4) z = sz / 2;
    if (face === 5) z = -sz / 2;

    points.push({
      x: cx + x + (seeded(index + 5) - 0.5) * jitter,
      y: cy + y + (seeded(index + 7) - 0.5) * jitter,
      z: cz + z + (seeded(index + 9) - 0.5) * jitter,
      objectId,
      color: originalColor,
      objectColor: colorForObject(objectId),
      opacity: 0.96,
      scale: [0.028, 0.018],
      rotation: seeded(index * 53 + objectId * 17) * Math.PI,
    });
  }
  return points;
}

function cylinderCloud({
  objectId,
  center,
  radius,
  height,
  count,
  originalColor,
}) {
  const points = [];
  const [cx, cy, cz] = center;
  for (let index = 0; index < count; index += 1) {
    const t = seeded(index * 17 + objectId * 31) * Math.PI * 2;
    const layer = seeded(index * 23 + objectId * 43) - 0.5;
    const edge = index % 5 === 0 ? 0.82 : 1;
    points.push({
      x: cx + Math.cos(t) * radius * edge,
      y: cy + layer * height,
      z: cz + Math.sin(t) * radius * edge,
      objectId,
      color: originalColor,
      objectColor: colorForObject(objectId),
      opacity: 0.94,
      scale: [0.03, 0.016],
      rotation: t,
    });
  }
  return points;
}

export function createSampleScene() {
  const points = [
    ...boxCloud({
      objectId: 0,
      center: [-1.65, 0.0, 0.15],
      size: [1.15, 1.6, 0.28],
      count: 1600,
      originalColor: [194, 86, 72],
    }),
    ...boxCloud({
      objectId: 1,
      center: [0.0, -0.05, 0.23],
      size: [1.35, 0.86, 0.22],
      count: 1400,
      originalColor: [74, 158, 150],
    }),
    ...boxCloud({
      objectId: 2,
      center: [1.42, 0.12, 0.32],
      size: [0.66, 1.48, 0.46],
      count: 1250,
      originalColor: [72, 128, 184],
    }),
    ...cylinderCloud({
      objectId: 3,
      center: [-0.55, 0.92, 0.53],
      radius: 0.28,
      height: 0.82,
      count: 900,
      originalColor: [208, 160, 82],
    }),
    ...cylinderCloud({
      objectId: 4,
      center: [0.88, 0.8, 0.28],
      radius: 0.2,
      height: 0.5,
      count: 650,
      originalColor: [128, 92, 168],
    }),
  ];

  return {
    name: "内置桌面 demo",
    points,
  };
}

function seeded(value) {
  const x = Math.sin(value * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}
