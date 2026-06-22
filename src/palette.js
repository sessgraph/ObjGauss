export const OBJECT_PALETTE = [
  [230, 57, 70],
  [42, 157, 143],
  [69, 123, 157],
  [244, 162, 97],
  [131, 56, 236],
  [255, 190, 11],
  [29, 53, 87],
  [138, 201, 38],
  [255, 89, 94],
  [25, 130, 196],
  [106, 76, 147],
  [255, 202, 58],
];

export function colorForObject(id) {
  if (id >= 0 && id < OBJECT_PALETTE.length) {
    return OBJECT_PALETTE[id];
  }
  const value = (id * 2654435761) >>> 0;
  return [
    64 + (value & 0x7f),
    64 + ((value >> 8) & 0x7f),
    64 + ((value >> 16) & 0x7f),
  ];
}

export function rgbToCss(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
