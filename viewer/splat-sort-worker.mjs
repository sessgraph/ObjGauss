import { sortSplatIndices } from "./splat-sort.mjs";

let activeGeneration = -1;
let positions = null;

self.addEventListener("message", (event) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      activeGeneration = message.generation;
      positions = new Float32Array(message.positions);
      return;
    }

    if (message.type !== "sort" || message.generation !== activeGeneration || positions === null) {
      return;
    }

    const startedAt = performance.now();
    const indices = sortSplatIndices(positions, message.viewMatrix);
    self.postMessage(
      {
        type: "sorted",
        generation: activeGeneration,
        requestId: message.requestId,
        indices: indices.buffer,
        sortMs: performance.now() - startedAt,
      },
      [indices.buffer],
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      generation: message.generation,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
