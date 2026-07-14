#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createContractDispatcher } from "../src/pr01/contract-dispatch.mjs";

function fail(message, detail = {}) {
  process.stdout.write(`${JSON.stringify({ valid: false, message, ...detail })}\n`);
  process.exitCode = 4;
}

const [input] = process.argv.slice(2);
if (!input) {
  fail("usage: validate-pr01-document.mjs <document.json>");
} else {
  try {
    const path = resolve(input);
    const document = JSON.parse(await readFile(path, "utf8"));
    const result = createContractDispatcher()(document);
    const valid = result.valid;
    process.stdout.write(`${JSON.stringify({
      valid,
      path,
      schema_errors: result.schema_errors,
      semantic_errors: result.semantic_errors,
    })}\n`);
    if (!valid) {
      process.exitCode = 4;
    }
  } catch (error) {
    fail("document validation failed", {
      error: { name: error.name, message: error.message },
    });
  }
}
