#!/usr/bin/env node

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createContractDispatcher } from "../contract-dispatch.mjs";
import { buildReport, evaluateCohort, EXIT_CODES, sha256, stableJson } from "./evaluator.mjs";

function argumentsFrom(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new TypeError("usage: cli.mjs --root DIR --manifest FILE --negative-cases FILE --output FILE --human FILE --checksums FILE");
    }
    result[key.slice(2)] = value;
  }
  for (const key of ["root", "manifest", "negative-cases", "output", "human", "checksums"]) {
    if (!result[key]) throw new TypeError(`missing --${key}`);
  }
  return result;
}

function artifactUri(path) {
  const normalized = path.replaceAll("\\", "/").replace(/^\.\//, "");
  if (normalized.startsWith("/") || normalized.includes("../")) {
    throw new TypeError(`artifact path must be repository-relative: ${path}`);
  }
  return normalized;
}

function humanReport(report) {
  const rows = report.checks.map((item) => `| ${item.check_id} | ${item.status} | ${item.reason_code} |`).join("\n");
  return `# PR-01D Independent Invariance Audit\n\n`
    + `- Verdict: \`${report.verdict.status}\` / \`${report.verdict.reason_code}\`\n`
    + `- Groups: ${report.counts.observed_groups}/${report.counts.expected_groups}\n`
    + `- Episodes: ${report.counts.observed_episodes}/${report.counts.expected_episodes}\n`
    + `- Mutation cases: ${report.negative_cases.length}/${report.negative_cases.length} passed\n\n`
    + `| Check | Status | Reason |\n| --- | --- | --- |\n${rows}\n\n`
    + `This report supports only the structural and semantic auditability of the scoped sibling evidence.\n`;
}

async function main() {
  const args = argumentsFrom(process.argv.slice(2));
  const negativeCases = JSON.parse(await readFile(resolve(args["negative-cases"]), "utf8"));
  const evaluation = await evaluateCohort({ root: args.root, manifestPath: args.manifest });
  const report = await buildReport({
    evaluation,
    negativeCases,
    artifacts: {
      machine_report_uri: artifactUri(args.output),
      human_report_uri: artifactUri(args.human),
      checksums_uri: artifactUri(args.checksums),
    },
  });
  const validate = createContractDispatcher();
  const validation = validate(report);
  if (!validation.valid) {
    throw new Error(`generated invariance report is contract-invalid: ${JSON.stringify(validation)}`);
  }
  const reportBytes = Buffer.from(`${stableJson(report)}\n`);
  const humanBytes = Buffer.from(humanReport(report));
  const output = resolve(args.output);
  const human = resolve(args.human);
  const checksums = resolve(args.checksums);
  await Promise.all([mkdir(dirname(output), { recursive: true }), mkdir(dirname(human), { recursive: true }), mkdir(dirname(checksums), { recursive: true })]);
  await writeFile(output, reportBytes, { flag: "wx" });
  await writeFile(human, humanBytes, { flag: "wx" });
  const checksumBytes = Buffer.from(
    `${sha256(reportBytes)}  ${artifactUri(args.output)}\n${sha256(humanBytes)}  ${artifactUri(args.human)}\n`,
  );
  await writeFile(checksums, checksumBytes, { flag: "wx" });
  process.stdout.write(`${JSON.stringify({
    verdict: report.verdict,
    counts: report.counts,
    report_sha256: sha256(reportBytes),
    exit_code: EXIT_CODES[report.verdict.status],
  })}\n`);
  process.exitCode = EXIT_CODES[report.verdict.status];
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
