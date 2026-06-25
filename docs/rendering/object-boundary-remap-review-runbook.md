# Object Boundary Remap Review Runbook

> Status: DEMO-005N review gate
> Last updated: 2026-06-25

This runbook defines how a sampled object-boundary remap target may enter the
reviewed allowlist consumed by policy-gated remap export.

The default action is always:

```text
keep-hard-mask
```

Adding an entry to
`docs/rendering/object-boundary-remap-reviewed-allowlist.json` is an explicit
approval to let `audit:object-boundary-remap-policy-export` patch `object_id`
for that exact `assetId:targetObjectId`, but only when the decision policy also
classifies the target as `allowlist-candidate`.

## Required Commands

Run the target-level browser policy gate first:

```bash
npm run audit:object-boundary-remap-policy
```

Remap browser audits use the fixed local preview port `5395`. If that port is
occupied, stop the occupying local preview process and rerun on `5395`; do not
switch to a new ad hoc port.

The gate writes:

```text
/tmp/objgauss-object-boundary-remap-policy/summary.md
/tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.md
/tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.json
```

If a target is approved, copy the reviewed markdown and screenshots into a
durable repository-relative evidence folder, for example:

```text
docs/rendering/remap-reviews/<asset-id>-object-<id>/
```

Do not approve targets using only transient `/tmp` paths.

## Manual Review Checklist

Approve a target only when every item below is true:

1. The policy target decision is exactly `allowlist-candidate`.
2. The original and remap-preview screenshots were inspected side by side.
3. `hiddenGaussianDelta` is negative, meaning the remap hides fewer target
   Gaussians after delete.
4. Non-target visual damage is within strict residual bounds:
   `afterDelta.coverageRatio` is at least `1.0` and no greater than `1.08`,
   `afterDelta.lumaDelta <= 0.04`, and `afterDelta.chromaDelta <= 0.04`.
5. The target is not part of a known risky pattern such as object contact
   surfaces, thin structures, fuzzy/fur boundaries, or severe occlusion.
6. The reviewer records a concrete reason, not just "looks better".
7. The owner approval block is filled with
   `decision="approved-for-policy-gated-remap"`.

Reject or leave as `review-only` if any of these are true:

- the policy target is `deny-*` or `review-only`;
- the remap hides more target Gaussians;
- the screenshot improvement is only cosmetic while non-target structure is
  damaged;
- the target needs inpainting, retraining, or reoptimization to look correct;
- the evidence came from a non-fixed port run or an unrepeatable local setup.

## Allowlist Entry Template

Use this shape for an approved target:

```json
{
  "assetId": "example-asset-local",
  "targetObjectId": 2,
  "approved": true,
  "reviewer": "reviewer-name",
  "reviewedAt": "2026-06-25",
  "reason": "Concrete visual and metric reason for allowing this exact target.",
  "ownerApproval": {
    "decision": "approved-for-policy-gated-remap",
    "approvedBy": "owner-name",
    "approvedAt": "2026-06-25"
  },
  "evidence": {
    "policyDecision": "allowlist-candidate",
    "policyReport": "docs/rendering/remap-reviews/example-asset-local-object-2/remap-decision-policy.md",
    "residualReport": "docs/rendering/remap-reviews/example-asset-local-object-2/summary.md",
    "originalScreenshot": "docs/rendering/remap-reviews/example-asset-local-object-2/original.png",
    "remapPreviewScreenshot": "docs/rendering/remap-reviews/example-asset-local-object-2/remap-preview.png",
    "hiddenGaussianDelta": -49,
    "hiddenGaussianDeltaShare": -0.008603,
    "afterDelta": {
      "coverageRatio": 1.000784,
      "lumaDelta": 0.004332,
      "chromaDelta": 0.01999
    },
    "deleteDeltaChange": {
      "coverageRatio": 0,
      "lumaDelta": 0,
      "chromaDelta": 0
    }
  }
}
```

The manifest audit requires evidence paths to be repository-relative and to
exist when an approved target is present:

```bash
npm run audit:object-boundary-remap-reviewed-allowlist-manifest
```

Then rerun the export gate:

```bash
npm run audit:object-boundary-remap-policy-export
```

Expected behavior:

- targets missing from the reviewed allowlist remain hard-mask assignments;
- targets present in the reviewed allowlist but not policy `allowlist-candidate`
  remain hard-mask assignments;
- only targets satisfying both files can produce applied remaps.
