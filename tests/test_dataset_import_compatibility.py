from __future__ import annotations

import objgauss.core.masks as legacy_core_masks
import objgauss.core.object_field as legacy_core_object_field
import objgauss.core.objectstate_bop_capture_adapter as legacy_bop_adapter
import objgauss.core.objectstate_bop_local_row_batch_handoff as legacy_bop_batch
import objgauss.core.objectstate_bop_local_row_batch_spec as legacy_bop_batch_authoring
import objgauss.core.objectstate_bop_phase1_batch_workspace as legacy_bop_batch_workspace
import objgauss.core.objectstate_bop_phase1_sample_workspace as legacy_bop_sample_workspace
import objgauss.core.objectstate_bop_phase1_subset_selector as legacy_bop_subset_selector
import objgauss.core.objectstate_controlled_real_evidence_bundle as legacy_controlled_real_bundle
import objgauss.core.objectstate_real_evidence_bundle as legacy_real_evidence_bundle
import objgauss.core.objectstate_teacher_evidence as legacy_teacher_evidence
import objgauss.core.objectstate_transition_dataset as legacy_transition_dataset
import objgauss.core.v2_stability_foundation as legacy_v2_foundation
import objgauss.masks as legacy_masks
import objgauss.object_field as legacy_object_field
import objgauss.datasets.masks as canonical_masks
import objgauss.datasets.nerf_inspection as canonical_nerf_inspection
import objgauss.datasets.objectstate_bop_capture_adapter as canonical_bop_adapter
import objgauss.datasets.objectstate_bop_local_row_batch_authoring as canonical_bop_batch_authoring
import objgauss.datasets.objectstate_bop_local_row_batch_spec as canonical_bop_batch_spec
import objgauss.datasets.objectstate_bop_phase1_batch_workspace as canonical_bop_batch_workspace
import objgauss.datasets.objectstate_bop_phase1_sample_workspace as canonical_bop_sample_workspace
import objgauss.datasets.objectstate_bop_phase1_subset_selector as canonical_bop_subset_selector
import objgauss.datasets.objectstate_controlled_real_evidence_bundle as canonical_controlled_real_bundle
import objgauss.datasets.objectstate_real_evidence_bundle as canonical_real_evidence_bundle
import objgauss.datasets.objectstate_teacher_evidence as canonical_teacher_evidence
import objgauss.datasets.objectstate_transition_dataset as canonical_transition_dataset
import objgauss.datasets.v2_stability_foundation as canonical_v2_foundation
import objgauss.pipelines.json_io as canonical_json_io


def test_mask_manifest_dataset_compatibility_surfaces_are_exact():
    expected = {
        "LEGO_COLOR_SLOTS",
        "AlphaFgBgMaskManifestResult",
        "ColorMaskManifestResult",
        "MaskManifestResult",
        "MaskManifestSplitResult",
        "MaskManifestValidationResult",
        "SamMaskManifestResult",
        "build_nerf_alpha_fgbg_mask_manifest",
        "build_nerf_alpha_mask_manifest",
        "build_nerf_rgba_color_mask_manifest",
        "build_nerf_sam_mask_manifest",
        "classify_lego_rgba",
        "read_image_rgba",
        "read_png_alpha",
        "read_png_rgba",
        "resize_rgba_max_size",
        "resolve_nerf_image",
        "slot_count_summary",
        "split_mask_manifest",
        "validate_mask_manifest",
    }
    assert set(canonical_masks.__all__) == expected
    for legacy_module in (legacy_core_masks, legacy_masks):
        assert set(legacy_module.__all__) == expected
        for name in expected:
            assert getattr(legacy_module, name) is getattr(canonical_masks, name)


def test_object_field_compatibility_surface_routes_to_canonical_owners():
    inspection_exports = {
        "NerfDatasetSummary",
        "NerfSplitSummary",
        "inspect_nerf_dataset",
    }
    primitive_exports = {
        "ObjectField",
        "ObjectFieldInit",
        "ObjectFieldLabelDelta",
        "ObjectFieldMetrics",
        "attach_hard_labels",
        "cloud_positions_for_metrics",
        "field_from_labels",
        "initialize_object_field",
        "load_object_field",
        "local_smoothness_loss",
        "object_field_label_delta",
        "object_field_metrics",
        "save_object_field",
        "softmax",
    }
    json_io_exports = {"write_json"}
    historical_exports = primitive_exports | inspection_exports | json_io_exports
    assert len(historical_exports) == 18
    assert set(canonical_nerf_inspection.__all__) == inspection_exports

    canonical_owners = {
        **{name: legacy_core_object_field for name in primitive_exports},
        **{name: canonical_nerf_inspection for name in inspection_exports},
        **{name: canonical_json_io for name in json_io_exports},
    }
    for historical_module in (legacy_core_object_field, legacy_object_field):
        assert set(historical_module.__all__) == historical_exports
        for name, canonical_module in canonical_owners.items():
            assert getattr(historical_module, name) is getattr(canonical_module, name)


def test_teacher_evidence_dataset_compatibility_surface_is_exact():
    expected = {
        "OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA",
        "OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA",
        "OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA",
        "TEACHER_EVIDENCE_SOURCES",
        "TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES",
        "TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS",
        "TEACHER_EVIDENCE_TRAINING_RISK_LEVELS",
        "TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS",
        "TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS",
        "TeacherEvidenceBatch",
        "validate_teacher_evidence_batch",
        "teacher_evidence_batch_summary",
        "validate_teacher_evidence_batch_summary",
        "objectstate_teacher_evidence_contract_summary",
        "validate_objectstate_teacher_evidence_contract_summary",
    }
    assert set(canonical_teacher_evidence.__all__) == expected
    assert set(legacy_teacher_evidence.__all__) == expected
    for name in expected:
        assert getattr(legacy_teacher_evidence, name) is getattr(
            canonical_teacher_evidence, name
        )


def test_legacy_bop_capture_adapter_preserves_canonical_object_identity():
    assert set(legacy_bop_adapter.__all__) == {
        "BOP_IDENTITY_POLICIES",
        "BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID",
        "BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID",
        "DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M",
        "OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA",
        "OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA",
        "OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA",
        "OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA",
        "objectstate_bop_capture_acceptance_summary",
        "objectstate_bop_capture_adapter_summary",
        "objectstate_bop_capture_condition_sidecar_summary",
        "objectstate_bop_capture_manifest_from_scene",
        "validate_objectstate_bop_capture_acceptance_summary",
        "validate_objectstate_bop_capture_adapter_summary",
        "validate_objectstate_bop_capture_condition_sidecar",
        "validate_objectstate_bop_capture_condition_sidecar_summary",
    }
    for name in legacy_bop_adapter.__all__:
        assert getattr(legacy_bop_adapter, name) is getattr(
            canonical_bop_adapter, name
        )


def test_legacy_bop_batch_spec_preserves_canonical_object_identity():
    spec_names = {
        "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA",
        "read_objectstate_bop_local_row_batch_spec",
        "validate_objectstate_bop_local_row_batch_spec",
    }
    assert spec_names < set(legacy_bop_batch.__all__)
    for name in spec_names:
        assert getattr(legacy_bop_batch, name) is getattr(
            canonical_bop_batch_spec, name
        )


def test_legacy_bop_batch_authoring_preserves_canonical_object_identity():
    assert set(legacy_bop_batch_authoring.__all__) == {
        "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA",
        "objectstate_bop_local_row_batch_spec_authoring",
        "validate_objectstate_bop_local_row_batch_spec_authoring_summary",
    }
    for name in legacy_bop_batch_authoring.__all__:
        assert getattr(legacy_bop_batch_authoring, name) is getattr(
            canonical_bop_batch_authoring, name
        )


def test_legacy_bop_sample_workspace_preserves_canonical_object_identity():
    assert set(legacy_bop_sample_workspace.__all__) == {
        "OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA",
        "objectstate_bop_phase1_sample_workspaces",
        "validate_objectstate_bop_phase1_sample_workspaces_summary",
    }
    for name in legacy_bop_sample_workspace.__all__:
        assert getattr(legacy_bop_sample_workspace, name) is getattr(
            canonical_bop_sample_workspace, name
        )


def test_legacy_bop_batch_workspace_preserves_canonical_object_identity():
    assert set(legacy_bop_batch_workspace.__all__) == {
        "OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA",
        "objectstate_bop_phase1_batch_workspace",
        "validate_objectstate_bop_phase1_batch_workspace_summary",
    }
    for name in legacy_bop_batch_workspace.__all__:
        assert getattr(legacy_bop_batch_workspace, name) is getattr(
            canonical_bop_batch_workspace, name
        )


def test_legacy_bop_subset_selector_preserves_canonical_object_identity():
    assert set(legacy_bop_subset_selector.__all__) == {
        "OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA",
        "objectstate_bop_phase1_subset_selector",
        "validate_objectstate_bop_phase1_subset_selector_summary",
    }
    for name in legacy_bop_subset_selector.__all__:
        assert getattr(legacy_bop_subset_selector, name) is getattr(
            canonical_bop_subset_selector, name
        )


def test_legacy_transition_dataset_preserves_canonical_object_identity():
    assert set(legacy_transition_dataset.__all__) == {
        "OBJECTSTATE_TRANSITION_DATASET_SCHEMA",
        "OBJECTSTATE_TRANSITION_ROW_SCHEMA",
        "OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA",
        "objectstate_transition_dataset_from_capture_manifest",
        "write_objectstate_transition_dataset",
        "read_objectstate_transition_dataset",
        "objectstate_transition_dataset_audit_from_path",
        "objectstate_transition_dataset_audit",
        "validate_objectstate_transition_dataset",
        "validate_objectstate_transition_dataset_audit",
    }
    for name in legacy_transition_dataset.__all__:
        assert getattr(legacy_transition_dataset, name) is getattr(
            canonical_transition_dataset, name
        )


def test_real_evidence_bundle_compatibility_surface_is_exact_and_canonical():
    assert set(legacy_real_evidence_bundle.__all__) == {
        "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA",
        "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA",
        "OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA",
        "OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA",
        "OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA",
        "OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA",
        "OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA",
        "OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA",
        "OBJECTSTATE_REAL_GATE_EVIDENCE_KINDS",
        "OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES",
        "read_objectstate_real_evidence_bundle",
        "objectstate_real_evidence_bundle_summary",
        "validate_objectstate_real_evidence_bundle",
        "validate_objectstate_real_evidence_bundle_summary",
    }
    for name in legacy_real_evidence_bundle.__all__:
        assert getattr(legacy_real_evidence_bundle, name) is getattr(
            canonical_real_evidence_bundle, name
        )


def test_controlled_real_bundle_adapter_compatibility_surface_is_exact():
    assert set(legacy_controlled_real_bundle.__all__) == {
        "OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA",
        "read_objectstate_controlled_real_evidence_bundle_adapter_summary",
        "objectstate_controlled_real_evidence_bundle_adapter_summary_from_file",
        "objectstate_controlled_real_evidence_bundle_adapter_summary",
        "objectstate_controlled_real_evidence_bundle_from_capture_manifest",
        "validate_objectstate_controlled_real_evidence_bundle_adapter_summary",
    }
    for name in legacy_controlled_real_bundle.__all__:
        assert getattr(legacy_controlled_real_bundle, name) is getattr(
            canonical_controlled_real_bundle, name
        )


def test_v2_stability_foundation_compatibility_surface_is_exact():
    assert set(legacy_v2_foundation.__all__) == {
        "V2_STABILITY_FOUNDATION_SCHEMA",
        "V2_SYNTHETIC_OBSERVATION_SCHEMA",
        "V2_STABILITY_SCENARIO_FIXTURE_SCHEMA",
        "V2_STABILITY_SCENARIO_KINDS",
        "ObjectIdentityRecord",
        "ObjectIdentityObservation",
        "ObjectIdentityOracle",
        "SyntheticWorldObject",
        "SyntheticWorldFrame",
        "SyntheticWorldState",
        "ObservationModelConfig",
        "SyntheticObservationFrame",
        "SyntheticStabilityScenarioFixture",
        "make_object_identity_oracle",
        "make_synthetic_world_state",
        "make_synthetic_stability_scenario_fixture",
        "make_synthetic_stability_scenario_suite",
        "observe_synthetic_world",
        "validate_synthetic_stability_scenario_fixture",
        "validate_object_identity_oracle",
        "validate_synthetic_world_frame",
        "validate_synthetic_world_state",
        "validate_observation_model_config",
        "validate_synthetic_observation_frame",
    }
    for name in legacy_v2_foundation.__all__:
        assert getattr(legacy_v2_foundation, name) is getattr(
            canonical_v2_foundation, name
        )
