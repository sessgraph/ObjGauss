from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    AssignmentEvidenceBatch,
    validate_assignment_evidence_batch,
)

V2_STABILITY_FOUNDATION_SCHEMA = "objgauss-v2-stability-foundation-v1"
V2_SYNTHETIC_OBSERVATION_SCHEMA = "objgauss-v2-synthetic-observation-v1"
V2_STABILITY_SCENARIO_FIXTURE_SCHEMA = "objgauss-v2-stability-scenario-fixture-v1"
V2_STABILITY_SCENARIO_KINDS = (
    "cross_view",
    "occlusion_recovery",
    "perturbation",
    "adversarial_swap",
)
_VALID_SCENARIO_KINDS = {
    *V2_STABILITY_SCENARIO_KINDS,
}
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectIdentityRecord:
    oracle_object_id: int
    lineage_id: str
    canonical_slot: int
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle_object_id": int(self.oracle_object_id),
            "lineage_id": self.lineage_id,
            "canonical_slot": int(self.canonical_slot),
            "label": self.label,
        }


@dataclass(frozen=True)
class ObjectIdentityObservation:
    oracle_object_id: int
    lineage_id: str
    frame_index: int
    visible: bool
    expected_slot: int
    expected_slot_relation: str = "same_lineage"

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle_object_id": int(self.oracle_object_id),
            "lineage_id": self.lineage_id,
            "frame_index": int(self.frame_index),
            "visible": bool(self.visible),
            "expected_slot": int(self.expected_slot),
            "expected_slot_relation": self.expected_slot_relation,
        }


@dataclass(frozen=True)
class ObjectIdentityOracle:
    scenario_id: str
    identities: tuple[ObjectIdentityRecord, ...]
    frames: tuple[tuple[ObjectIdentityObservation, ...], ...]
    schema: str = V2_STABILITY_FOUNDATION_SCHEMA

    @property
    def object_count(self) -> int:
        return len(self.identities)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def slots(self) -> int:
        return len({identity.canonical_slot for identity in self.identities})

    def as_dict(self) -> dict[str, Any]:
        oracle = validate_object_identity_oracle(self)
        return {
            "schema": oracle.schema,
            "kind": "object_identity_oracle",
            "scenario_id": oracle.scenario_id,
            "object_count": oracle.object_count,
            "frame_count": oracle.frame_count,
            "slots": oracle.slots,
            "identity_source": "synthetic_oracle_labels",
            "slot_source": "canonical_oracle_slot",
            "has_occlusion": any(
                not observation.visible
                for frame in oracle.frames
                for observation in frame
            ),
            "identities": [identity.as_dict() for identity in oracle.identities],
            "frames": [
                [observation.as_dict() for observation in frame]
                for frame in oracle.frames
            ],
        }


@dataclass(frozen=True)
class SyntheticWorldObject:
    oracle_object_id: int
    lineage_id: str
    frame_index: int
    pose_center: np.ndarray
    appearance_feature: np.ndarray
    appearance_rgb: np.ndarray
    trajectory: np.ndarray
    visible: bool
    perturbation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        obj = _validate_world_object(self)
        return {
            "oracle_object_id": int(obj.oracle_object_id),
            "lineage_id": obj.lineage_id,
            "frame_index": int(obj.frame_index),
            "pose_center": np.round(obj.pose_center, 6).tolist(),
            "appearance_feature": np.round(obj.appearance_feature, 6).tolist(),
            "appearance_rgb": np.round(obj.appearance_rgb, 6).tolist(),
            "trajectory": np.round(obj.trajectory, 6).tolist(),
            "visible": bool(obj.visible),
            "perturbation": obj.perturbation,
        }


@dataclass(frozen=True)
class SyntheticWorldFrame:
    frame_index: int
    view_id: str
    objects: tuple[SyntheticWorldObject, ...]
    perturbation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        frame = validate_synthetic_world_frame(self)
        return {
            "frame_index": int(frame.frame_index),
            "view_id": frame.view_id,
            "object_count": len(frame.objects),
            "visible_object_count": sum(1 for obj in frame.objects if obj.visible),
            "perturbation": frame.perturbation,
            "objects": [obj.as_dict() for obj in frame.objects],
        }


@dataclass(frozen=True)
class SyntheticWorldState:
    scenario_id: str
    scenario_kind: str
    oracle: ObjectIdentityOracle
    frames: tuple[SyntheticWorldFrame, ...]
    schema: str = V2_STABILITY_FOUNDATION_SCHEMA

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def object_count(self) -> int:
        return self.oracle.object_count

    def as_dict(self) -> dict[str, Any]:
        world = validate_synthetic_world_state(self)
        return {
            "schema": world.schema,
            "kind": "synthetic_world_state",
            "scenario_id": world.scenario_id,
            "scenario_kind": world.scenario_kind,
            "object_count": world.object_count,
            "frame_count": world.frame_count,
            "identity_contract": {
                "oracle": "ObjectIdentityOracle",
                "object_id_field": "oracle_object_id",
                "lineage_field": "lineage_id",
                "slot_field": "canonical_slot",
            },
            "oracle": world.oracle.as_dict(),
            "frames": [frame.as_dict() for frame in world.frames],
        }


@dataclass(frozen=True)
class ObservationModelConfig:
    points_per_object: int = 3
    position_jitter: float = 0.01
    feature_noise: float = 0.0
    include_mask_votes: bool = True
    include_track_hints: bool = True
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        config = validate_observation_model_config(self)
        return {
            "points_per_object": int(config.points_per_object),
            "position_jitter": float(config.position_jitter),
            "feature_noise": float(config.feature_noise),
            "include_mask_votes": bool(config.include_mask_votes),
            "include_track_hints": bool(config.include_track_hints),
            "seed": int(config.seed),
        }


@dataclass(frozen=True)
class SyntheticObservationFrame:
    frame_index: int
    view_id: str
    evidence: AssignmentEvidenceBatch
    oracle_object_ids: np.ndarray
    lineage_ids: tuple[str, ...]
    expected_slots: np.ndarray
    schema: str = V2_SYNTHETIC_OBSERVATION_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        frame = validate_synthetic_observation_frame(self)
        return {
            "schema": frame.schema,
            "kind": "synthetic_observation_frame",
            "frame_index": int(frame.frame_index),
            "view_id": frame.view_id,
            "evidence_count": frame.evidence.evidence_count,
            "lineage_ids": list(frame.lineage_ids),
            "oracle_object_ids": frame.oracle_object_ids.astype(int).tolist(),
            "expected_slots": frame.expected_slots.astype(int).tolist(),
            "evidence": frame.evidence.as_dict(),
        }


@dataclass(frozen=True)
class SyntheticStabilityScenarioFixture:
    scenario_id: str
    scenario_kind: str
    world: SyntheticWorldState
    observations: tuple[SyntheticObservationFrame, ...]
    observation_config: ObservationModelConfig
    world_seed: int
    schema: str = V2_STABILITY_SCENARIO_FIXTURE_SCHEMA

    @property
    def frame_count(self) -> int:
        return self.world.frame_count

    @property
    def object_count(self) -> int:
        return self.world.object_count

    def as_dict(self) -> dict[str, Any]:
        fixture = validate_synthetic_stability_scenario_fixture(self)
        return {
            "schema": fixture.schema,
            "kind": "synthetic_stability_scenario_fixture",
            "scenario_id": fixture.scenario_id,
            "scenario_kind": fixture.scenario_kind,
            "object_count": fixture.object_count,
            "frame_count": fixture.frame_count,
            "observation_count": len(fixture.observations),
            "identity_source": "synthetic_oracle_labels",
            "expected_slot_source": "canonical_oracle_slot",
            "scenario_role": "fixture_only_not_final_gate",
            "world_seed": int(fixture.world_seed),
            "observation_config": fixture.observation_config.as_dict(),
            "visibility_transitions": _visibility_transition_records(fixture.world.oracle),
            "oracle": fixture.world.oracle.as_dict(),
            "world": fixture.world.as_dict(),
            "observations": [observation.as_dict() for observation in fixture.observations],
        }


def make_object_identity_oracle(
    *,
    scenario_id: str,
    object_count: int = 3,
    frame_count: int = 2,
    occluded_object_ids: Sequence[int] = (),
    occluded_frame_indices: Sequence[int] = (),
) -> ObjectIdentityOracle:
    if object_count < 1:
        raise ValueError("object_count must be >= 1")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    occluded_ids = {int(value) for value in occluded_object_ids}
    occluded_frames = {int(value) for value in occluded_frame_indices}
    identities = tuple(
        ObjectIdentityRecord(
            oracle_object_id=index,
            lineage_id=f"lineage-{index:04d}",
            canonical_slot=index,
            label=f"synthetic-object-{index}",
        )
        for index in range(object_count)
    )
    frames: list[tuple[ObjectIdentityObservation, ...]] = []
    for frame_index in range(frame_count):
        observations = []
        for identity in identities:
            visible = not (
                identity.oracle_object_id in occluded_ids
                and frame_index in occluded_frames
            )
            observations.append(
                ObjectIdentityObservation(
                    oracle_object_id=identity.oracle_object_id,
                    lineage_id=identity.lineage_id,
                    frame_index=frame_index,
                    visible=visible,
                    expected_slot=identity.canonical_slot,
                    expected_slot_relation="same_lineage" if visible else "occluded",
                )
            )
        frames.append(tuple(observations))
    return validate_object_identity_oracle(
        ObjectIdentityOracle(
            scenario_id=str(scenario_id),
            identities=identities,
            frames=tuple(frames),
        )
    )


def make_synthetic_world_state(
    *,
    scenario_id: str,
    scenario_kind: str = "cross_view",
    object_count: int = 3,
    frame_count: int = 2,
    feature_dim: int = 6,
    seed: int = 0,
) -> SyntheticWorldState:
    if scenario_kind not in _VALID_SCENARIO_KINDS:
        raise ValueError(f"unsupported scenario_kind: {scenario_kind}")
    if feature_dim < object_count:
        raise ValueError("feature_dim must be >= object_count for oracle one-hot features")
    occluded_object_ids: tuple[int, ...] = ()
    occluded_frame_indices: tuple[int, ...] = ()
    if scenario_kind == "occlusion_recovery" and frame_count >= 3:
        occluded_object_ids = (0,)
        occluded_frame_indices = tuple(range(1, frame_count - 1))
    oracle = make_object_identity_oracle(
        scenario_id=scenario_id,
        object_count=object_count,
        frame_count=frame_count,
        occluded_object_ids=occluded_object_ids,
        occluded_frame_indices=occluded_frame_indices,
    )
    rng = np.random.default_rng(seed)
    canonical_centers = _canonical_centers(object_count)
    frames = []
    for frame_index, oracle_frame in enumerate(oracle.frames):
        centered_index = frame_index - (frame_count - 1) / 2.0
        objects = []
        for observation in oracle_frame:
            appearance_object_id = _appearance_object_id_for_scenario(
                scenario_kind,
                frame_index=frame_index,
                object_id=observation.oracle_object_id,
                object_count=object_count,
            )
            feature = np.zeros(feature_dim, dtype=np.float32)
            feature[appearance_object_id] = 1.0
            rgb = _canonical_rgb(appearance_object_id)
            trajectory = np.asarray(
                [
                    0.03 * (observation.oracle_object_id + 1),
                    0.01 * ((observation.oracle_object_id % 2) * 2 - 1),
                    0.0,
                ],
                dtype=np.float32,
            )
            pose_center = canonical_centers[observation.oracle_object_id] + trajectory * centered_index
            perturbation = _object_perturbation(
                scenario_kind,
                frame_index=frame_index,
                object_id=observation.oracle_object_id,
                rng=rng,
            )
            if appearance_object_id != observation.oracle_object_id:
                perturbation = {
                    **perturbation,
                    "appearance_source_object_id": int(appearance_object_id),
                }
            objects.append(
                SyntheticWorldObject(
                    oracle_object_id=observation.oracle_object_id,
                    lineage_id=observation.lineage_id,
                    frame_index=frame_index,
                    pose_center=pose_center.astype(np.float32, copy=False),
                    appearance_feature=_apply_feature_perturbation(feature, perturbation),
                    appearance_rgb=_apply_rgb_perturbation(rgb, perturbation),
                    trajectory=trajectory,
                    visible=observation.visible,
                    perturbation=perturbation,
                )
            )
        frames.append(
            SyntheticWorldFrame(
                frame_index=frame_index,
                view_id=_view_id(scenario_kind, frame_index),
                objects=tuple(objects),
                perturbation=_frame_perturbation(scenario_kind, frame_index),
            )
        )
    return validate_synthetic_world_state(
        SyntheticWorldState(
            scenario_id=scenario_id,
            scenario_kind=scenario_kind,
            oracle=oracle,
            frames=tuple(frames),
        )
    )


def make_synthetic_stability_scenario_fixture(
    *,
    scenario_kind: str,
    scenario_id: str | None = None,
    object_count: int = 3,
    frame_count: int | None = None,
    feature_dim: int | None = None,
    seed: int = 0,
    observation_config: ObservationModelConfig | None = None,
) -> SyntheticStabilityScenarioFixture:
    if scenario_kind not in _VALID_SCENARIO_KINDS:
        raise ValueError(f"unsupported scenario_kind: {scenario_kind}")
    if object_count < 1:
        raise ValueError("object_count must be >= 1")
    world_seed = int(seed)
    resolved_frame_count = (
        _default_scenario_frame_count(scenario_kind)
        if frame_count is None
        else int(frame_count)
    )
    resolved_feature_dim = max(object_count, 6) if feature_dim is None else int(feature_dim)
    resolved_scenario_id = scenario_id or f"v2-stability-{scenario_kind}"
    config = observation_config or _default_observation_config(world_seed)
    world = make_synthetic_world_state(
        scenario_id=resolved_scenario_id,
        scenario_kind=scenario_kind,
        object_count=object_count,
        frame_count=resolved_frame_count,
        feature_dim=resolved_feature_dim,
        seed=world_seed,
    )
    observations = observe_synthetic_world(world, config=config)
    return validate_synthetic_stability_scenario_fixture(
        SyntheticStabilityScenarioFixture(
            scenario_id=resolved_scenario_id,
            scenario_kind=scenario_kind,
            world=world,
            observations=observations,
            observation_config=config,
            world_seed=world_seed,
        )
    )


def make_synthetic_stability_scenario_suite(
    *,
    object_count: int = 3,
    seed: int = 0,
) -> tuple[SyntheticStabilityScenarioFixture, ...]:
    return tuple(
        make_synthetic_stability_scenario_fixture(
            scenario_kind=scenario_kind,
            object_count=object_count,
            seed=int(seed) + index,
        )
        for index, scenario_kind in enumerate(V2_STABILITY_SCENARIO_KINDS)
    )


def observe_synthetic_world(
    world: SyntheticWorldState,
    *,
    config: ObservationModelConfig | None = None,
) -> tuple[SyntheticObservationFrame, ...]:
    world = validate_synthetic_world_state(world)
    config = validate_observation_model_config(config or ObservationModelConfig())
    rng = np.random.default_rng(config.seed)
    observed = []
    slot_count = world.oracle.slots
    for frame in world.frames:
        positions: list[np.ndarray] = []
        features: list[np.ndarray] = []
        oracle_object_ids: list[int] = []
        lineage_ids: list[str] = []
        expected_slots: list[int] = []
        for obj in frame.objects:
            if not obj.visible:
                continue
            identity = _identity_by_object_id(world.oracle)[obj.oracle_object_id]
            offsets = _point_offsets(config.points_per_object, scale=config.position_jitter)
            for offset in offsets:
                jitter = rng.normal(0.0, config.position_jitter * 0.1, size=3).astype(np.float32)
                position = _apply_view_transform(
                    obj.pose_center + offset + jitter,
                    view_id=frame.view_id,
                )
                feature = obj.appearance_feature.copy()
                if config.feature_noise > 0:
                    feature = feature + rng.normal(0.0, config.feature_noise, size=feature.shape).astype(np.float32)
                positions.append(position.astype(np.float32, copy=False))
                features.append(feature.astype(np.float32, copy=False))
                oracle_object_ids.append(obj.oracle_object_id)
                lineage_ids.append(obj.lineage_id)
                expected_slots.append(identity.canonical_slot)
        if not positions:
            raise ValueError("observation frame must contain at least one visible object")
        expected = np.asarray(expected_slots, dtype=np.int64)
        target_assignment = np.zeros((expected.shape[0], slot_count), dtype=np.float32)
        target_assignment[np.arange(expected.shape[0]), expected] = 1.0
        mask_votes = target_assignment.copy() if config.include_mask_votes else None
        track_hints = np.asarray(oracle_object_ids, dtype=np.int64) if config.include_track_hints else None
        evidence = validate_assignment_evidence_batch(
            AssignmentEvidenceBatch(
                positions=np.vstack(positions).astype(np.float32, copy=False),
                features=np.vstack(features).astype(np.float32, copy=False),
                frame_index=frame.frame_index,
                mask_votes=mask_votes,
                track_hints=track_hints,
                target_assignment=target_assignment,
                source=f"synthetic_world:{world.scenario_id}:{world.scenario_kind}",
            )
        )
        observed.append(
            validate_synthetic_observation_frame(
                SyntheticObservationFrame(
                    frame_index=frame.frame_index,
                    view_id=frame.view_id,
                    evidence=evidence,
                    oracle_object_ids=np.asarray(oracle_object_ids, dtype=np.int64),
                    lineage_ids=tuple(lineage_ids),
                    expected_slots=expected,
                )
            )
        )
    return tuple(observed)


def validate_synthetic_stability_scenario_fixture(
    fixture: SyntheticStabilityScenarioFixture,
) -> SyntheticStabilityScenarioFixture:
    if not isinstance(fixture, SyntheticStabilityScenarioFixture):
        raise TypeError("fixture must be SyntheticStabilityScenarioFixture")
    if fixture.schema != V2_STABILITY_SCENARIO_FIXTURE_SCHEMA:
        raise ValueError(f"unsupported stability scenario fixture schema: {fixture.schema}")
    if fixture.scenario_kind not in _VALID_SCENARIO_KINDS:
        raise ValueError(f"unsupported scenario_kind: {fixture.scenario_kind}")
    world = validate_synthetic_world_state(fixture.world)
    config = validate_observation_model_config(fixture.observation_config)
    if fixture.scenario_id != world.scenario_id:
        raise ValueError("fixture scenario_id must match world scenario_id")
    if fixture.scenario_kind != world.scenario_kind:
        raise ValueError("fixture scenario_kind must match world scenario_kind")
    if len(fixture.observations) != world.frame_count:
        raise ValueError("fixture observations must cover every world frame")
    validated_observations = tuple(
        validate_synthetic_observation_frame(observation)
        for observation in fixture.observations
    )
    for frame_index, observation in enumerate(validated_observations):
        if int(observation.frame_index) != frame_index:
            raise ValueError("fixture observations must be ordered by frame_index")
        if observation.view_id != world.frames[frame_index].view_id:
            raise ValueError("observation view_id must match world frame view_id")
        oracle_frame = world.oracle.frames[frame_index]
        visible_ids = {
            int(oracle_observation.oracle_object_id)
            for oracle_observation in oracle_frame
            if oracle_observation.visible
        }
        observed_ids = set(observation.oracle_object_ids.astype(int).tolist())
        if observed_ids != visible_ids:
            raise ValueError("observation oracle ids must match visible oracle objects")
        slots_by_id = {
            int(oracle_observation.oracle_object_id): int(oracle_observation.expected_slot)
            for oracle_observation in oracle_frame
        }
        for object_id, expected_slot in zip(
            observation.oracle_object_ids.astype(int).tolist(),
            observation.expected_slots.astype(int).tolist(),
        ):
            if int(expected_slot) != slots_by_id[int(object_id)]:
                raise ValueError("observation expected slots must match oracle slots")
    return SyntheticStabilityScenarioFixture(
        scenario_id=fixture.scenario_id,
        scenario_kind=fixture.scenario_kind,
        world=world,
        observations=validated_observations,
        observation_config=config,
        world_seed=int(fixture.world_seed),
        schema=fixture.schema,
    )


def validate_object_identity_oracle(oracle: ObjectIdentityOracle) -> ObjectIdentityOracle:
    if not isinstance(oracle, ObjectIdentityOracle):
        raise TypeError("oracle must be ObjectIdentityOracle")
    if oracle.schema != V2_STABILITY_FOUNDATION_SCHEMA:
        raise ValueError(f"unsupported identity oracle schema: {oracle.schema}")
    if not oracle.scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if not oracle.identities:
        raise ValueError("oracle must contain at least one identity")
    if not oracle.frames:
        raise ValueError("oracle must contain at least one frame")
    object_ids = [int(identity.oracle_object_id) for identity in oracle.identities]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("oracle_object_id values must be unique")
    lineage_ids = [identity.lineage_id for identity in oracle.identities]
    if len(set(lineage_ids)) != len(lineage_ids):
        raise ValueError("lineage_id values must be unique")
    slots = [int(identity.canonical_slot) for identity in oracle.identities]
    if sorted(slots) != list(range(len(slots))):
        raise ValueError("canonical slots must be contiguous and unique from 0")
    by_object_id = {identity.oracle_object_id: identity for identity in oracle.identities}
    for frame_index, frame in enumerate(oracle.frames):
        if len(frame) != len(oracle.identities):
            raise ValueError("each oracle frame must observe every identity as visible or occluded")
        seen: set[int] = set()
        for observation in frame:
            identity = by_object_id.get(int(observation.oracle_object_id))
            if identity is None:
                raise ValueError("oracle frame references unknown object id")
            if int(observation.frame_index) != frame_index:
                raise ValueError("oracle observation frame_index must match its frame")
            if observation.lineage_id != identity.lineage_id:
                raise ValueError("oracle observation lineage_id must match identity")
            if int(observation.expected_slot) != int(identity.canonical_slot):
                raise ValueError("expected_slot must match identity canonical_slot")
            if observation.expected_slot_relation == "occluded" and observation.visible:
                raise ValueError("visible observation cannot use occluded expected_slot_relation")
            if observation.expected_slot_relation != "occluded" and not observation.visible:
                raise ValueError("occluded observation must use occluded expected_slot_relation")
            seen.add(int(observation.oracle_object_id))
        if seen != set(object_ids):
            raise ValueError("oracle frame must cover every object id exactly once")
    return oracle


def validate_synthetic_world_frame(frame: SyntheticWorldFrame) -> SyntheticWorldFrame:
    if not isinstance(frame, SyntheticWorldFrame):
        raise TypeError("frame must be SyntheticWorldFrame")
    if int(frame.frame_index) < 0:
        raise ValueError("frame_index must be >= 0")
    if not frame.view_id:
        raise ValueError("view_id must be non-empty")
    if not frame.objects:
        raise ValueError("synthetic world frame must contain objects")
    for obj in frame.objects:
        _validate_world_object(obj)
        if int(obj.frame_index) != int(frame.frame_index):
            raise ValueError("world object frame_index must match frame")
    return frame


def validate_synthetic_world_state(world: SyntheticWorldState) -> SyntheticWorldState:
    if not isinstance(world, SyntheticWorldState):
        raise TypeError("world must be SyntheticWorldState")
    if world.schema != V2_STABILITY_FOUNDATION_SCHEMA:
        raise ValueError(f"unsupported synthetic world schema: {world.schema}")
    if world.scenario_kind not in _VALID_SCENARIO_KINDS:
        raise ValueError(f"unsupported scenario_kind: {world.scenario_kind}")
    oracle = validate_object_identity_oracle(world.oracle)
    if world.scenario_id != oracle.scenario_id:
        raise ValueError("world scenario_id must match oracle scenario_id")
    if len(world.frames) != oracle.frame_count:
        raise ValueError("world frame count must match oracle frame count")
    identity_by_id = _identity_by_object_id(oracle)
    for index, frame in enumerate(world.frames):
        frame = validate_synthetic_world_frame(frame)
        if frame.frame_index != index:
            raise ValueError("world frames must be ordered by frame_index")
        if len(frame.objects) != oracle.object_count:
            raise ValueError("world frame must contain every oracle object")
        oracle_observations = {
            int(observation.oracle_object_id): observation
            for observation in oracle.frames[index]
        }
        for obj in frame.objects:
            identity = identity_by_id.get(int(obj.oracle_object_id))
            if identity is None:
                raise ValueError("world object references unknown oracle object")
            if obj.lineage_id != identity.lineage_id:
                raise ValueError("world object lineage_id must match oracle identity")
            oracle_observation = oracle_observations[int(identity.oracle_object_id)]
            if bool(obj.visible) != bool(oracle_observation.visible):
                raise ValueError("world object visibility must match oracle observation")
    return world


def validate_observation_model_config(config: ObservationModelConfig) -> ObservationModelConfig:
    if not isinstance(config, ObservationModelConfig):
        raise TypeError("config must be ObservationModelConfig")
    if config.points_per_object < 1:
        raise ValueError("points_per_object must be >= 1")
    if config.position_jitter < 0:
        raise ValueError("position_jitter must be >= 0")
    if config.feature_noise < 0:
        raise ValueError("feature_noise must be >= 0")
    return config


def validate_synthetic_observation_frame(frame: SyntheticObservationFrame) -> SyntheticObservationFrame:
    if not isinstance(frame, SyntheticObservationFrame):
        raise TypeError("frame must be SyntheticObservationFrame")
    if frame.schema != V2_SYNTHETIC_OBSERVATION_SCHEMA:
        raise ValueError(f"unsupported synthetic observation schema: {frame.schema}")
    evidence = validate_assignment_evidence_batch(frame.evidence)
    oracle_ids = _int_vector(frame.oracle_object_ids, "oracle_object_ids")
    expected_slots = _int_vector(frame.expected_slots, "expected_slots")
    if oracle_ids.shape[0] != evidence.evidence_count:
        raise ValueError("oracle_object_ids length must match evidence rows")
    if expected_slots.shape[0] != evidence.evidence_count:
        raise ValueError("expected_slots length must match evidence rows")
    if len(frame.lineage_ids) != evidence.evidence_count:
        raise ValueError("lineage_ids length must match evidence rows")
    if evidence.target_assignment is not None:
        target_slots = np.argmax(evidence.target_assignment, axis=1).astype(np.int64, copy=False)
        if not np.array_equal(target_slots, expected_slots):
            raise ValueError("target_assignment must encode expected_slots")
    return frame


def _validate_world_object(obj: SyntheticWorldObject) -> SyntheticWorldObject:
    if not isinstance(obj, SyntheticWorldObject):
        raise TypeError("world object must be SyntheticWorldObject")
    if int(obj.oracle_object_id) < 0:
        raise ValueError("oracle_object_id must be >= 0")
    if not obj.lineage_id:
        raise ValueError("lineage_id must be non-empty")
    _float_vector(obj.pose_center, "pose_center", length=3)
    _float_vector(obj.appearance_feature, "appearance_feature")
    rgb = _float_vector(obj.appearance_rgb, "appearance_rgb", length=3)
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError("appearance_rgb must be in [0, 1]")
    _float_vector(obj.trajectory, "trajectory", length=3)
    return obj


def _canonical_centers(object_count: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, object_count, dtype=np.float32)
    centers = np.zeros((object_count, 3), dtype=np.float32)
    centers[:, 0] = x
    return centers


def _canonical_rgb(object_id: int) -> np.ndarray:
    palette = np.asarray(
        [
            [0.88, 0.18, 0.12],
            [0.15, 0.68, 0.88],
            [0.25, 0.78, 0.32],
            [0.84, 0.64, 0.18],
        ],
        dtype=np.float32,
    )
    return palette[int(object_id) % palette.shape[0]].copy()


def _default_scenario_frame_count(scenario_kind: str) -> int:
    if scenario_kind == "occlusion_recovery":
        return 4
    return 3


def _default_observation_config(seed: int) -> ObservationModelConfig:
    return ObservationModelConfig(
        points_per_object=3,
        position_jitter=0.02,
        feature_noise=0.0,
        include_mask_votes=True,
        include_track_hints=True,
        seed=int(seed) + 101,
    )


def _appearance_object_id_for_scenario(
    scenario_kind: str,
    *,
    frame_index: int,
    object_id: int,
    object_count: int,
) -> int:
    if (
        scenario_kind == "adversarial_swap"
        and frame_index > 0
        and object_count >= 2
        and object_id in (0, 1)
    ):
        return 1 - int(object_id)
    return int(object_id)


def _visibility_transition_records(oracle: ObjectIdentityOracle) -> list[dict[str, Any]]:
    oracle = validate_object_identity_oracle(oracle)
    records: list[dict[str, Any]] = []
    for identity in oracle.identities:
        statuses = []
        transitions = []
        previous_visible: bool | None = None
        for frame_index, frame in enumerate(oracle.frames):
            observation = next(
                item
                for item in frame
                if int(item.oracle_object_id) == int(identity.oracle_object_id)
            )
            visible = bool(observation.visible)
            statuses.append(
                {
                    "frame_index": int(frame_index),
                    "visible": visible,
                    "expected_slot": int(observation.expected_slot),
                    "expected_slot_relation": observation.expected_slot_relation,
                }
            )
            if previous_visible is not None:
                transitions.append(
                    {
                        "from_frame": int(frame_index - 1),
                        "to_frame": int(frame_index),
                        "transition": _visibility_transition_name(
                            previous_visible,
                            visible,
                        ),
                    }
                )
            previous_visible = visible
        records.append(
            {
                "oracle_object_id": int(identity.oracle_object_id),
                "lineage_id": identity.lineage_id,
                "expected_slot": int(identity.canonical_slot),
                "statuses": statuses,
                "transitions": transitions,
            }
        )
    return records


def _visibility_transition_name(previous_visible: bool, visible: bool) -> str:
    if previous_visible and visible:
        return "visible_to_visible"
    if previous_visible and not visible:
        return "visible_to_occluded"
    if not previous_visible and visible:
        return "occluded_to_visible"
    return "occluded_to_occluded"


def _object_perturbation(
    scenario_kind: str,
    *,
    frame_index: int,
    object_id: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if scenario_kind == "perturbation" and frame_index > 0:
        return {
            "kind": "appearance_noise",
            "feature_delta": float(0.05 + 0.01 * object_id),
            "brightness": float(1.0 + 0.05 * (object_id + 1)),
        }
    if scenario_kind == "adversarial_swap" and frame_index > 0 and object_id in (0, 1):
        return {
            "kind": "appearance_swap",
            "swapped_with": int(1 - object_id),
        }
    return {"kind": "none"}


def _frame_perturbation(scenario_kind: str, frame_index: int) -> dict[str, Any]:
    if scenario_kind == "cross_view":
        return {"kind": "camera_view_change", "view_index": int(frame_index)}
    if scenario_kind == "occlusion_recovery":
        return {"kind": "occlusion_window" if frame_index > 0 else "none"}
    if scenario_kind == "perturbation":
        return {"kind": "brightness_camera_jitter" if frame_index > 0 else "none"}
    if scenario_kind == "adversarial_swap":
        return {"kind": "object_swap_stress" if frame_index > 0 else "none"}
    return {"kind": "none"}


def _apply_feature_perturbation(feature: np.ndarray, perturbation: dict[str, Any]) -> np.ndarray:
    result = np.asarray(feature, dtype=np.float32).copy()
    delta = float(perturbation.get("feature_delta", perturbation.get("appearance_mutation", 0.0)))
    if delta > 0:
        result = result + delta
        norm = float(np.linalg.norm(result))
        if norm > _EPS:
            result = result / norm
    return result.astype(np.float32, copy=False)


def _apply_rgb_perturbation(rgb: np.ndarray, perturbation: dict[str, Any]) -> np.ndarray:
    brightness = float(perturbation.get("brightness", 1.0))
    return np.clip(np.asarray(rgb, dtype=np.float32) * brightness, 0.0, 1.0).astype(np.float32, copy=False)


def _view_id(scenario_kind: str, frame_index: int) -> str:
    if scenario_kind == "cross_view":
        return f"camera-view-{frame_index}"
    return f"frame-view-{frame_index}"


def _point_offsets(points_per_object: int, *, scale: float) -> tuple[np.ndarray, ...]:
    if points_per_object == 1:
        return (np.zeros(3, dtype=np.float32),)
    angles = np.linspace(0.0, 2.0 * np.pi, points_per_object, endpoint=False, dtype=np.float32)
    return tuple(
        np.asarray([np.cos(angle) * scale, np.sin(angle) * scale, 0.0], dtype=np.float32)
        for angle in angles
    )


def _apply_view_transform(position: np.ndarray, *, view_id: str) -> np.ndarray:
    result = np.asarray(position, dtype=np.float32).copy()
    if view_id.startswith("camera-view-"):
        try:
            view_index = int(view_id.rsplit("-", 1)[1])
        except ValueError:
            view_index = 0
        result[1] += 0.04 * view_index
    return result.astype(np.float32, copy=False)


def _identity_by_object_id(oracle: ObjectIdentityOracle) -> dict[int, ObjectIdentityRecord]:
    return {int(identity.oracle_object_id): identity for identity in oracle.identities}


def _float_vector(value: np.ndarray, label: str, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{label} must have length {length}")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _int_vector(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one value")
    return array.astype(np.int64, copy=False)
