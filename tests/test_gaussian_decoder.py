from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.gaussian_decoder import (
    OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA,
    OBJECT_STATE_GAUSSIAN_POLICY,
    ObjectStateGaussianDecode,
    decode_gaussian_from_object_state,
)
from objgauss.core.object_state import project_object_states


def test_decode_gaussian_from_object_state_maps_slots_to_gaussian_tokens():
    projection = project_object_states(
        _cloud(),
        np.array(
            [
                [1.0, 0.0],
                [0.75, 0.25],
                [0.0, 1.0],
                [0.25, 0.75],
            ],
            dtype=np.float32,
        ),
        evidence_features=_features(),
    )
    object_colors = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.7],
        ],
        dtype=np.float32,
    )

    decoded = decode_gaussian_from_object_state(
        _positions(),
        projection,
        object_colors,
        default_scale=0.025,
        default_opacity=0.8,
    )
    payload = decoded.as_dict()

    assert isinstance(decoded, ObjectStateGaussianDecode)
    assert payload["schema"] == OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA
    assert payload["gaussian_policy"] == OBJECT_STATE_GAUSSIAN_POLICY
    assert payload["object_count"] == 2
    assert payload["gaussian_count"] == 4
    assert payload["differentiable_fields"] == ["object_colors", "assignment"]
    assert payload["frozen_fields"] == ["means", "quats", "scales", "opacities"]
    assert payload["shapes"]["colors"] == [4, 3]
    np.testing.assert_allclose(decoded.means, _positions(), atol=1e-6)
    np.testing.assert_allclose(decoded.quats[:, 0], np.ones(4, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(decoded.scales, np.full((4, 3), 0.025, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(decoded.opacities, np.full(4, 0.8, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        decoded.colors,
        projection.assignment @ object_colors,
        atol=1e-6,
    )
    assert decoded.object_ids.tolist() == projection.derived_object_ids.tolist()


def test_decode_gaussian_from_object_state_validates_shapes():
    projection = project_object_states(_cloud(), np.full((4, 2), 0.5, dtype=np.float32))

    with pytest.raises(ValueError, match="object_colors rows"):
        decode_gaussian_from_object_state(_positions(), projection, np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="evidence"):
        decode_gaussian_from_object_state(_positions()[:2], projection, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="default_scale"):
        decode_gaussian_from_object_state(_positions(), projection, np.zeros((2, 3), dtype=np.float32), default_scale=0)
    with pytest.raises(ValueError, match="default_opacity"):
        decode_gaussian_from_object_state(
            _positions(),
            projection,
            np.zeros((2, 3), dtype=np.float32),
            default_opacity=1.5,
        )


def _cloud() -> GaussianCloud:
    vertices = np.zeros(
        4,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
        ],
    )
    xyz = _positions()
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["opacity"] = 1.0
    return GaussianCloud(vertices=vertices, source_format="fixture")


def _positions() -> np.ndarray:
    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [-0.8, 0.1, 0.0],
            [0.8, -0.1, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
