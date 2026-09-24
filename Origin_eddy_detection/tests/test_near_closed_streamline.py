"""Synthetic acceptance tests for the non-circular near-closed deep-flow kernel."""
from __future__ import annotations

import unittest

import numpy as np

from Origin_eddy_detection.src.eddy_pipeline.detection_hybrid import DetectionParams, _hua_verify_radius


def params() -> DetectionParams:
    return DetectionParams(
        ssh_window_cells=7,
        start_radius_cells=2,
        max_radius_cells=12,
        speed_ratio_max=3.0,
        angle_jump_max_deg=150.0,
        tangent_tolerance_deg=24.0,
        symmetry_tolerance_deg=120.0,
        min_tangent_fraction=0.70,
        min_reversal_fraction=0.70,
        min_finite_fraction=0.95,
        direction_exception_extra=0,
        surface_search_cells=8,
        deep_search_cells=6,
        deep_hua_mode="near_closed_streamline",
        boundary_mode="near_closed_streamline",
    )


def fallback_params() -> DetectionParams:
    return DetectionParams(
        **{
            **params().__dict__,
            "deep_hua_mode": "tangent_then_near_closed_streamline",
            "boundary_mode": "near_closed_streamline",
            "tangent_tolerance_deg": 10.0,
            "min_tangent_fraction": 0.95,
            "streamline_min_winding_turns": 0.50,
            "streamline_closure_tolerance_cells": 2.5,
            "streamline_min_points": 12,
            "min_finite_fraction": 0.90,
        }
    )


def rotating_field(*, ellipse: float = 1.0, size: int = 65) -> tuple[np.ndarray, np.ndarray, float, float]:
    yy, xx = np.mgrid[:size, :size].astype("f8")
    cx = cy = (size - 1) / 2.0
    # Ellipse != 1 produces elliptical, not circular, streamlines.
    u = -(yy - cy) / ellipse
    v = (xx - cx) * ellipse
    return u, v, cx, cy


class NearClosedStreamlineTests(unittest.TestCase):
    def test_circular_and_elliptical_rotation_pass(self) -> None:
        for ellipse in (1.0, 2.0):
            u, v, cx, cy = rotating_field(ellipse=ellipse)
            check = _hua_verify_radius(u, v, cx, cy, params())
            self.assertTrue(check["hua_pass"])
            self.assertGreaterEqual(float(check["streamline_winding_turns"]), 0.75)
            self.assertGreaterEqual(float(check["streamline_points"]), 16.0)
            self.assertTrue(str(check["streamline_boundary_i"]))

    def test_open_shear_fails_as_no_closed_streamline(self) -> None:
        u = np.ones((65, 65), dtype="f8")
        v = np.zeros_like(u)
        check = _hua_verify_radius(u, v, 32.0, 32.0, params())
        self.assertFalse(check["hua_pass"])
        self.assertEqual(check["first_hard_failure"], "no_closed_streamline")

    def test_missing_velocity_crossing_fails_as_invalid_velocity(self) -> None:
        u, v, cx, cy = rotating_field()
        u[:, 29:36] = np.nan
        v[:, 29:36] = np.nan
        check = _hua_verify_radius(u, v, cx, cy, params())
        self.assertFalse(check["hua_pass"])
        self.assertEqual(check["first_hard_failure"], "invalid_velocity")

    def test_fallback_only_runs_after_tangent_rejection(self) -> None:
        u, v, cx, cy = rotating_field(ellipse=3.0)
        check = _hua_verify_radius(u, v, cx, cy, fallback_params())
        self.assertTrue(check["hua_pass"])
        self.assertTrue(check["fallback_attempted"])
        self.assertEqual(check["deep_boundary_branch"], "near_closed_streamline_fallback")

    def test_circular_primary_does_not_invoke_fallback(self) -> None:
        u, v, cx, cy = rotating_field()
        check = _hua_verify_radius(u, v, cx, cy, fallback_params())
        self.assertTrue(check["hua_pass"])
        self.assertFalse(check["fallback_attempted"])
        self.assertEqual(check["deep_boundary_branch"], "tangent_primary")


if __name__ == "__main__":
    unittest.main()
