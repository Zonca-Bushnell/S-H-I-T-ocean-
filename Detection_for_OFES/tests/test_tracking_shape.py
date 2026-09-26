from __future__ import annotations

import unittest

import pandas as pd

from Detection_for_OFES.core.shape import classify_object_days
from Detection_for_OFES.core.tracking import overlap_score, track_object_days


def _row(day: str, object_id: str, depth: int, ci: float, cj: float, polarity: str = "cyclonic") -> dict:
    return {
        "date": day, "hua_object_id": object_id, "depth_index": depth,
        "depth_m": 2.5 + 50 * depth, "polarity": polarity, "hua_pass": True,
        "center_i_refined": ci, "center_j_refined": cj,
        "center_lon": 150 + ci * 0.1, "center_lat": 30 + cj * 0.1,
        "accepted_radius_cells": 3.0, "radius_km": 30.0,
        "ssh_contour_boundary_i": "", "ssh_contour_boundary_j": "",
        "streamline_boundary_i": "", "streamline_boundary_j": "",
    }


class TrackingShapeTests(unittest.TestCase):
    def test_overlap_allows_one_cell_shift(self) -> None:
        left = {(0, 0, 0), (0, 1, 0)}
        right = {(0, 1, 0), (0, 2, 0)}
        self.assertEqual(overlap_score(left, right, 1), 1.0)

    def test_tracking_is_deterministic_and_same_polarity(self) -> None:
        rows = []
        for depth in range(2):
            rows.append(_row("1991-01-01", "a", depth, 10, 10))
            rows.append(_row("1991-01-02", "b", depth, 11, 10))
            rows.append(_row("1991-01-02", "c", depth, 11, 10, "anticyclonic"))
        objects, edges, events = track_object_days(pd.DataFrame(rows))
        self.assertEqual(objects.loc[objects.hua_object_id == "a", "track_id"].iloc[0],
                         objects.loc[objects.hua_object_id == "b", "track_id"].iloc[0])
        self.assertFalse(((edges.source_id == "a") & (edges.target_id == "c")).any())
        self.assertIn("continuous", set(events.event))

    def test_shape_classes_and_unknown(self) -> None:
        rows = [_row("1991-01-01", "upright", depth, 10, 10) for depth in range(6)]
        rows += [_row("1991-01-01", "short", depth, 10 + depth, 10) for depth in range(5)]
        metrics, _ = classify_object_days(pd.DataFrame(rows), upright_fallback=0.12)
        self.assertEqual(metrics.loc[metrics.hua_object_id == "upright", "shape_class"].iloc[0], "upright")
        self.assertEqual(metrics.loc[metrics.hua_object_id == "short", "shape_class"].iloc[0], "unknown")


if __name__ == "__main__":
    unittest.main()
