import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from Origin_eddy_detection.src.eddy_pipeline import detection_hybrid as detection
from Origin_eddy_detection.src.eddy_pipeline import tracking


class MatlabDayVoxelTests(unittest.TestCase):
    def detect(self, day, write_voxels=True):
        lon = np.arange(9, dtype=float)
        lat = np.arange(9, dtype=float) - 50
        velocity = np.ones((1, 2, 9, 9), dtype=np.float32)
        velocity[:, :, 4, 5] = np.nan
        dataset = SimpleNamespace(variables={
            'zos_glor': np.zeros((1, 9, 9)),
            'uo_glor': velocity, 'vo_glor': velocity,
        })
        seed = pd.DataFrame([{
            'seed_i': 4, 'seed_j': 4,
            'ssh_extremum_type': 'max', 'ssh_value_m': 0.1,
        }])
        args = SimpleNamespace(
            candidate_cache_dir=None, max_candidates_per_day=0,
            candidate_selection='tile_topn', max_depth_m=0,
            preload_day_uv=True, subgrid_target_degree=1 / 24,
            subgrid_window_radius_cells=2, subgrid_min_finite_fraction=0.6,
            stop_at_first_failed_layer=True, keep_matlab_bridge_files=False,
            write_object_voxels=write_voxels, write_day_figures=False,
        )
        rows = []
        for layer, passed in enumerate((True, False)):
            rows.append({
                'state_index': 1, 'depth_index': layer,
                'speed_min_i_grid': 4, 'speed_min_j_grid': 4,
                'hua_center_i': 4.2, 'hua_center_j': 4.1,
                'center_lon_refined': 4.2, 'center_lat_refined': -45.9,
                'matlab_batch_size': 1, 'matlab_refine_batch_size': 1,
                'matlab_unique_refine_batch_size': 1,
                'matlab_unique_hua_batch_size': 1,
                '_hua_check': {
                    'hua_pass': passed, 'accepted_radius_cells': 2.0,
                    'circulation_sign': 1.0,
                },
            })
        backend = Mock()
        backend.detect_day_many.return_value = rows
        params = detection.DetectionParams(
            ssh_window_cells=7, start_radius_cells=2, max_radius_cells=8,
            speed_ratio_max=3.0, angle_jump_max_deg=150.0,
            tangent_tolerance_deg=24.0, symmetry_tolerance_deg=120.0,
            min_tangent_fraction=0.55, min_reversal_fraction=0.55,
            min_finite_fraction=0.75, direction_exception_extra=2,
            surface_search_cells=3, deep_search_cells=3,
        )
        with patch.object(detection, '_time_lookup', return_value={day: 0}), \
                patch.object(detection, '_load_cached_extrema', return_value=seed):
            result = detection._detect_day(
                day, dataset, None, lon, lat, np.array([0.0, 10.0]),
                params, args, Path('.'), backend,
            )
        return result

    def test_passed_layer_emits_finite_voxels_and_tracking_links(self):
        day = date(2018, 1, 1)
        centers, _, structures, voxels = self.detect(day)
        self.assertEqual(centers.hua_pass.tolist(), [True, False])
        self.assertEqual(len(structures), 1)
        self.assertEqual(len(voxels), 12)  # Radius-two disk minus one missing point.
        self.assertEqual(set(voxels.depth_index), {0})
        self.assertFalse(((voxels.i == 5) & (voxels.j == 4)).any())
        following = self.detect(day + timedelta(days=1))[3]
        volumes0 = voxels.groupby('hua_object_id').node_key_3d.nunique()
        volumes1 = following.groupby('hua_object_id').node_key_3d.nunique()
        edges = tracking._overlap_edges(
            voxels, following, volumes0, volumes1,
            shift_cells=1, surface_only=False,
        )
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(float(edges.iloc[0]['score']), 1.0)

    def test_disabled_output_preserves_detection(self):
        day = date(2018, 1, 1)
        enabled = self.detect(day)
        disabled = self.detect(day, write_voxels=False)
        for before, after in zip(enabled[:3], disabled[:3]):
            pd.testing.assert_frame_equal(before, after)
        self.assertTrue(disabled[3].empty)
        self.assertEqual(list(disabled[3].columns), detection.OBJECT_VOXEL_COLUMNS)


if __name__ == '__main__':
    unittest.main()
