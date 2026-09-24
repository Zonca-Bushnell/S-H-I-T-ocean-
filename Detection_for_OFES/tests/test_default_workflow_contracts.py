from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Detection_for_OFES.profiles import geometry_vertical_profile
from Detection_for_OFES.workflows.context import RunContext
from Detection_for_OFES.workflows.contracts import fingerprint, is_current, status_path, write_status
from Detection_for_OFES.workflows.default_pipeline import parse_stages


class DefaultWorkflowContractTests(unittest.TestCase):
    def test_profile_is_eta_highpass_geometry_only(self) -> None:
        profile = geometry_vertical_profile()
        self.assertEqual(profile.name, "eta_hp500_geometry_vertical_v1")
        self.assertEqual(profile.ssh_definition, "ofes_eta_free_surface")
        self.assertEqual(profile.highpass_cutoff_km, 500.0)
        self.assertEqual(profile.kernel, "gaussian")
        self.assertTrue(profile.remove_streamline_gate)
        self.assertEqual(profile.temporal_window_days, 1)
        self.assertEqual(profile.deep_hua_mode, "tangent_then_near_closed_streamline")
        self.assertFalse(profile.write_object_voxels)
        self.assertEqual((profile.surface_workers, profile.velocity_filter_workers, profile.vertical_workers, profile.composite_workers), (8, 2, 2, 8))
        profile_text = repr(profile).lower()
        for prohibited in ("pressur", "mss", "rossby", "persistence", "tracking", "rebuild"):
            self.assertNotIn(prohibited, profile_text)

    def test_context_uses_only_canonical_run_layout(self) -> None:
        context = RunContext(geometry_vertical_profile(), "1991-01-01", "1991-01-19")
        root = str(context.root).replace("/", "\\")
        self.assertIn("Fromthebeginning\\runs\\eta_hp500_geometry_vertical_v1\\19910101_19910119", root)
        self.assertEqual(context.vertical.name, "tangent45_fraction35_then_near_closed_relaxed")
        self.assertEqual(context.native_w.name, "nh_cyclonic_strict_core")

    def test_date_contract_requires_matching_fingerprint_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output.csv"
            output.write_text("ok", encoding="utf-8")
            contract = {"stage": "geometry-qc", "date": "1991-01-01"}
            contract["fingerprint"] = fingerprint(contract)
            status = status_path(root, "geometry-qc", "19910101")
            write_status(status, contract, [output])
            self.assertTrue(is_current(status, contract, [output]))
            changed = {**contract, "fingerprint": fingerprint({"stage": "geometry-qc", "date": "1991-01-02"})}
            self.assertFalse(is_current(status, changed, [output]))
            output.unlink()
            self.assertFalse(is_current(status, contract, [output]))

    def test_stage_selection_is_explicit(self) -> None:
        self.assertEqual(parse_stages("all")[0], "surface-inputs")
        self.assertEqual(parse_stages("vertical,section-bipolar"), ("vertical", "section-bipolar"))
        with self.assertRaises(ValueError):
            parse_stages("persistence")


if __name__ == "__main__":
    unittest.main()
