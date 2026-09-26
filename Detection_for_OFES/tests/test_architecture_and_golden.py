from __future__ import annotations

import csv
import inspect
import unittest
from pathlib import Path

from Detection_for_OFES.core import detection
from Detection_for_OFES.filters import gaussian_highpass
from Detection_for_OFES.workflows import default_pipeline


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical"
    r"\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19"
)
JAN1_QC = HISTORICAL_ROOT / "surface_geometry_qc/daily_runs/19910101/centers_hua_style.csv"
JAN1_STRICT = (
    HISTORICAL_ROOT
    / "section_bipolar_catalogs_tangent_then_near_closed_relaxed/19910101"
    / "section_bipolar_strict_core_layer_catalog.csv"
)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_formal_package_has_no_origin_imports(self) -> None:
        offenders: list[str] = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if {"legacy", "tools", "tests"}.intersection(path.parts):
                continue
            if "Origin_eddy_detection" in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(path.relative_to(PACKAGE_ROOT)))
        self.assertEqual(offenders, [])

    def test_canonical_filter_has_no_retired_filter_branches(self) -> None:
        source = inspect.getsource(gaussian_highpass).lower()
        for retired in ("rossby", "lanczos", "bessel", "bandpass", "annual_mss"):
            self.assertNotIn(retired, source)

    def test_detection_has_no_matlab_bridge(self) -> None:
        self.assertNotIn("matlab", inspect.getsource(detection).lower())

    def test_default_workflow_does_not_import_tools_or_legacy(self) -> None:
        source = inspect.getsource(default_pipeline)
        self.assertNotIn("Detection_for_OFES.tools", source)
        self.assertNotIn("Detection_for_OFES.legacy", source)


@unittest.skipUnless(JAN1_QC.exists() and JAN1_STRICT.exists(), "local Jan1 historical baseline is unavailable")
class Jan1GoldenBaselineTests(unittest.TestCase):
    def test_geometry_qc_and_strict_core_counts(self) -> None:
        with JAN1_QC.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            qc_count = sum(str(row.get("qc_pass", "")).strip().lower() in {"1", "true"} for row in rows)
        with JAN1_STRICT.open(newline="", encoding="utf-8") as handle:
            strict_ids = {row["hua_object_id"] for row in csv.DictReader(handle)}
        self.assertEqual(qc_count, 1163)
        self.assertEqual(len(strict_ids), 157)


if __name__ == "__main__":
    unittest.main()
