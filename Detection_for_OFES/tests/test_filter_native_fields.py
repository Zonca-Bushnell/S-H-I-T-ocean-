from __future__ import annotations

import unittest

import numpy as np

from Detection_for_OFES.filters.gaussian_highpass import nan_gaussian_lowpass
from Detection_for_OFES.io.native_fields import align_native_w_vertical


class FilterNativeFieldTests(unittest.TestCase):
    def test_fft_matches_direct_discrete_gaussian(self) -> None:
        rng = np.random.default_rng(19910101)
        field = rng.normal(size=(31, 64)).astype("f4")
        lon = np.arange(64, dtype="f8") * 0.1
        lat = -15.0 + np.arange(31, dtype="f8") * 0.1
        direct = nan_gaussian_lowpass(field, lon, lat, 120.0, convolution_engine="direct")
        fft = nan_gaussian_lowpass(field, lon, lat, 120.0, convolution_engine="fft")
        np.testing.assert_allclose(fft, direct, rtol=2e-5, atol=2e-6)

    def test_native_w_layer_center_alignment_is_vectorized_and_linear(self) -> None:
        raw = np.asarray([0.0, 10.0, 20.0], dtype="f4")[:, None, None]
        aligned, metadata = align_native_w_vertical(
            raw, np.asarray([0.0, 10.0, 20.0]), np.asarray([5.0, 15.0]), "linear"
        )
        np.testing.assert_allclose(aligned[:, 0, 0], [5.0, 15.0])
        self.assertEqual(metadata["native_w_vertical_alignment"], "layer_center")


if __name__ == "__main__":
    unittest.main()
