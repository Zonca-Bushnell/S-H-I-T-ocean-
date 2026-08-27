import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qg_cylindrical_eigenmodel import (
    QGColumnParams,
    boundary_jumps,
    exterior_amplitude,
    exterior_radial,
    interior_radial,
    piecewise_streamfunction,
    radial_ode_residual,
    radial_wavenumbers,
    wave_packet_streamfunction,
)


class QGCylindricalEigenmodelTests(unittest.TestCase):
    def setUp(self):
        self.params = QGColumnParams(radius=1.0, pv_wavenumber=3.0, stretching=0.25)
        self.kz = 2.0
        self.m = 1

    def test_piecewise_mode_matches_streamfunction_at_boundary(self):
        psi_jump, dr_jump = boundary_jumps(self.m, self.params, self.kz)
        self.assertAlmostEqual(psi_jump, 0.0, places=10)
        self.assertTrue(math.isfinite(dr_jump))

    def test_internal_radial_mode_satisfies_helmholtz_ode(self):
        kappa, _ = radial_wavenumbers(self.params, self.kz)
        r = np.linspace(0.15, 0.9, 300)
        radial = interior_radial(self.m, r, kappa)
        residual = radial_ode_residual(r, radial, m=self.m, coefficient=kappa**2)
        interior = residual[8:-8]
        scale = np.max(np.abs(radial[8:-8]))
        self.assertLess(np.max(np.abs(interior)) / scale, 2e-3)

    def test_external_radial_mode_satisfies_qg_laplace_ode(self):
        kappa, gamma = radial_wavenumbers(self.params, self.kz)
        amplitude = exterior_amplitude(self.m, self.params, kappa, gamma)
        r = np.linspace(1.2, 4.0, 360)
        radial = exterior_radial(self.m, r, gamma, amplitude)
        residual = radial_ode_residual(r, radial, m=self.m, coefficient=-(gamma**2))
        interior = residual[10:-10]
        scale = np.max(np.abs(radial[10:-10]))
        self.assertLess(np.max(np.abs(interior)) / scale, 5e-3)

    def test_helical_mode_has_expected_vertical_phase_period(self):
        x = np.array([0.4, 0.7])
        y = np.array([0.2, -0.1])
        z = np.array([0.3, 0.6])
        period = 2.0 * math.pi / self.kz
        psi0 = piecewise_streamfunction(
            x, y, z, m=self.m, kz=self.kz, params=self.params
        )
        psi1 = piecewise_streamfunction(
            x, y, z + period, m=self.m, kz=self.kz, params=self.params
        )
        np.testing.assert_allclose(psi0, psi1, rtol=1e-11, atol=1e-11)

    def test_wave_packet_accepts_continuous_spectrum_quadrature(self):
        x = np.linspace(-0.5, 0.5, 5)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        psi = wave_packet_streamfunction(
            x,
            y,
            z,
            m=1,
            kz_values=[1.5, 2.0, 2.5],
            weights=[0.25, 1.0, 0.25],
            params=self.params,
        )
        self.assertEqual(psi.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(psi)))


if __name__ == "__main__":
    unittest.main()
