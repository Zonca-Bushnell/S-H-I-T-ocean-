"""Compatibility entry for the canonical climatology builder."""
from Detection_for_OFES.datasets.build_ofes2_prho_annual_mss import *  # noqa: F401,F403
from Detection_for_OFES.datasets.build_ofes2_prho_annual_mss import main

if __name__ == "__main__":
    main()
