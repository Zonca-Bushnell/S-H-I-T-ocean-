"""Compatibility entry retained while existing prho workers finish."""
from Detection_for_OFES.datasets.download_ofes2_annual_prho import *  # noqa: F401,F403
from Detection_for_OFES.datasets.download_ofes2_annual_prho import main

if __name__ == "__main__":
    main()
