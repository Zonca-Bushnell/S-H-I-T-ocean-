"""Compatibility entry for the canonical prho supervisor."""
from Detection_for_OFES.datasets.supervise_ofes2_monthly_prho import *  # noqa: F401,F403
from Detection_for_OFES.datasets.supervise_ofes2_monthly_prho import main

if __name__ == "__main__":
    main()
