"""Compatibility entry for the canonical dataset downloader."""
from Detection_for_OFES.datasets.download_ofes2_monthly_climatology import *  # noqa: F401,F403
from Detection_for_OFES.datasets.download_ofes2_monthly_climatology import main

if __name__ == "__main__":
    main()
