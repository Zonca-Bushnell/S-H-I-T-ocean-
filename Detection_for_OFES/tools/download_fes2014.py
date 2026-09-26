"""Compatibility entry for the canonical dataset downloader."""
from Detection_for_OFES.datasets.download_fes2014 import *  # noqa: F401,F403
from Detection_for_OFES.datasets.download_fes2014 import main

if __name__ == "__main__":
    main()
