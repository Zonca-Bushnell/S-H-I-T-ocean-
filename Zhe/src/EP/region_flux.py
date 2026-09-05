"""Region-wise heat/PV/momentum transport interfaces.

The public functions here are intentionally narrow: callers ask for
region-wise aggregate-product moments, while the validated implementation is
kept behind this module seam.
"""

from __future__ import annotations

from .core_shell_runner import (
    _add_fraction_of_total_abs as add_fraction_of_total_abs,
    _add_grouped_abs_fraction as add_grouped_abs_fraction,
    _add_object_region_terms as add_object_region_terms,
    _compute_object_aggregate_transport_partition as compute_object_aggregate_transport_partition,
    _empty_object_region_accumulator as empty_object_region_accumulator,
    _finalize_object_region_accumulator as finalize_object_region_accumulator,
    _region_flux_partition_stats as region_flux_partition_stats,
    _region_internal_stats as region_internal_stats,
)

__all__ = [
    "add_fraction_of_total_abs",
    "add_grouped_abs_fraction",
    "add_object_region_terms",
    "compute_object_aggregate_transport_partition",
    "empty_object_region_accumulator",
    "finalize_object_region_accumulator",
    "region_flux_partition_stats",
    "region_internal_stats",
]
