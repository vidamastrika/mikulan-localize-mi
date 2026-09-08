"""
Modern tools for reproducing and extending Localize-MI analyses.
"""

from .inverse import (
    InverseResult,
    SUPPORTED_METHODS,
    normalize_method_name,
    run_inverse,
)
from .io import (
    LocalizeMIRun,
    load_run,
)
from .metrics import (
    LocalizationMetrics,
    StimulationInfo,
    calculate_localization_metrics,
    load_stimulation_info,
    load_surface_transform,
)

__all__ = [
    "InverseResult",
    "LocalizeMIRun",
    "LocalizationMetrics",
    "StimulationInfo",
    "SUPPORTED_METHODS",
    "calculate_localization_metrics",
    "load_run",
    "load_stimulation_info",
    "load_surface_transform",
    "normalize_method_name",
    "run_inverse",
]