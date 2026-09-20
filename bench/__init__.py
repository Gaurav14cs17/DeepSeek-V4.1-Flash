"""Shared BEFORE → AFTER → DELTA benchmarking for deepseek_flash_lab.

MOST IMPORTANT RULE:
  BEFORE → MEASURE → IMPLEMENT → AFTER → COMPARE → %Δ → EXPLAIN → SAVE

Never claim an optimization without quantitative evidence against a recorded baseline.
"""

from .accuracy import numerical_diff, quality_tradeoff_table
from .delta import LOWER_IS_BETTER, HIGHER_IS_BETTER, NEUTRAL, compare_metrics, format_table, verdict
from .history import append_history, HISTORY_PATH
from .inference import (
    GEN_LENGTHS,
    PROMPT_LENGTHS,
    allowed_configs,
    format_inference_grid,
    inference_delta_summary,
    measure_inference_point,
)
from .kv import (
    CONTEXT_LENGTHS,
    format_kv_ascii_chart,
    format_kv_sweep_table,
    kv_bytes_per_token,
    measure_kv_point,
)
from .ladder import LEVEL_NAMES, format_ladder_table, ladder_level_label
from .latency import format_component_table, time_components, time_ms
from .memory import MemoryReport, collect_memory, format_memory_report, memory_delta
from .protocol import ExperimentMeta, hardware_info, software_info, save_result_bundle
from .report import write_optimization_report
from .training_metrics import (
    format_training_table,
    measure_training_step,
    normalize_training_metrics,
    training_delta_summary,
)

# CLI modules (bench.regression / bench.compare) are not imported here so
# `python -m bench.regression` stays clean. Use:
#   from bench.regression import check_regression

__all__ = [
    "LOWER_IS_BETTER",
    "HIGHER_IS_BETTER",
    "NEUTRAL",
    "compare_metrics",
    "format_table",
    "verdict",
    "append_history",
    "HISTORY_PATH",
    "MemoryReport",
    "collect_memory",
    "format_memory_report",
    "memory_delta",
    "ExperimentMeta",
    "hardware_info",
    "software_info",
    "save_result_bundle",
    "write_optimization_report",
    "numerical_diff",
    "quality_tradeoff_table",
    "PROMPT_LENGTHS",
    "GEN_LENGTHS",
    "allowed_configs",
    "format_inference_grid",
    "inference_delta_summary",
    "measure_inference_point",
    "CONTEXT_LENGTHS",
    "format_kv_ascii_chart",
    "format_kv_sweep_table",
    "kv_bytes_per_token",
    "measure_kv_point",
    "LEVEL_NAMES",
    "format_ladder_table",
    "ladder_level_label",
    "format_component_table",
    "time_components",
    "time_ms",
    "format_training_table",
    "measure_training_step",
    "normalize_training_metrics",
    "training_delta_summary",
]
