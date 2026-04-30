from __future__ import annotations

from importlib import import_module

__all__ = [
    "LatestPickSummary",
    "build_latest_picks_markdown",
    "build_latest_picks_table",
    "build_research_summary_markdown",
    "build_walk_forward_splits",
    "build_walk_forward_summary_markdown",
    "run_walk_forward_analysis",
    "write_latest_picks_json",
    "write_latest_picks_markdown",
    "write_research_summary",
    "write_research_summary_json",
]

_SYMBOL_TO_MODULE = {
    "LatestPickSummary": "ashare_quant.analysis.latest_picks",
    "build_latest_picks_markdown": "ashare_quant.analysis.latest_picks",
    "build_latest_picks_table": "ashare_quant.analysis.latest_picks",
    "write_latest_picks_json": "ashare_quant.analysis.latest_picks",
    "write_latest_picks_markdown": "ashare_quant.analysis.latest_picks",
    "build_research_summary_markdown": "ashare_quant.analysis.research_report",
    "write_research_summary": "ashare_quant.analysis.research_report",
    "write_research_summary_json": "ashare_quant.analysis.research_report",
    "build_walk_forward_splits": "ashare_quant.analysis.walk_forward",
    "build_walk_forward_summary_markdown": "ashare_quant.analysis.walk_forward",
    "run_walk_forward_analysis": "ashare_quant.analysis.walk_forward",
}


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if not module_name:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
