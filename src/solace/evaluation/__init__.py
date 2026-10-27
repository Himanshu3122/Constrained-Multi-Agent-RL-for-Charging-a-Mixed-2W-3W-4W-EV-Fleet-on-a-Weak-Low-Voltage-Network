"""Evaluation: episode metrics, method registry, aggregation and reports."""

from solace.evaluation.runner import METHODS, aggregate, evaluate_method, run_evaluation, write_report

__all__ = ["METHODS", "aggregate", "evaluate_method", "run_evaluation", "write_report"]
