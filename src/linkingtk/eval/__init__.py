"""Standardized evaluation interface for EA, EL, WSD and WSA."""

from linkingtk.eval.evaluator import Evaluator
from linkingtk.eval.ranking import rank_exhaustive
from linkingtk.eval.report import EvaluationReport

__all__ = ["EvaluationReport", "Evaluator", "rank_exhaustive"]
