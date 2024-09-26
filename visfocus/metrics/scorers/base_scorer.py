""""
Taken from
https://github.com/due-benchmark/evaluator/blob/master/due_evaluator/scorers/base_scorer.py
"""

import abc
from typing import List, AnyStr, Tuple
import numpy as np
import textdistance


class BaseScorer(abc.ABC):
    """Abstract class for scorers."""

    @abc.abstractmethod
    def add(self, out_items: List[dict], ref_items: List[dict]):
        pass

    @abc.abstractmethod
    def score(self):
        pass

    @abc.abstractclassmethod
    def support_feature_scores(cls) -> bool:
        pass

    @abc.abstractclassmethod
    def metric_name(cls) -> str:
        pass

    @staticmethod
    def match(pred_val: str, annot_vals: List[str]) -> Tuple[int, float]:
        best_scores = [textdistance.levenshtein.normalized_similarity(pred_val, pos) for pos in annot_vals]
        best_answer_index = np.argmax(best_scores)
        return best_answer_index, best_scores[best_answer_index]