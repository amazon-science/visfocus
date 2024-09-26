import logging
import string
from typing import List, Dict
from operator import itemgetter

import textdistance
import numpy as np

from visfocus.metrics.scorers.base_scorer import BaseScorer

logger = logging.getLogger(__name__)


class AnlsScorer(BaseScorer):
    """ANSL Scorer."""

    def __init__(self, threshold: float = 0.5, case_sensitive=False):
        self.__scores: List[float] = []
        self.__indices: List[int] = []
        self.threshold = threshold
        self.case_sensitive = case_sensitive

    @property
    def scores(self):
        return self.__scores

    @property
    def indices(self):
        return self.__indices

    def add(self, out_items: dict, ref_items: dict):
        """Add more items for computing corpus level scores.

        Args:
            out_items: outs from a single document (line)
            ref_items: reference of the evaluated document (line)

        """
        out_ann = sorted(out_items['annotations'], key=itemgetter('key'))
        ref_ann = sorted(ref_items['annotations'], key=itemgetter('key'))
        assert [a['key'][:100] for a in out_ann] == [a['key'][:100] for a in ref_ann]

        for out, ref in zip(out_ann, ref_ann):
            assert len(out['values']) == 1
            val = out['values'][0]['value']
            possible_vals = ref['values'][0]['value_variants']

            if not self.case_sensitive:
                val = val.lower()
                possible_vals = list(map(str.lower, possible_vals))

            best_answer_index, best_score = self.match(val, possible_vals)
            self.__indices.append(best_answer_index)
            if 1 - self.threshold > best_score:
                best_score = 0.0
            self.__scores.append(best_score)

    def score(self) -> float:
        if self.__scores:
            return sum(self.__scores) / len(self.__scores)
        return 0.0

    @classmethod
    def support_feature_scores(cls) -> bool:
        return False

    @classmethod
    def metric_name(cls) -> str:
        return "ANLS"

    @classmethod
    def from_items(cls, vals) -> Dict:
        anns = []
        for idx, val in enumerate(vals):
            anns.append({
                'key': str(idx),
                'values': [
                    {'value': val, 'value_variants': val if isinstance(val, list) else [val]}
                ]
            })
        return {
            'name': 'test',
            'annotations': anns
        }
