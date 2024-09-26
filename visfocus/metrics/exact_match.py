import numpy as np
from transformers import EvalPrediction

from visfocus.metrics.base_metric import BaseMetric


class ExactMatch(BaseMetric):

    def __call__(self, eval_pred: EvalPrediction) -> dict:
        labels, predictions = self._from_eval_pred_to_strings(eval_pred)
        return dict(exact_match=np.mean(np.array(labels) == np.array(predictions)).item())