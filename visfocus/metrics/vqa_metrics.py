import logging
from collections import namedtuple
from typing import Optional
import os
from transformers import EvalPrediction

from visfocus.metrics.base_metric import BaseMetric
from visfocus.metrics.scorers.accuracy_scorer import AccuracyScorer
from visfocus.metrics.scorers.anls_scorer import AnlsScorer
from visfocus.metrics.anls_evaluation import Evaluator as ANLSEvaluator

logger = logging.getLogger()


class VQAMetrics(BaseMetric):

    def __call__(self, eval_pred: EvalPrediction) -> dict:
        labels, predictions = self._from_eval_pred_to_strings(eval_pred)
        logging.info(f'Calculated metrics over {len(labels)} samples')
        scorers = [AnlsScorer(), AccuracyScorer()]
        metrics = {}
        logging.info("Evaluation metrics:")
        for scorer in scorers:
            preds = scorer.from_items(predictions)
            annots = scorer.from_items(labels)
            scorer.add(preds, annots)
            metrics[scorer.metric_name()] = scorer.score()
            logging.info(f"{scorer.metric_name()}: {metrics[scorer.metric_name()]}")

        results = []
        ## To match the ANLS our team uses
        ANLSEntry = namedtuple("ANLSEntry", ["text", "gt_text"])
        for label, pred in zip(labels, predictions):
            results.append(ANLSEntry(pred, [label]))
        eval_anls = ANLSEvaluator.get_anls(results)
        logging.info(f'ANLS Score: {eval_anls}')
        metrics['Orig ANLS'] = eval_anls

        return metrics


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
  """Calculates relaxed correctness.

  The correctness tolerates certain error ratio defined by max_relative_change.
  See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
  “Following Methani et al. (2020), we use a relaxed accuracy measure for the
  numeric answers to allow a minor inaccuracy that may result from the automatic
  data extraction process. We consider an answer to be correct if it is within
  5% of the gold answer. For non-numeric answers, we still need an exact match
  to consider an answer to be correct.”

  Args:
    target: Target string.
    prediction: Predicted string.
    max_relative_change: Maximum relative change.

  Returns:
    Whether the prediction was correct given the specified tolerance.
  """

  def _to_float(text: str) -> Optional[float]:
    try:
      if text.endswith("%"):
        # Convert percentages to floats.
        return float(text.rstrip("%")) / 100.0
      else:
        return float(text)
    except ValueError:
      return None

  prediction_float = _to_float(prediction)
  target_float = _to_float(target)
  if prediction_float is not None and target_float:
    relative_change = abs(prediction_float - target_float) / abs(target_float)
    return relative_change <= max_relative_change
  else:
    return prediction.lower() == target.lower()
