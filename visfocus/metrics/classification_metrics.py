import logging
from collections import namedtuple

from transformers import EvalPrediction

from visfocus.metrics.base_metric import BaseMetric
from visfocus.metrics.scorers.accuracy_scorer import AccuracyScorer

logger = logging.getLogger()


class ClassificationMetrics(BaseMetric):

    def __call__(self, eval_pred: EvalPrediction) -> dict:
        labels, predictions = self._from_eval_pred_to_strings(eval_pred)
        logging.info(f'Calculated metrics over {len(labels)} samples')
        scorers = [AccuracyScorer()]
        metrics = {}
        logging.info("Evaluation metrics:")
        for scorer in scorers:
            preds = scorer.from_items(predictions)
            annots = scorer.from_items(labels)
            scorer.add(preds, annots)
            metrics[scorer.metric_name()] = scorer.score()
            logging.info(f"{scorer.metric_name()}: {metrics[scorer.metric_name()]}")

        results = []
        return metrics