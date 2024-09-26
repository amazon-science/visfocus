from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from transformers import PreTrainedTokenizerBase, EvalPrediction


class BaseMetric:
    def __init__(self, tokenizer: PreTrainedTokenizerBase = None, **kwargs: Any):
        self.tokenizer = tokenizer

    def _from_eval_pred_to_strings(self, eval_pred: EvalPrediction):
        label_ids = eval_pred.label_ids
        prediction_ids = eval_pred.predictions
        self.assign_pad_value(label_ids)
        self.assign_pad_value(prediction_ids)
        labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        return labels, predictions

    def assign_pad_value(self, x):
        pad_token = self.tokenizer.pad_token_id
        if isinstance(x, np.ndarray):
            pad_token = np.array(pad_token, dtype=x.dtype)
        if isinstance(x, torch.Tensor):
            pad_token = torch.tensor(pad_token, dtype=x.dtype, device=x.device)
        x[x <= -100] = pad_token

    @abstractmethod
    def __call__(self, eval_pred: EvalPrediction) -> dict:
        # labels, predictions = self._from_eval_pred_to_strings(eval_pred)
        pass
