
import re
import json
import typing
import editdistance


def exact_string_match_fn(s1, s2):
    # Return whether two strings are exact matched
    s1 = s1.strip().lower()
    s2 = s2.strip().lower()
    return s1 == s2


def alphanum_string_match_fn(s1, s2):
    # only match letters and numbers
    s1 = s1.strip().lower()
    s1 = ''.join(ch for ch in s1 if ch.isalnum())
    s2 = s2.strip().lower()
    s2 = ''.join(ch for ch in s2 if ch.isalnum())
    return s1 == s2


def string_match_fn_ed_tolerance(s1, s2, ed_tol=1, min_s1_len=8):
    # Return whether two strings are matched with edit distance tolerance
    # the relaxation only kicks in for len(s1) >= min_s1_len
    # when this is called, s1 is always the GT string.
    s1 = s1.strip().lower()
    s2 = s2.strip().lower()
    if len(s1) < min_s1_len:
        return s1 == s2
    else:
        ed_dist = editdistance.eval(s1, s2)
        if ed_dist <= ed_tol:
            return True
        else:
            return False


class Evaluator:
    @staticmethod
    def _decide_tp_tn_fp_fn(answer, answer_pred, matcher, is_empty):
        tp = tn = fp = fn = False
        if is_empty(answer):
            if is_empty(answer_pred):
                # If both gt and pred are empty answers
                # we say it is a true negative
                tn = True
            else:
                # Any non-empty predictions are considered
                # as false positives
                fp = True
        else:
            if matcher(answer, answer_pred):
                tp = True
            else:
                # There are two cases of false negative:
                # 1) we predict something but it does not match gt
                # 2) we predict nothing (empty answer).
                # This will lead to a lot many false negatives,
                # and as a result the recall will be low.
                # This makes it a strict metric.
                fn = True
        return tp, tn, fp, fn

    @staticmethod
    def get_f1_scores(
            answers, answer_preds,
            matcher=exact_string_match_fn,
            is_empty=lambda x: x == ''
    ):
        counts = {
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
        }
        for answer, answer_pred in zip(
                answers, answer_preds
        ):
            tp, tn, fp, fn = Evaluator._decide_tp_tn_fp_fn(answer, answer_pred, matcher, is_empty)
            counts['tp'] += tp
            counts['tn'] += tn
            counts['fp'] += fp
            counts['fn'] += fn

        if counts['tp'] == 0:  # handle undefined scienarios
            if counts['fp'] == 0 and counts['fn'] == 0:
                recall = precision = f1 = 1.0
            else:
                recall = precision = f1 = 0.0
        else:
            recall = counts['tp'] / (counts['tp'] + counts['fn'])
            precision = counts['tp'] / (counts['tp'] + counts['fp'])
            f1 = 2 * recall * precision / (recall + precision)
        return {'f1': f1, 'recall': recall, 'precision': precision, 'counts': counts}

    @staticmethod
    def _get_anls(s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= 0.5 else 0.0
        return anls

    @staticmethod
    def get_anls(pred_list):
        pred_scores = []
        if isinstance(pred_list, typing.Dict):
            pred_list = list(pred_list.values())
        for entry in pred_list:
            anls = max(Evaluator._get_anls(entry.text, gt) for gt in entry.gt_text)
            pred_scores.append(anls)
        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy