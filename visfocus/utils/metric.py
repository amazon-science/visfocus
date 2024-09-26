class ANLSEvaluator:
    def __init__(self):
        import editdistance  # install with `pip install editdistance`
        self.get_edit_distance = editdistance.eval

    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        if len(s1) ==0 and len(s2) ==0:
            return 1.0
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= 0.5 else 0.0
        return anls

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for (index, entry) in pred_list.items():
            anls = max(self.get_anls(entry.text, gt) for gt in entry.gt_text )
            pred_scores.append(anls)
        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy
