import editdistance


class ANLSMetric:

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def __call__(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= self.iou_threshold else 0.0
        return anls
