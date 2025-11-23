class BinarySegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred_logits, target):
        pred = (pred_logits.sigmoid() > self.threshold).float()
        target = target.float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()

    def compute(self):
        eps = 1e-6
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        pixel_acc = (tp + self.tn) / (tp + fp + fn + self.tn + eps)
        return {
            'IoU_forgery': iou,
            'F1_forgery': f1,
            'Precision_forgery': precision,
            'Recall_forgery': recall,
            'Pixel_Accuracy': pixel_acc
        }