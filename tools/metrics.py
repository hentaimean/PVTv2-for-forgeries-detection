# tools/metrics.py — метрики для бинарной сегментации подделок

# ==============================
# КОНСТАНТЫ МЕТРИК
# ==============================

# Порог по умолчанию для бинаризации предсказаний
DEFAULT_THRESHOLD = 0.5

# Малое значение для избежания деления на ноль
METRICS_EPS = 1e-6


# ==============================
# ОСНОВНОЙ КЛАСС МЕТРИК
# ==============================

class BinarySegmentationMetrics:
    """
    Класс для вычисления метрик бинарной сегментации.

    Поддерживаемые метрики:
      - IoU (Intersection over Union) для класса подделок
      - F1-score
      - Precision и Recall
      - Pixel Accuracy (общая точность по пикселям)
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        """
        Инициализация с порогом бинаризации.

        Аргументы:
            threshold (float): порог для перевода логитов в бинарное предсказание.
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Сбрасывает накопленные значения TP, FP, FN, TN."""
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred_logits, target):
        """
        Обновляет счётчики на основе текущего батча.

        Аргументы:
            pred_logits: [B, 1, H, W] — сырые логиты модели
            target: [B, 1, H, W] — бинарная маска (0/1)
        """
        pred = (pred_logits.sigmoid() > self.threshold).float()
        target = target.float()

        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()

    def compute(self):
        """
        Вычисляет и возвращает словарь метрик.

        Возвращает:
            dict: словарь с ключами:
                - 'IoU_forgery'
                - 'F1_forgery'
                - 'Precision_forgery'
                - 'Recall_forgery'
                - 'Pixel_Accuracy'
        """
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = METRICS_EPS

        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        pixel_acc = (tp + tn) / (tp + fp + fn + tn + eps)

        return {
            'IoU_forgery': iou,
            'F1_forgery': f1,
            'Precision_forgery': precision,
            'Recall_forgery': recall,
            'Pixel_Accuracy': pixel_acc
        }