# tools/metrics.py — метрики для бинарной сегментации с поддержкой нескольких порогов

import torch

# ==============================
# КОНСТАНТЫ МЕТРИК
# ==============================

# Порог по умолчанию для бинаризации предсказаний
DEFAULT_THRESHOLD = 0.5

# Список порогов для анализа
ANALYSIS_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Малое значение для избежания деления на ноль
METRICS_EPS = 1e-6


# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def _compute_metrics_from_counts(tp, fp, fn, tn, eps=METRICS_EPS):
    """Вычисляет словарь метрик из счётчиков."""
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


# ==============================
# ОСНОВНОЙ КЛАСС МЕТРИК
# ==============================

class BinarySegmentationMetrics:
    """
    Класс для вычисления метрик бинарной сегментации.

    Поддерживает:
      - накопление по батчам (для валидации)
      - вычисление при одном пороге (для совместимости)
      - анализ при нескольких порогах (для отладки и выбора порога)
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.all_pred_logits = None
        self.fp = None
        self.tp = None
        self.fn = None
        self.tn = None
        self.all_targets = None
        self.default_threshold = threshold
        self.reset()

    def reset(self):
        """Сбрасывает накопленные значения."""
        self.tp = self.fp = self.fn = self.tn = 0
        self.all_pred_logits = []
        self.all_targets = []

    def update(self, pred_logits, target):
        """
        Обновляет счётчики для default_threshold и сохраняет сырые данные для анализа.
        """
        # Сохраняет сырые логиты и таргеты для последующего анализа при разных порогах
        self.all_pred_logits.append(pred_logits.detach().cpu())
        self.all_targets.append(target.detach().cpu())

        # Обновляет счётчики для порога по умолчанию
        pred = (pred_logits.sigmoid() > self.default_threshold).float()
        target_bin = target.float()
        self.tp += (pred * target_bin).sum().item()
        self.fp += (pred * (1 - target_bin)).sum().item()
        self.fn += ((1 - pred) * target_bin).sum().item()
        self.tn += ((1 - pred) * (1 - target_bin)).sum().item()

    def compute(self):
        """
        Возвращает метрики для порога по умолчанию (для совместимости с существующим кодом).
        """
        return _compute_metrics_from_counts(self.tp, self.fp, self.fn, self.tn)

    def compute_at_thresholds(self, thresholds=None):
        """
        Вычисляет метрики для списка порогов на всех сохранённых данных.

        Аргументы:
            thresholds (list): список порогов от 0 до 1.

        Возвращает:
            dict: {threshold: {metric_name: value, ...}, ...}
        """
        if thresholds is None:
            thresholds = ANALYSIS_THRESHOLDS
        if not self.all_pred_logits:
            return {}

        # Объединяет все батчи
        all_logits = torch.cat(self.all_pred_logits, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)

        results = {}
        for th in thresholds:
            pred = (all_logits.sigmoid() > th).float()
            target_bin = all_targets.float()

            tp = (pred * target_bin).sum().item()
            fp = (pred * (1 - target_bin)).sum().item()
            fn = ((1 - pred) * target_bin).sum().item()
            tn = ((1 - pred) * (1 - target_bin)).sum().item()

            results[th] = _compute_metrics_from_counts(tp, fp, fn, tn)

        return results