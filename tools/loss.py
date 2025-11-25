# tools/loss.py — функции потерь для бинарной сегментации подделок

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# КОНСТАНТЫ ПОТЕРЬ
# ==============================

# Параметры по умолчанию для BinaryCrossEntropyLoss
DEFAULT_BCE_POS_WEIGHT = 50.0
DEFAULT_BCE_LOSS_WEIGHT = 1.0

# Параметры по умолчанию для DiceLoss
DEFAULT_DICE_LOSS_WEIGHT = 1.0
DEFAULT_DICE_EPS = 1e-6
DEFAULT_DICE_USE_SIGMOID = True

# Параметры по умолчанию для FocalLoss
DEFAULT_FOCAL_ALPHA = 1.0
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_FOCAL_LOSS_WEIGHT = 1.0
DEFAULT_FOCAL_USE_SIGMOID = True


# ==============================
# ФУНКЦИИ ПОТЕРЬ
# ==============================

class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross-Entropy Loss с поддержкой веса положительного класса.

    Используется с логитами (до sigmoid).
    """

    def __init__(self, pos_weight=DEFAULT_BCE_POS_WEIGHT, loss_weight=DEFAULT_BCE_LOSS_WEIGHT):
        super().__init__()
        self.pos_weight = pos_weight
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        Аргументы:
            pred: [B, 1, H, W] — логиты
            target: [B, 1, H, W] — бинарная маска (0/1)
        """
        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor(self.pos_weight, device=pred.device),
            reduction='mean'
        )
        return self.loss_weight * loss


class DiceLoss(nn.Module):
    """
    Dice Loss для бинарной сегментации.

    Поддерживает работу с логитами (через sigmoid) или с вероятностями.
    """

    def __init__(
            self,
            loss_weight=DEFAULT_DICE_LOSS_WEIGHT,
            use_sigmoid=DEFAULT_DICE_USE_SIGMOID,
            eps=DEFAULT_DICE_EPS
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, pred, target):
        """
        Аргументы:
            pred: [B, 1, H, W] — логиты или вероятности
            target: [B, 1, H, W] — бинарная маска (0/1)
        """
        if self.use_sigmoid:
            pred = pred.sigmoid()

        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(1)
        denominator = pred.sum(1) + target.sum(1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        loss = 1.0 - dice.mean()

        return self.loss_weight * loss


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов.

    Поддерживает работу с логитами (через sigmoid) или с вероятностями.
    """

    def __init__(
            self,
            alpha=DEFAULT_FOCAL_ALPHA,
            gamma=DEFAULT_FOCAL_GAMMA,
            loss_weight=DEFAULT_FOCAL_LOSS_WEIGHT,
            use_sigmoid=DEFAULT_FOCAL_USE_SIGMOID
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid

    def forward(self, pred, target):
        """
        Аргументы:
            pred: [B, 1, H, W] — логиты или вероятности
            target: [B, 1, H, W] — бинарная маска (0/1)
        """
        if self.use_sigmoid:
            p = pred.sigmoid()
        else:
            p = pred

        # Стабильный подсчёт focal loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return self.loss_weight * loss.mean()