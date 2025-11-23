import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, avg_non_ignore=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] — logits
        target: [B, 1, H, W] — 0/1
        """
        loss = self.bce(pred, target)

        if self.avg_non_ignore:
            # Усредняем только по валидным пикселям (все — валидны, но логика сохранена)
            loss = loss.mean()
        else:
            loss = loss.sum() / (target.numel() + 1e-8)

        return self.loss_weight * loss


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_sigmoid=True, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] — logits
        target: [B, 1, H, W] — 0/1
        """
        if self.use_sigmoid:
            pred = pred.sigmoid()

        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(1)
        denominator = pred.sum(1) + target.sum(1)
        dice = (2. * intersection + self.eps) / (denominator + self.eps)
        loss = 1 - dice.mean()

        return self.loss_weight * loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, loss_weight=1.0, use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid

    def forward(self, pred, target):
        if self.use_sigmoid:
            p = pred.sigmoid()
        else:
            p = pred

        # Стабильный подсчёт focal loss без log
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return self.loss_weight * loss.mean()