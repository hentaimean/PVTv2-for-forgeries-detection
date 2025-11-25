# tools/scheduler.py — комбинированный планировщик обучения: Linear Warmup + Poly Decay

from torch.optim.lr_scheduler import _LRScheduler, LinearLR, SequentialLR

# ==============================
# КОНСТАНТЫ ПЛАНИРОВЩИКА
# ==============================

# Параметры по умолчанию для create_scheduler
DEFAULT_WARMUP_ITERS = 1500
DEFAULT_TOTAL_ITERS = 320_000
DEFAULT_MIN_LR = 0.0
DEFAULT_POWER = 1.0
DEFAULT_WARMUP_START_FACTOR = 1e-6


# ==============================
# КАСТОМНЫЙ ПОЛИНОМИАЛЬНЫЙ ПЛАНИРОВЩИК
# ==============================

class PolyLRScheduler(_LRScheduler):
    """
    Полиномиальный decay (Poly LR):
        lr = base_lr * (1 - iter / max_iters) ** power

    Используется после warmup-фазы.
    """

    def __init__(self, optimizer, max_iters, power=DEFAULT_POWER, last_epoch=-1, min_lr=DEFAULT_MIN_LR):
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Вычисляет текущую скорость обучения для каждой группы параметров."""
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            max(
                self.min_lr,
                base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]


# ==============================
# ФАБРИКА ПЛАНИРОВЩИКА
# ==============================

def create_scheduler(
        optimizer,
        warmup_iters=DEFAULT_WARMUP_ITERS,
        total_iters=DEFAULT_TOTAL_ITERS,
        min_lr=DEFAULT_MIN_LR,
        power=DEFAULT_POWER
):
    """
    Создаёт комбинированный планировщик:
        1. Linear Warmup: от очень малого lr до базового за `warmup_iters` итераций.
        2. Poly Decay: полиномиальное уменьшение lr до `min_lr` за оставшиеся итерации.

    Аргументы:
        optimizer: оптимизатор PyTorch.
        warmup_iters (int): число итераций для warmup.
        total_iters (int): общее число итераций обучения.
        min_lr (float): минимальная скорость обучения (не ниже этого значения).
        power (float): степень полинома в decay (1.0 = линейный decay).

    Возвращает:
        torch.optim.lr_scheduler.SequentialLR: комбинированный планировщик.
    """
    # Warmup: линейное увеличение LR от start_factor * base_lr до base_lr
    warmup = LinearLR(
        optimizer,
        start_factor=DEFAULT_WARMUP_START_FACTOR,
        total_iters=warmup_iters
    )

    # Основной decay: полиномиальное уменьшение
    poly = PolyLRScheduler(
        optimizer,
        max_iters=total_iters - warmup_iters,  # decay применяется после warmup
        power=power,
        min_lr=min_lr
    )

    # Объединение в последовательный планировщик
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, poly],
        milestones=[warmup_iters]  # переход от warmup к decay
    )

    return scheduler