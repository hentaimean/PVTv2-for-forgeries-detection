from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=1.0, last_epoch=-1, min_lr=0.0):
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            max(
                self.min_lr,
                base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]


def create_scheduler(optimizer, warmup_iters=1500, total_iters=320000, min_lr=0.0, power=1.0):
    """
    Создаёт комбинированный scheduler: Linear warmup + Poly decay.
    """
    from torch.optim.lr_scheduler import SequentialLR, LinearLR

    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        total_iters=warmup_iters
    )

    poly = PolyLRScheduler(
        optimizer,
        max_iters=total_iters - warmup_iters,
        power=power,
        min_lr=min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, poly],
        milestones=[warmup_iters]
    )

    return scheduler