import torch


def build_param_groups(model, weight_decay=0.01, lr=6e-5, head_lr_mult=10.0):
    """
    Создаёт группы параметров с разными настройками, как в mmsegmentation.
    """
    params = []

    # Группы параметров
    norm_params = []  # LayerNorm, BatchNorm
    head_params = []  # decode_head
    backbone_params = []  # остальное

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Проверяем, относится ли к head
        if 'decode_head' in name:
            head_params.append(param)
        # Проверяем, является ли нормализацией
        elif 'norm' in name or 'bn' in name or 'ln' in name:
            norm_params.append(param)
        else:
            backbone_params.append(param)

    # Группа 1: backbone (стандартный wd, lr)
    params.append({
        'params': backbone_params,
        'weight_decay': weight_decay,
        'lr': lr
    })

    # Группа 2: нормализации (wd=0, lr=std)
    params.append({
        'params': norm_params,
        'weight_decay': 0.0,
        'lr': lr
    })

    # Группа 3: head (wd=std, lr=увеличенный)
    params.append({
        'params': head_params,
        'weight_decay': weight_decay,
        'lr': lr * head_lr_mult
    })

    print(f"Параметры разбиты на группы:")
    print(f"   Backbone: {len(backbone_params)} параметров")
    print(f"   Norm:     {len(norm_params)} параметров")
    print(f"   Head:     {len(head_params)} параметров\n")

    return params


# Создание оптимизатора
def create_optimizer(model, lr=6e-5, weight_decay=0.01, head_lr_mult=10.0):
    param_groups = build_param_groups(
        model,
        weight_decay=weight_decay,
        lr=lr,
        head_lr_mult=head_lr_mult
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    return optimizer