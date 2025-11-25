# tools/optimizer.py — создание оптимизатора с раздельными группами параметров

import torch


# ==============================
# КОНСТАНТЫ ОПТИМИЗАТОРА
# ==============================

# Имена частей модели для группировки параметров
BACKBONE_KEYWORD = 'backbone'
NECK_KEYWORD = 'neck'        # (опционально, если используется)
HEAD_KEYWORD = 'decode_head'

# Ключевые слова для определения нормализационных слоёв
NORM_KEYWORDS = {'norm', 'bn', 'ln'}

# Параметры по умолчанию
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LEARNING_RATE = 6e-5
DEFAULT_HEAD_LR_MULT = 10.0


# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def build_param_groups(model, weight_decay=DEFAULT_WEIGHT_DECAY, lr=DEFAULT_LEARNING_RATE, head_lr_mult=DEFAULT_HEAD_LR_MULT):
    """
    Группирует параметры модели для дифференцированного обучения:
      - backbone: стандартный lr и weight_decay
      - norm-слои: lr без weight_decay (обычная практика)
      - head: увеличенный lr, стандартный weight_decay

    Аргументы:
        model (nn.Module): модель для обучения.
        weight_decay (float): коэффициент L2-регуляризации.
        lr (float): базовая скорость обучения.
        head_lr_mult (float): множитель скорости обучения для головы.

    Возвращает:
        list[dict]: список групп параметров для оптимизатора.
    """
    params = []

    # Списки для разных типов параметров
    norm_params = []      # LayerNorm, BatchNorm и др.
    head_params = []      # decode_head
    backbone_params = []  # всё остальное (включая neck, если есть)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Определение типа параметра по имени
        if HEAD_KEYWORD in name:
            head_params.append(param)
        elif any(kw in name for kw in NORM_KEYWORDS):
            norm_params.append(param)
        else:
            backbone_params.append(param)

    # Группа 1: backbone (включая neck, если он не в head)
    params.append({
        'params': backbone_params,
        'weight_decay': weight_decay,
        'lr': lr
    })

    # Группа 2: нормализационные слои (без weight_decay)
    params.append({
        'params': norm_params,
        'weight_decay': 0.0,
        'lr': lr
    })

    # Группа 3: голова сегментации (увеличенный lr)
    params.append({
        'params': head_params,
        'weight_decay': weight_decay,
        'lr': lr * head_lr_mult
    })

    print("Параметры модели разбиты на группы:")
    print(f"   Backbone (+ Neck): {len(backbone_params)} параметров")
    print(f"   Norm layers:       {len(norm_params)} параметров")
    print(f"   Decode Head:       {len(head_params)} параметров\n")

    return params


# ==============================
# ОСНОВНАЯ ФУНКЦИЯ
# ==============================

def create_optimizer(
    model,
    lr=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    head_lr_mult=DEFAULT_HEAD_LR_MULT
):
    """
    Создаёт AdamW-оптимизатор с дифференцированными группами параметров.

    Аргументы:
        model (nn.Module): модель для оптимизации.
        lr (float): базовая скорость обучения.
        weight_decay (float): коэффициент регуляризации.
        head_lr_mult (float): множитель lr для головы.

    Возвращает:
        torch.optim.AdamW: сконфигурированный оптимизатор.
    """
    param_groups = build_param_groups(
        model,
        weight_decay=weight_decay,
        lr=lr,
        head_lr_mult=head_lr_mult
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,  # базовый lr (в группах он уже переопределён)
        betas=(0.9, 0.999),
        weight_decay=weight_decay  # общий wd, но в группах он может быть 0
    )
    return optimizer