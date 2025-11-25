# model/pvtv2.py — реализация модели PVTv2-B5 с FPN-шейкой и головой для бинарной сегментации

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И МОДУЛИ
# ==============================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Стохастическая глубина (Stochastic Depth): отключает случайные ветви в residual-блоках.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Обёртка для drop_path в виде nn.Module."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(tensor, mean=0., std=1.):
    """
    Инициализация усечённым нормальным распределением (без scipy).
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


# ==============================
# 2. ОСНОВНЫЕ МОДУЛИ PVTv2
# ==============================

class DWConv(nn.Module):
    """Глубинное свёрточное преобразование (Depthwise Convolution)."""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP с глубинной свёрткой (используется в PVTv2)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Мультголовое внимание с уменьшением разрешения (Spatial Reduction Attention, SRA).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} не делится на num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.attn_drop = nn.Dropout(0.)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer-блок PVTv2 с SRA и MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Перекрывающееся патч-вложение (аналог Conv2d с padding).
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ==============================
# 3. BACKBONE: PVTv2-B5
# ==============================

class pvt_v2_b5(nn.Module):
    """
    Реализация backbone PVTv2-B5 (без головы классификации).
    Параметры соответствуют официальной спецификации PVTv2-B5.
    """
    # === Архитектурные параметры PVTv2-B5 (фиксированные) ===
    DEPTHS = [3, 6, 40, 3]
    EMBED_DIMS = [64, 128, 320, 512]
    NUM_HEADS = [1, 2, 5, 8]
    MLP_RATIOS = [4, 4, 4, 4]
    SR_RATIOS = [8, 4, 2, 1]
    DROP_PATH_RATE = 0.1

    def __init__(self, img_size=224, in_chans=3):
        super().__init__()
        # Используем константы класса
        self.depths = self.DEPTHS
        self.embed_dims = self.EMBED_DIMS
        self.num_heads = self.NUM_HEADS
        self.mlp_ratios = self.MLP_RATIOS
        self.sr_ratios = self.SR_RATIOS
        self.drop_path_rate = self.DROP_PATH_RATE

        # Stochastic Depth: распределение вероятностей по глубине
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # Патч-эмбеддинги для каждого уровня
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=self.embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=self.embed_dims[0], embed_dim=self.embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=self.embed_dims[1], embed_dim=self.embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=self.embed_dims[2], embed_dim=self.embed_dims[3]
        )

        # Transformer-блоки для каждого уровня
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=self.embed_dims[0], num_heads=self.num_heads[0], mlp_ratio=self.mlp_ratios[0],
                  qkv_bias=True, drop=0., attn_drop=0., drop_path=dpr[cur + i], sr_ratio=self.sr_ratios[0])
            for i in range(self.depths[0])
        ])
        self.norm1 = nn.LayerNorm(self.embed_dims[0])

        cur += self.depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=self.embed_dims[1], num_heads=self.num_heads[1], mlp_ratio=self.mlp_ratios[1],
                  qkv_bias=True, drop=0., attn_drop=0., drop_path=dpr[cur + i], sr_ratio=self.sr_ratios[1])
            for i in range(self.depths[1])
        ])
        self.norm2 = nn.LayerNorm(self.embed_dims[1])

        cur += self.depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=self.embed_dims[2], num_heads=self.num_heads[2], mlp_ratio=self.mlp_ratios[2],
                  qkv_bias=True, drop=0., attn_drop=0., drop_path=dpr[cur + i], sr_ratio=self.sr_ratios[2])
            for i in range(self.depths[2])
        ])
        self.norm3 = nn.LayerNorm(self.embed_dims[2])

        cur += self.depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=self.embed_dims[3], num_heads=self.num_heads[3], mlp_ratio=self.mlp_ratios[3],
                  qkv_bias=True, drop=0., attn_drop=0., drop_path=dpr[cur + i], sr_ratio=self.sr_ratios[3])
            for i in range(self.depths[3])
        ])
        self.norm4 = nn.LayerNorm(self.embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        Возвращает список feature maps для 4 уровней: [B, C, H/4, W/4], ..., [B, C, H/32, W/32]
        """
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# ==============================
# 4. NECK: FPN (Feature Pyramid Network)
# ==============================

class FPN(nn.Module):
    """
    Feature Pyramid Network для агрегации мультимасштабных признаков.
    """
    def __init__(self, in_channels, out_channels, num_outs):
        super(FPN, self).__init__()
        assert num_outs == len(in_channels)
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_outs):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Принимает список признаков из backbone, возвращает FPN-пирамиду.
        """
        assert len(inputs) == len(self.lateral_convs)

        # Латеральные свёртки
        laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]

        # Top-down путь с интерполяцией
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )

        # Выходные свёртки FPN
        outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        return outs


# ==============================
# 5. HEAD: FPNHead для бинарной сегментации
# ==============================

class FPNHead(nn.Module):
    """
    Голова сегментации на основе FPN с масштабируемыми ветвями.
    """
    def __init__(self, in_channels, in_index, feature_strides, channels, num_classes, out_channels,
                 dropout_ratio=0.1, align_corners=False):
        super(FPNHead, self).__init__()
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.align_corners = align_corners

        # Ветви для каждого уровня FPN
        self.scale_heads = nn.ModuleList()
        for i in range(len(in_channels)):
            head_length = max(
                1, int(torch.log2(torch.tensor(feature_strides[i]))) - 2
            )
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Conv2d(
                        in_channels[i] if k == 0 else channels,
                        channels,
                        kernel_size=3,
                        padding=1
                    )
                )
                scale_head.append(nn.BatchNorm2d(channels))
                scale_head.append(nn.GELU())
                if feature_strides[i] // (2 ** (k + 1)) == 4:
                    break
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, out_channels, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Агрегирует признаки с разных уровней FPN в один выход [B, 1, H/4, W/4].
        """
        x = [inputs[i] for i in self.in_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.scale_heads)):
            interp_size = output.shape[2:]
            head_output = self.scale_heads[i](x[i])
            head_output = F.interpolate(
                head_output,
                size=interp_size,
                mode='bilinear',
                align_corners=self.align_corners
            )
            output += head_output

        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output


# ==============================
# 6. ПОЛНАЯ МОДЕЛЬ ДЛЯ СЕГМЕНТАЦИИ ПОДДЕЛОК
# ==============================

class PVTv2B5ForForgerySegmentation(nn.Module):
    """
    Полная модель для бинарной сегментации подделок:
    PVTv2-B5 backbone + FPN neck + FPNHead.
    """
    def __init__(self, num_classes=2, img_size=512):
        super().__init__()
        assert num_classes == 2, "Поддерживается только бинарная сегментация"

        # Backbone
        self.backbone = pvt_v2_b5(img_size=img_size, in_chans=3)

        # Neck (FPN)
        self.neck = FPN(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            num_outs=4
        )

        # Голова сегментации
        self.decode_head = FPNHead(
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            num_classes=2,
            out_channels=1,
            dropout_ratio=0.1,
            align_corners=False
        )

        self.input_img_size = img_size

    def forward(self, x):
        """
        Вход:  [B, 3, H, W]
        Выход: [B, 1, H, W] — logits (без sigmoid!)
        """
        H, W = x.shape[2], x.shape[3]

        # Backbone → 4 уровня признаков
        feats = self.backbone(x)

        # Neck → FPN-пирамида
        fpn_feats = self.neck(feats)

        # Head → [B, 1, H/4, W/4]
        out = self.decode_head(fpn_feats)

        # Восстановление до исходного размера
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out