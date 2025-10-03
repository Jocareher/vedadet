# vedadet/ops/__init__.py
# --- añade al principio de vedadet/ops/__init__.py, junto a otros imports
import warnings
import torch.nn as nn

# NMS: imprescindible
from .nms import batched_nms, nms, nms_match, soft_nms

# Sigmoid Focal Loss: opcional (solo si se compila su ext)
try:
    from .sigmoid_focal_loss import SigmoidFocalLoss, sigmoid_focal_loss
    HAS_SFL = True
except Exception:
    HAS_SFL = False
    # stubs mínimos por si alguien lo importa
    SigmoidFocalLoss = None
    sigmoid_focal_loss = None

# DCN: OPCIONAL. No intentes importarlo si no está compilado.
try:
    from .dcn import (
        DeformConv, DeformConvPack, DeformRoIPooling,
        DeformRoIPoolingPack, ModulatedDeformConv,
        ModulatedDeformConvPack, deform_conv, modulated_deform_conv,
        deform_roi_pooling
    )
    HAS_DCN = True
except Exception:
    HAS_DCN = False
    DeformConv = DeformConvPack = DeformRoIPooling = None
    DeformRoIPoolingPack = ModulatedDeformConv = None
    ModulatedDeformConvPack = None
    deform_conv = modulated_deform_conv = None
    deform_roi_pooling = None
    
# --- añade esta función dentro de vedadet/ops/__init__.py
def build_plugin_layer(cfg, in_channels, **kwargs):
    """
    Stub compatible con mmcv.cnn.bricks.plugin.build_plugin_layer.

    Devuelve (layer, plugin_name). Si no hay soporte de plugins en esta build,
    o el cfg es None, devuelve nn.Identity().

    Args:
        cfg (dict|None): e.g., {'type': 'ContextBlock', 'reduction': 16, ...}
        in_channels (int): canales de entrada al bloque
        **kwargs: params adicionales que suelen pasar los backbones (no usados)

    Returns:
        tuple: (nn.Module, str)
    """
    if cfg is None:
        return nn.Identity(), ''

    # Si llegamos aquí, el modelo/config pidió un plugin específico que
    # no tenemos implementado. Para no romper la inferencia, devolvemos
    # Identity y avisamos una sola vez.
    warnings.warn(
        f"[vedadet.ops] build_plugin_layer STUB: se pidió plugin "
        f"{cfg.get('type', cfg)} pero no está disponible. "
        "Se usará nn.Identity() en su lugar."
    )
    name = cfg.get('name', cfg.get('type', 'plugin'))
    return nn.Identity(), name


__all__ = [
    'nms', 'soft_nms', 'batched_nms', 'nms_match',
    'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'DeformConv', 'DeformConvPack', 'DeformRoIPooling',
    'DeformRoIPoolingPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'deform_conv',
    'modulated_deform_conv', 'deform_roi_pooling',
    'HAS_DCN', 'HAS_SFL', 'build_plugin_layer',
]
