# vedadet/ops/dcn/__init__.py

try:
    from .deform_cov import (  # ojo: en el repo el archivo se llama deform_cov.py
        DeformConv, DeformConvPack, deform_conv,
        ModulatedDeformConv, ModulatedDeformConvPack, modulated_deform_conv
    )
except Exception:
    DeformConv = DeformConvPack = None
    ModulatedDeformConv = ModulatedDeformConvPack = None
    deform_conv = modulated_deform_conv = None

try:
    from .deform_pool import (
        DeformRoIPooling, DeformRoIPoolingPack, deform_roi_pooling
    )
except Exception:
    DeformRoIPooling = DeformRoIPoolingPack = None
    deform_roi_pooling = None

__all__ = [
    'DeformConv', 'DeformConvPack', 'deform_conv',
    'ModulatedDeformConv', 'ModulatedDeformConvPack', 'modulated_deform_conv',
    'DeformRoIPooling', 'DeformRoIPoolingPack', 'deform_roi_pooling'
]
