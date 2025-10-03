# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Intentar extensión nativa (si en algún entorno se compila)
try:
    from .sigmoid_focal_loss_ext import sigmoid_focal_loss as _ext_sigmoid_focal
    _HAS_EXT = True
except Exception:
    _ext_sigmoid_focal = None
    _HAS_EXT = False

# 2) Intentar torchvision (recomendado en PyTorch/torchvision recientes)
try:
    from torchvision.ops import sigmoid_focal_loss as _tv_sigmoid_focal
    _HAS_TV = True
except Exception:
    _tv_sigmoid_focal = None
    _HAS_TV = False


def _to_one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convierte etiquetas enteras [N] o [N, ...] a one-hot con num_classes.
    Si targets ya tiene shape == logits.shape, se devuelve tal cual.
    """
    # si ya está en formato one-hot / multi-label (mismo shape que logits)
    if targets.dim() > 1:
        return targets

    # targets: (N,) o (N,*) de índices largos
    if targets.dtype not in (torch.long, torch.int64):
        raise TypeError(
            f"targets dtype must be Long when converting to one-hot, got {targets.dtype}"
        )
    flat = targets.view(-1)
    oh = torch.zeros((flat.numel(), num_classes), device=targets.device, dtype=torch.float32)
    oh.scatter_(1, flat.unsqueeze(1), 1.0)
    return oh.view(*targets.shape, num_classes)


def _pure_torch_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Implementación pura en PyTorch de la Sigmoid Focal Loss (tipo RetinaNet).
    - inputs: logits (N, C, ...)  (NO aplicar sigmoid antes)
    - targets: o bien one-hot con misma shape que inputs, o bien índices enteros
    """
    if targets.shape != inputs.shape:
        # convertir índices → one-hot
        num_classes = inputs.size(1)
        targets = _to_one_hot(targets, num_classes)
        # reacomodar a shape de inputs, asumiendo inputs [N,C,...] y targets [N,...,C] si vino con dims extra
        if targets.dim() != inputs.dim():
            # mover la última dim (C) a dim=1
            # targets: (..., C) -> (N, C, ...)
            perm = [0, targets.dim() - 1] + list(range(1, targets.dim() - 1))
            targets = targets.permute(*perm)
        targets = targets.to(dtype=inputs.dtype, device=inputs.device)
    else:
        targets = targets.to(dtype=inputs.dtype, device=inputs.device)

    # focal
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * torch.pow(1.0 - p_t, gamma) * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # "none"


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "none",
):
    """
    Interfaz pública compatible con la versión original/extensión:
      - logits: (N, C, ...)  (logits sin sigmoid)
      - targets: mismo shape (one-hot) o índices enteros
      - gamma, alpha, reduction in {"none", "mean", "sum"}
    """
    # 1) Ext nativa (si estuviera disponible en otro entorno)
    if _HAS_EXT:
        # OJO: la ext original asumía logits [N,C] y targets one-hot [N,C] (o índices),
        # aquí homogeneizamos para que funcione con shapes >2D también.
        # Para mantener compatibilidad, reducimos cualquier shape a 2D [N*, C] temporalmente.
        orig_shape = logits.shape
        C = logits.size(1)
        logits_flat = logits.permute(0, *range(2, logits.dim()), 1).reshape(-1, C)
        if targets.shape != logits.shape:
            targets_flat = _to_one_hot(
                targets.permute(0, *range(2, targets.dim()), 1).reshape(-1), C
            )
        else:
            targets_flat = targets.permute(0, *range(2, targets.dim()), 1).reshape(-1, C).to(logits_flat.dtype)

        out = _ext_sigmoid_focal(logits_flat, targets_flat, C, gamma, alpha)
        # La ext devuelve pérdidas por elemento (sum/none) según su diseño;
        # por simplicidad, devolvemos el mismo valor (ya reducido) y dejamos reduction a cargo del llamador.
        return out

    # 2) Torchvision si existe (recomendado en 2.x)
    if _HAS_TV:
        # torchvision espera:
        #   - inputs: logits
        #   - targets: one-hot o índices (mismo soporte que arriba)
        # Soportemos both:
        if targets.shape != logits.shape:
            num_classes = logits.size(1)
            targets_oh = _to_one_hot(targets, num_classes)
            if targets_oh.dim() != logits.dim():
                perm = [0, targets_oh.dim() - 1] + list(range(1, targets_oh.dim() - 1))
                targets_oh = targets_oh.permute(*perm)
            targets_oh = targets_oh.to(dtype=logits.dtype, device=logits.device)
            return _tv_sigmoid_focal(logits, targets_oh, alpha=alpha, gamma=gamma, reduction=reduction)
        else:
            return _tv_sigmoid_focal(logits, targets.to(logits.dtype), alpha=alpha, gamma=gamma, reduction=reduction)

    # 3) Fallback puro PyTorch
    return _pure_torch_sigmoid_focal_loss(logits, targets, gamma=gamma, alpha=alpha, reduction=reduction)


# API de módulo compatible con la original
class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "sum"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # La implementación original hacía assert logits.is_cuda;
        # aquí lo removemos para permitir CPU/GPU indistinto.
        loss = sigmoid_focal_loss(
            logits, targets, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction
        )
        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(gamma={self.gamma}, alpha={self.alpha}, reduction='{self.reduction}')"
