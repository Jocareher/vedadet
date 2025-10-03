# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch

# Intentar extensión nativa
try:
    from . import nms_ext  # compilada por setup.py
    _HAS_EXT = True
except Exception:
    nms_ext = None
    _HAS_EXT = False

def _as_tensor(dets, device_id=None):
    """Convierte dets (np.ndarray o Tensor) a Tensor y devuelve (tensor, is_numpy)."""
    if isinstance(dets, torch.Tensor):
        return dets, False
    elif isinstance(dets, np.ndarray):
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        return torch.from_numpy(dets).to(device), True
    else:
        raise TypeError(f'dets must be Tensor or numpy array, got {type(dets)}')


def nms(dets, iou_thr, device_id=None):
    """NMS estándar.
    Entrada: dets (N,5) con [x1,y1,x2,y2,score] en Tensor o ndarray.
    Devuelve: (dets_kept_mismo_tipo, indices) manteniendo el tipo de la entrada.
    """
    dets_th, is_numpy = _as_tensor(dets, device_id=device_id)

    if dets_th.numel() == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
        return (dets if is_numpy else dets_th)[inds, :], (inds.cpu().numpy() if is_numpy else inds)

    # separar boxes y scores
    if dets_th.size(-1) == 5:
        boxes = dets_th[:, :4]
        scores = dets_th[:, 4]
    else:
        raise ValueError(f'nms expects dets with shape (N,5), got {tuple(dets_th.shape)}')

    # Si la extensión existe se usa; si no, fallback a torchvision.ops.nms
    if _HAS_EXT:
        inds = nms_ext.nms(dets_th, float(iou_thr))
    else:
        from torchvision.ops import nms as tv_nms
        inds = tv_nms(boxes, scores, float(iou_thr))

    # preservar tipo de salida
    if is_numpy:
        inds_np = inds.detach().cpu().numpy()
        return dets[inds_np, :], inds_np
    else:
        return dets_th[inds, :], inds


def _soft_nms_torch(dets_t: torch.Tensor, iou_thr: float, method: str = 'linear',
                    sigma: float = 0.5, min_score: float = 1e-3):
    """
    Soft-NMS puro PyTorch (CPU/GPU), O(N^2), similar al paper:
      - method: 'linear' | 'gaussian'
      - dets_t: Tensor [N,5] -> x1,y1,x2,y2,score
    Devuelve (new_dets[N_kept,5], inds[N_kept]) ordenados por score desc.
    """
    assert dets_t.dim() == 2 and dets_t.size(1) == 5, "dets must be (N,5)"
    device = dets_t.device
    dtype = dets_t.dtype

    boxes = dets_t[:, :4].clone()
    scores = dets_t[:, 4].clone()

    # Ordenar por score desc
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]
    keep_inds = []

    # IoU helper
    def box_iou_single(box, boxes):
        # box: (4,), boxes: (M,4)
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        union = area1 + area2 - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
        return iou

    i = 0
    while i < boxes.size(0):
        # actual top
        max_box = boxes[i]
        max_score = scores[i]
        if max_score < min_score:
            break
        keep_inds.append(i)

        if i + 1 >= boxes.size(0):
            break

        ious = box_iou_single(max_box, boxes[i+1:])
        if method == 'linear':
            weights = torch.where(ious > iou_thr, 1 - ious, torch.ones_like(ious, device=device, dtype=dtype))
        elif method == 'gaussian':
            weights = torch.exp(-(ious * ious) / sigma)
        else:  # 'original' behavior
            weights = torch.where(ious > iou_thr, torch.zeros_like(ious), torch.ones_like(ious))

        scores[i+1:] = scores[i+1:] * weights

        # filtrar por min_score y reordenar la cola
        remain = scores[i+1:] >= min_score
        if remain.any():
            # mantener y resort por score
            keep_mask = torch.cat([torch.ones((i+1,), dtype=torch.bool, device=device), remain])
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            # resort segment i+1..end
            if i + 1 < boxes.size(0):
                tail_scores = scores[i+1:]
                tail_order = torch.argsort(tail_scores, descending=True)
                boxes[i+1:] = boxes[i+1:][tail_order]
                scores[i+1:] = tail_scores[tail_order]
            i += 1
        else:
            # no queda nada por encima de min_score
            break

    keep_inds_t = torch.tensor(keep_inds, device=device, dtype=torch.long)
    kept = torch.cat([boxes[keep_inds_t], scores[keep_inds_t][:, None]], dim=1)
    return kept, keep_inds_t


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    """Soft-NMS.
    Entrada: dets (N,5) [x1,y1,x2,y2,score]
    Salida: (new_dets, inds) igual que original.
    """
    # convert dets (tensor or numpy array) to tensor on CPU (algoritmo no necesita kernel custom)
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_t = dets.detach()  # en el mismo device, puede ser cuda
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_t = torch.from_numpy(dets)
    else:
        raise TypeError(f'dets must be Tensor or numpy array, got {type(dets)}')

    # Usar extensión si existe; si no, fallback
    if _HAS_EXT:
            results = nms_ext.soft_nms(dets_t.detach().cpu(), float(iou_thr),
                                       1 if method == 'linear' else 2, float(sigma), float(min_score))
            new_dets = results[:, :5]
            inds = results[:, 5].long()
            if is_tensor:
                return new_dets.to(device=dets.device, dtype=dets.dtype), inds.to(device=dets.device, dtype=torch.long)
            else:
                return new_dets.numpy().astype(dets.dtype), inds.cpu().numpy().astype(np.int64)
    else:
        new_dets, inds = _soft_nms_torch(dets_t, float(iou_thr), method=method, sigma=float(sigma), min_score=float(min_score))
        if is_tensor:
            return new_dets.to(device=dets.device, dtype=dets.dtype), inds.to(device=dets.device, dtype=torch.long)
        else:
            return new_dets.detach().cpu().numpy().astype(dets.dtype), inds.detach().cpu().numpy().astype(np.int64)


def batched_nms(bboxes, scores, inds, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    (Sin cambios relevantes: llama a nms_op que ahora puede ser fallback.)
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        max_coordinate = bboxes.max()
        offsets = inds.to(bboxes) * (max_coordinate + 1)
        bboxes_for_nms = bboxes + offsets[:, None]
    nms_type = nms_cfg_.pop('typename', 'nms')
    nms_op = eval(nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], -1), **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    return torch.cat([bboxes, scores[:, None]], -1), keep


def _nms_match_fallback(dets_t: torch.Tensor, thresh: float):
    """Agrupa índices por NMS al estilo 'nms_match'.
    Entrada dets_t: (N,5) [x1,y1,x2,y2,score] (Tensor).
    Devuelve: List[List[int]] grupos, cada grupo ordenado por score.
    """
    assert dets_t.dim() == 2 and dets_t.size(1) == 5
    boxes = dets_t[:, :4]
    scores = dets_t[:, 4]
    # ordenar por score desc
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    idxs = torch.arange(boxes.size(0), device=boxes.device, dtype=torch.long)

    from torchvision.ops import box_iou
    iou_mat = box_iou(boxes, boxes)  # (N,N)

    visited = torch.zeros(boxes.size(0), dtype=torch.bool, device=boxes.device)
    groups = []
    for i in range(boxes.size(0)):
        if visited[i]:
            continue
        # grupo para la caja i: todas j>=i con iou >= thresh que todavía no estén visitadas
        mask = (iou_mat[i] >= thresh) & (~visited)
        members = torch.nonzero(mask, as_tuple=False).squeeze(1)
        # marcar visitados
        visited[members] = True
        # mapear a índices originales (antes del sort)
        orig_members = order[members].tolist()
        # ordenar los miembros por score desc (ya lo están porque 'order' lo garantizó)
        groups.append(orig_members)
    return groups


def nms_match(dets, thresh):
    """Matched dets into different groups by NMS.

    Entrada: dets (N,5) -> [x1,y1,x2,y2,score], Tensor o ndarray
    Salida: List[Tensor | ndarray] (grupos de índices)
    """
    if isinstance(dets, torch.Tensor):
        dets_t = dets.detach()
        is_tensor = True
    elif isinstance(dets, np.ndarray):
        dets_t = torch.from_numpy(dets)
        is_tensor = False
    else:
        raise TypeError(f'dets must be Tensor or numpy array, got {type(dets)}')

    if dets_t.shape[0] == 0:
        matched = []
    else:
        assert dets_t.shape[-1] == 5, f'inputs dets.shape should be (N,5), but get {tuple(dets_t.shape)}'
        if _HAS_EXT:
            matched = nms_ext.nms_match(dets_t.detach().cpu(), float(thresh))
            # `matched` es lista de listas de índices (int) respecto a dets_t.cpu()
            # convertimos a índices del mismo tipo que espera abajo
        else:
            matched = _nms_match_fallback(dets_t, float(thresh))

    if is_tensor:
        # salida como lista de tensores long (como el original)
        return [dets_t.new_tensor(m, dtype=torch.long) for m in matched]
    else:
        # numpy: usa int nativo (np.int está deprecado)
        return [np.array(m, dtype=np.int64) for m in matched]
