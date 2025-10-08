#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import glob

import cv2
import numpy as np
import torch

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia por directorio (vedadet/TinaFace)")
    parser.add_argument("--config", help="ruta al archivo de config")
    parser.add_argument("--source", help="directorio con imágenes")
    parser.add_argument(
        "-o", "--output_dir", default="predictions_tinaface",
        help="directorio de salida (se crean images/ y labels/)"
    )
    parser.add_argument(
        "--score", type=float, default=0.05,
        help="umbral mínimo de score para guardar detecciones"
    )
    return parser.parse_args()


def prepare(cfg):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    engine = build_engine(cfg.infer_engine)
    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)
    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def list_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(str(folder / ext)))
    files = [Path(p) for p in files]
    files.sort()
    return files


def run_pipeline_on_image(pipeline, img_path, device):
    """Construye el batch como en el script original (collate + scatter/unwrap)."""
    data = dict(img_info=dict(filename=str(img_path)), img_prefix=None)
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if device != "cpu":
        data = scatter(data, [device])[0]
        img = data["img"]
        img_metas = data["img_metas"]
    else:
        img = data["img"][0].data
        img_metas = data["img_metas"][0].data
    return img, img_metas


def flatten_and_filter(result, score_thr=0.05):
    """
    result es el mismo tipo que usa plot_result en tu script original:
    lista por clase de arrays Nx5: [x1,y1,x2,y2,score].

    Devuelve arreglo (M,5) concatenado, filtrado por score.
    """
    parts = []
    for cls_dets in result:
        if cls_dets is None or len(cls_dets) == 0:
            continue
        arr = np.asarray(cls_dets)
        if arr.ndim == 2 and arr.shape[1] >= 5:
            parts.append(arr[:, :5])  # x1,y1,x2,y2,score
    if not parts:
        return np.empty((0, 5), dtype=np.float32)

    dets = np.vstack(parts)
    # filtra por score
    keep = dets[:, 4] >= float(score_thr)
    dets = dets[keep]
    # ordena por score desc
    if dets.size > 0:
        dets = dets[np.argsort(-dets[:, 4])]
    return dets


def draw_and_save(imgfp, dets_xyxy_score, outfp, color=(255, 0, 255), thickness=1, font_scale=0.5):
    img = imread(imgfp)
    bgr = color_val(color)
    txt = color_val(color)

    for x1, y1, x2, y2, s in dets_xyxy_score:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, bgr, thickness)
        # cv2.putText(
        #     img, f"{s:.2f}", (p1[0], max(0, p1[1] - 2)),
        #     cv2.FONT_HERSHEY_COMPLEX, font_scale, txt, 1, cv2.LINE_AA
        # )
    imwrite(img, outfp)


def save_txt(txt_path, dets_xyxy_score):
    """
    Guarda en formato: score x1 y1 x2 y2 (por línea).
    Coordenadas como enteros.
    """
    with open(txt_path, "w") as f:
        for x1, y1, x2, y2, s in dets_xyxy_score:
            f.write(f"{s:.6f} {int(round(x1))} {int(round(y1))} {int(round(x2))} {int(round(y2))}\n")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    engine, data_pipeline, device = prepare(cfg)

    src = Path(args.source)
    out_root = Path(args.output_dir)
    out_imgs = out_root / "images"
    out_lbls = out_root / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    img_paths = list_images(src)
    print(f"[INFO] Imágenes encontradas: {len(img_paths)}")
    print(f"[INFO] Guardando en: {out_root}")

    for ip in img_paths:
        try:
            # prepara batch
            img, img_metas = run_pipeline_on_image(data_pipeline, ip, device)
            print(f"[INFO] Procesando {ip.name}, img.shape={img.shape}")

            # inferencia (como en tu script original)
            result = engine.infer(img, img_metas)[0]  # lista por clase

            # a (N,5) con score-thr
            dets = flatten_and_filter(result, score_thr=args.score)

            # guarda txt
            txt_path = out_lbls / f"{ip.stem}.txt"
            save_txt(txt_path, dets)
            print(f"[INFO]   Detecciones guardadas: {len(dets)} en {txt_path.name}")

            # guarda imagen con bboxes
            out_img_path = out_imgs / ip.name
            draw_and_save(str(ip), dets, str(out_img_path), color=(255, 0, 255), thickness=1, font_scale=0.5)
            print(f"[INFO]   Imagen guardada en {out_img_path.name}")

        except Exception as e:
            print(f"[WARN] Falló {ip.name}: {e}")

    print("[DONE] Inferencia finalizada.")


if __name__ == "__main__":
    main()
