#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_to_square.py
- Pede pasta com imagens (ideal: PNGs com alpha/círculo)
- Detecta círculo (via alpha ou Hough)
- Recorta o quadrado circunscrito e salva um PNG quadrado
- Fundo neutro (RGB 235) quando há transparência
- Salva em <pasta>/roiSquare
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

NEUTRAL_BG = (235, 235, 235)  # cinza claro, não estoura branco

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def find_circle_from_alpha(img_rgba):
    # máscara = alpha > 0
    alpha = img_rgba[:,:,3]
    _, m = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return int(x), int(y), int(r)

def find_circle_hough(img_bgr):
    # útil caso venha JPG com vinheta escura
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rows/8,
                               param1=120, param2=40,
                               minRadius=int(min(gray.shape)*0.25),
                               maxRadius=int(min(gray.shape)*0.48))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return None

def crop_square(img, cx, cy, r):
    h, w = img.shape[:2]
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    return img[y1:y2, x1:x2].copy()

def composite_on_bg(rgba, bg_color=NEUTRAL_BG):
    """Se houver canal alfa, compõe no fundo neutro mantendo RGBA -> BGR."""
    if rgba.shape[2] == 4:
        bgr = rgba[...,:3]
        a   = rgba[...,3:4].astype(np.float32) / 255.0
        bg  = np.full_like(bgr, bg_color, dtype=np.uint8).astype(np.float32)
        comp = (bgr.astype(np.float32) * a + bg * (1.0 - a)).astype(np.uint8)
        return comp
    return rgba

def main():
    folder = Path(input("Digite o caminho da pasta com as imagens circulares: ").strip('" ').strip())
    if not folder.exists():
        print("Pasta não encontrada.")
        return

    imgs = list_images(folder)
    print(f"Encontradas {len(imgs)} imagens.")
    out_dir = folder / "roiSquare"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Falha ao ler:", p)
            continue

        circle = None
        if img.ndim == 3 and img.shape[2] == 4:
            circle = find_circle_from_alpha(img)
        if circle is None:
            # tenta Hough em BGR
            bgr = img if img.shape[2] == 3 else img[...,:3]
            circle = find_circle_hough(bgr)

        h, w = img.shape[:2]
        if circle is None:
            # fallback: supõe círculo central com 90% do menor lado
            r = int(0.45 * min(h, w))
            cx, cy = w//2, h//2
        else:
            cx, cy, r = circle

        crop = crop_square(img, cx, cy, r)
        # Se tinha alpha, compõe no fundo neutro
        if crop.shape[2] == 4:
            bgr = composite_on_bg(crop)
        else:
            bgr = crop

        # opcional: redimensionar para tamanho padrão (descomente para 512x512)
        # bgr = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA)

        out_name = f"{p.stem}_square.png"
        cv2.imwrite(str(out_dir / out_name), bgr)

    print("Concluído! Veja:", out_dir.resolve())

if __name__ == "__main__":
    main()
