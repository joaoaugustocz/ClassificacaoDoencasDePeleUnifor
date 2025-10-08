#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_constancy_clahe.py
- Pede pasta de entrada
- Pergunta se quer CLAHE no L (Lab) ou V (HSV)
- Aplica Gray-World (balanceamento de branco) + CLAHE no canal escolhido
- Salva em subpasta colorFixed_<MODE>
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def gray_world(img_bgr):
    """Color constancy simples (Gray-World): equaliza médias dos canais B,G,R."""
    img = img_bgr.astype(np.float32)
    mean_b, mean_g, mean_r = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    # evita divisão por zero
    scale_b = mean_gray / (mean_b + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_r = mean_gray / (mean_r + 1e-6)
    img[:,:,0] *= scale_b
    img[:,:,1] *= scale_g
    img[:,:,2] *= scale_r
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_on_L(img_bgr, clip=2.0, tile=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    Lc = clahe.apply(L)
    out = cv2.merge([Lc, a, b])
    return cv2.cvtColor(out, cv2.COLOR_Lab2BGR)

def clahe_on_V(img_bgr, clip=2.0, tile=(8,8)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    Vc = clahe.apply(V)
    out = cv2.merge([H, S, Vc])
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

def main():
    folder = Path(input("Digite o caminho da pasta com as imagens: ").strip('" ').strip())
    if not folder.exists():
        print("Pasta não encontrada.")
        return

    mode = input("Aplicar CLAHE em (L)ab ou (H)SV? [L/H]: ").strip().lower()
    if mode not in {"l", "h"}:
        print("Opção inválida.")
        return

    imgs = list_images(folder)
    print(f"Encontradas {len(imgs)} imagens.")

    out_dir = folder / ("colorFixed_Lab" if mode == "l" else "colorFixed_HSV")
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("Falha ao ler:", p)
            continue

        wb = gray_world(img)  # balanceamento de branco simples
        if mode == "l":
            out = clahe_on_L(wb, clip=2.0, tile=(8,8))
        else:
            out = clahe_on_V(wb, clip=2.0, tile=(8,8))

        cv2.imwrite(str(out_dir / f"{p.stem}_cc.png"), out)

    print("Concluído! Veja:", out_dir.resolve())

if __name__ == "__main__":
    main()
