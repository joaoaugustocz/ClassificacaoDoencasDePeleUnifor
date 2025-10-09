#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_skin_images.py
- Pergunta a pasta com imagens
- Mostra quantas imagens existem
- Pergunta o total desejado e gera augmentations até alcançar
- Salva tudo em <pasta>/imgOutput
"""

import os, random, math, shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

# ================== AUGMENTATIONS BÁSICAS (OpenCV) ==================

def random_rotation(img):
    # rotação pequena (clínico: não distorcer muito)
    ang = random.uniform(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def random_flip(img):
    # -1: flip H+V, 0: V, 1: H, ou não flip
    r = random.random()
    if r < 0.25:
        return cv2.flip(img, 1)
    elif r < 0.50:
        return cv2.flip(img, 0)
    elif r < 0.60:
        return cv2.flip(img, -1)
    return img

def random_translate_zoom(img):
    # translação e zoom leves; mantém tamanho final
    h, w = img.shape[:2]
    max_shift = int(0.06 * min(w, h))
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    scale = random.uniform(0.92, 1.08)

    M = np.float32([[scale, 0, tx], [0, scale, ty]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def random_brightness_contrast(img):
    # alpha = contraste; beta = brilho
    alpha = random.uniform(0.9, 1.1)
    beta  = random.uniform(-15, 15)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def gaussian_blur_or_sharpen(img):
    r = random.random()
    if r < 0.5:
        k = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        # leve sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5.2, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

def gaussian_noise(img):
    # ruído leve (evitar exagero em pele)
    sigma = random.uniform(5, 12)  # desvio-padrão
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def salt_and_pepper(img):
    # quantidade limitada: 0.5% a 2%
    amount = random.uniform(0.005, 0.02)
    out = img.copy()
    num = int(amount * img.shape[0] * img.shape[1])
    # pontos “sal”
    coords = (np.random.randint(0, img.shape[0], num), np.random.randint(0, img.shape[1], num))
    out[coords] = 255
    # pontos “pimenta”
    coords = (np.random.randint(0, img.shape[0], num), np.random.randint(0, img.shape[1], num))
    out[coords] = 0
    return out

def augment_pipeline(img):
    """Combina transformações diferentes por amostra (ordem varia)."""
    ops = [
        random_rotation,
        random_flip,
        random_translate_zoom,
        random_brightness_contrast,
        gaussian_blur_or_sharpen,
        gaussian_noise,
        salt_and_pepper,
    ]
    random.shuffle(ops)

    out = img.copy()
    # aplica de 3 a 5 ops aleatórias por imagem
    for op in ops[:random.randint(3,5)]:
        out = op(out)
    return out

# ====================================================================

def main():
    folder = Path(input("Digite o caminho da pasta com as imagens: ").strip('" ').strip())
    if not folder.exists():
        print("Pasta não encontrada.")
        return

    imgs = list_images(folder)
    n = len(imgs)
    print(f"Encontradas {n} imagens.")

    try:
        desired = int(input("Quantas imagens você quer ter ao final (originais + aumentadas)? ").strip())
    except ValueError:
        print("Valor inválido.")
        return

    if desired < n:
        print("O total desejado é menor que o número de originais; nada a fazer.")
        return

    out_dir = folder / "imgOutput"
    ensure_dir(out_dir)

    # # 1) copia originais pro output (mantém dataset “espelhado”)
    # print("Copiando originais para imgOutput/ ...")
    # for p in tqdm(imgs):
    #     shutil.copy2(p, out_dir / p.name)

    needed = desired - n
    if needed == 0:
        print("Já atingiu o total desejado. Concluído.")
        return

    print(f"Gerando {needed} imagens de augmentation...")
    i = 0
    while i < needed:
        src = random.choice(imgs)
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Falha ao ler:", src)
            continue

        aug = augment_pipeline(img)
        name = f"aug_{i:05d}_{src.stem}.png"
        cv2.imwrite(str(out_dir / name), aug)
        i += 1

    print("Concluído! Veja a pasta:", out_dir.resolve())

if __name__ == "__main__":
    random.seed()
    np.random.seed(None)
    main()
