#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_skin_images.py
- Pergunta a pasta com imagens
- Mostra quantas imagens existem
- Pergunta o total desejado e gera augmentations ate alcancar
- Salva tudo em <pasta>/imgOutput
- Aumentacoes evitam alterar a area preta (mascara) das imagens
"""

import random, shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_THRESHOLD = 8  # valor minimo (0-255) para considerar o pixel como conteudo real


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


# ================== SUPORTE A MASCARAS ==================

def compute_foreground_mask(img, threshold: int = MASK_THRESHOLD) -> np.ndarray:
    """Retorna mascara binaria (0/255) onde existe informacao real na imagem."""
    if img.ndim == 2:
        base = img
    else:
        base = img[..., :3]
        if base.ndim == 3:
            base = np.max(base, axis=2)
    mask = (base > threshold).astype(np.uint8) * 255
    return mask


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def apply_mask_zero(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    mask_bool = mask.astype(bool)
    out[~mask_bool] = 0
    return out


def warp_image_and_mask(img: np.ndarray, mask: np.ndarray, M: np.ndarray):
    h, w = img.shape[:2]
    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    new_mask = cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    new_mask = normalize_mask(new_mask)
    out = apply_mask_zero(out, new_mask)
    return out, new_mask


# ================== AUGMENTATIONS BASICAS (OpenCV) ==================

def random_rotation(img, mask):
    ang = random.uniform(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return warp_image_and_mask(img, mask, M)


def random_flip(img, mask):
    r = random.random()
    if r < 0.25:
        code = 1
    elif r < 0.50:
        code = 0
    elif r < 0.60:
        code = -1
    else:
        return img, mask
    out = cv2.flip(img, code)
    new_mask = cv2.flip(mask, code)
    new_mask = normalize_mask(new_mask)
    out = apply_mask_zero(out, new_mask)
    return out, new_mask


def random_translate_zoom(img, mask):
    h, w = img.shape[:2]
    max_shift = int(0.06 * min(w, h))
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    scale = random.uniform(0.92, 1.08)

    M = np.float32([[scale, 0, tx], [0, scale, ty]])
    return warp_image_and_mask(img, mask, M)


def random_brightness_contrast(img, mask):
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-15, 15)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    out = apply_mask_zero(out, mask)
    return out, mask


def gaussian_blur_or_sharpen(img, mask):
    r = random.random()
    if r < 0.5:
        k = random.choice([3, 5, 7])
        out = cv2.GaussianBlur(img, (k, k), 0)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(img, -1, kernel)
    out = apply_mask_zero(out, mask)
    return out, mask


def gaussian_noise(img, mask):
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return apply_mask_zero(img, mask), mask
    sigma = random.uniform(5, 12)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32)
    out[mask_bool] = np.clip(out[mask_bool] + noise[mask_bool], 0, 255)
    out = out.astype(np.uint8)
    out = apply_mask_zero(out, mask)
    return out, mask


def salt_and_pepper(img, mask):
    mask_coords = np.column_stack(np.where(mask > 0))
    if mask_coords.size == 0:
        return apply_mask_zero(img, mask), mask

    amount = random.uniform(0.005, 0.02)
    num = int(amount * mask_coords.shape[0])
    if num == 0:
        return apply_mask_zero(img, mask), mask

    num = min(num, mask_coords.shape[0])
    idx = np.random.choice(mask_coords.shape[0], size=num, replace=False)
    chosen = mask_coords[idx]
    np.random.shuffle(chosen)
    half = num // 2
    salt_coords = chosen[:half]
    pepper_coords = chosen[half:]

    out = img.copy()
    for r, c in salt_coords:
        out[r, c] = 255
    for r, c in pepper_coords:
        out[r, c] = 0
    out = apply_mask_zero(out, mask)
    return out, mask


def augment_pipeline(img):
    """Combina transformacoes diferentes por amostra (ordem varia) respeitando a mascara."""
    mask = compute_foreground_mask(img)
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
    cur_mask = mask
    for op in ops[: random.randint(3, 5)]:
        out, cur_mask = op(out, cur_mask)
    out = apply_mask_zero(out, cur_mask)
    return out


# ====================================================================

def main():
    folder = Path(input("Digite o caminho da pasta com as imagens: ").strip('" ').strip())
    if not folder.exists():
        print("Pasta nao encontrada.")
        return

    imgs = list_images(folder)
    n = len(imgs)
    print(f"Encontradas {n} imagens.")

    try:
        desired = int(input("Quantas imagens voce quer ter ao final (originais + aumentadas)? ").strip())
    except ValueError:
        print("Valor invalido.")
        return

    if desired < n:
        print("O total desejado e menor que o numero de originais; nada a fazer.")
        return

    out_dir = folder / "imgOutput"
    ensure_dir(out_dir)

    needed = desired - n
    if needed == 0:
        print("Ja atingiu o total desejado. Concluido.")
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

    print("Concluido! Veja a pasta:", out_dir.resolve())


if __name__ == "__main__":
    random.seed()
    np.random.seed(None)
    main()
