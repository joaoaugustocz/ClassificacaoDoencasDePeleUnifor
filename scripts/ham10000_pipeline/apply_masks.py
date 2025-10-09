#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_masks.py
- Aplica máscaras de segmentação nas imagens do HAM10000
- Cria imagens com a lesão isolada (fundo neutro) ou recortada
- Opção de correção de vinheta (iluminação não-uniforme)
- Salva em <pasta>/masked_images_*
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png"}
NEUTRAL_BG = (235, 235, 235)  # fundo cinza claro


def list_images(folder: Path):
    """Lista todas as imagens na pasta."""
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def find_mask_for_image(img_path: Path, masks_folder: Path):
    """
    Encontra a máscara correspondente para uma imagem.
    Imagem: ISIC_0024306.jpg -> Máscara: ISIC_0024306_segmentation.png
    """
    img_id = img_path.stem  # ISIC_0024306
    mask_name = f"{img_id}_segmentation.png"
    mask_path = masks_folder / mask_name
    return mask_path if mask_path.exists() else None


def apply_mask_isolated(img_bgr, mask_gray, bg_color=NEUTRAL_BG):
    """
    Aplica a máscara isolando a lesão sobre fundo neutro.

    Args:
        img_bgr: Imagem original (BGR)
        mask_gray: Máscara binária (0=fundo, 255=lesão)
        bg_color: Cor do fundo (BGR tuple)

    Returns:
        Imagem com lesão isolada sobre fundo neutro
    """
    # Normaliza máscara para [0, 1]
    mask_norm = (mask_gray / 255.0).astype(np.float32)
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)

    # Cria fundo neutro
    bg = np.full_like(img_bgr, bg_color, dtype=np.uint8).astype(np.float32)

    # Compõe: lesão onde mask=1, fundo onde mask=0
    result = (img_bgr.astype(np.float32) * mask_3ch + bg * (1.0 - mask_3ch))

    return result.astype(np.uint8)


def apply_mask_cropped(img_bgr, mask_gray, bg_color=NEUTRAL_BG):
    """
    Aplica a máscara e recorta para o bounding box da lesão.

    Args:
        img_bgr: Imagem original (BGR)
        mask_gray: Máscara binária (0=fundo, 255=lesão)
        bg_color: Cor do fundo (BGR tuple)

    Returns:
        Imagem recortada ao redor da lesão
    """
    # Primeiro isola a lesão
    isolated = apply_mask_isolated(img_bgr, mask_gray, bg_color)

    # Encontra bounding box da máscara
    coords = cv2.findNonZero(mask_gray)
    if coords is None:
        # Se máscara vazia, retorna imagem original
        return isolated

    x, y, w, h = cv2.boundingRect(coords)

    # Adiciona pequena margem (5%)
    margin = int(0.05 * max(w, h))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_bgr.shape[1] - x, w + 2 * margin)
    h = min(img_bgr.shape[0] - y, h + 2 * margin)

    # Recorta
    cropped = isolated[y:y+h, x:x+w]

    return cropped


def correct_vignetting(img_bgr, mask_gray):
    """
    Corrige vinheta (iluminação não-uniforme) usando a própria imagem.

    Estratégia:
    1. Estima o mapa de iluminação usando blur gaussiano
    2. Normaliza a imagem dividindo pelo mapa de iluminação
    3. Aplica apenas na região da lesão (dentro da máscara)

    Args:
        img_bgr: Imagem original (BGR)
        mask_gray: Máscara binária da lesão

    Returns:
        Imagem com vinheta corrigida
    """
    # Converte para Lab para trabalhar com luminância
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    # Estima mapa de iluminação com blur muito forte
    # Kernel grande captura o gradiente suave da vinheta
    kernel_size = max(img_bgr.shape[:2]) // 8
    if kernel_size % 2 == 0:
        kernel_size += 1  # deve ser ímpar
    kernel_size = max(kernel_size, 51)  # mínimo 51

    illumination = cv2.GaussianBlur(L, (kernel_size, kernel_size), 0)

    # Normaliza: remove o efeito de iluminação não-uniforme
    # L_corrigido = L * (média_global / iluminação_local)
    L_float = L.astype(np.float32)
    illum_float = illumination.astype(np.float32) + 1e-6  # evita divisão por zero

    mean_illum = np.mean(illumination[mask_gray > 0])  # média apenas na lesão
    L_corrected = L_float * (mean_illum / illum_float)
    L_corrected = np.clip(L_corrected, 0, 255).astype(np.uint8)

    # Reconstrói Lab e volta para BGR
    lab_corrected = cv2.merge([L_corrected, a, b])
    corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)

    # Aplica correção apenas dentro da máscara (preserva fundo original)
    mask_norm = (mask_gray / 255.0).astype(np.float32)
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)

    result = (corrected.astype(np.float32) * mask_3ch +
              img_bgr.astype(np.float32) * (1.0 - mask_3ch))

    return result.astype(np.uint8)


def main():
    # Caminho fixo para o dataset HAM10000
    base_folder = Path("HAM10000")

    if not base_folder.exists():
        print(f"Erro: Pasta '{base_folder}' não encontrada.")
        return

    images_folder = base_folder / "images"
    masks_folder = base_folder / "masks"

    if not images_folder.exists() or not masks_folder.exists():
        print("Erro: Pastas 'images' ou 'masks' não encontradas.")
        return

    # Pergunta o modo de processamento
    print("\n" + "="*60)
    print("MODOS DE PROCESSAMENTO")
    print("="*60)
    print("1. Isolado - Lesão sobre fundo neutro (tamanho original)")
    print("2. Recortado - Lesão recortada ao bounding box")
    print("3. Isolado + Vinheta corrigida")
    print("4. Recortado + Vinheta corrigida")
    print("5. Todos - Gera todas as 4 versões")
    print("="*60)

    mode = input("\nEscolha o modo [1/2/3/4/5]: ").strip()
    if mode not in {"1", "2", "3", "4", "5"}:
        print("Opção inválida.")
        return

    # Lista todas as imagens
    images = list_images(images_folder)
    print(f"\nEncontradas {len(images)} imagens.")

    # Cria diretórios de saída conforme modo escolhido
    if mode in {"1", "5"}:
        out_isolated = base_folder / "masked_images_isolated"
        out_isolated.mkdir(parents=True, exist_ok=True)

    if mode in {"2", "5"}:
        out_cropped = base_folder / "masked_images_cropped"
        out_cropped.mkdir(parents=True, exist_ok=True)

    if mode in {"3", "5"}:
        out_isolated_vignette = base_folder / "masked_images_isolated_vignette_corrected"
        out_isolated_vignette.mkdir(parents=True, exist_ok=True)

    if mode in {"4", "5"}:
        out_cropped_vignette = base_folder / "masked_images_cropped_vignette_corrected"
        out_cropped_vignette.mkdir(parents=True, exist_ok=True)

    # Processa cada imagem
    processed = 0
    skipped = 0

    for img_path in tqdm(images, desc="Processando"):
        # Carrega imagem
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"\nErro ao ler: {img_path.name}")
            skipped += 1
            continue

        # Encontra máscara correspondente
        mask_path = find_mask_for_image(img_path, masks_folder)
        if mask_path is None:
            print(f"\nMáscara não encontrada para: {img_path.name}")
            skipped += 1
            continue

        # Carrega máscara
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"\nErro ao ler máscara: {mask_path.name}")
            skipped += 1
            continue

        # Verifica se dimensões correspondem
        if img.shape[:2] != mask.shape[:2]:
            print(f"\nDimensões não correspondem: {img_path.name}")
            skipped += 1
            continue

        # Binariza máscara (caso não esteja binária)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Processa conforme modo escolhido
        # Modo 1: Apenas isolado
        if mode == "1":
            isolated = apply_mask_isolated(img, mask_binary)
            out_name = f"{img_path.stem}_isolated.png"
            cv2.imwrite(str(out_isolated / out_name), isolated)

        # Modo 2: Apenas recortado
        elif mode == "2":
            cropped = apply_mask_cropped(img, mask_binary)
            out_name = f"{img_path.stem}_cropped.png"
            cv2.imwrite(str(out_cropped / out_name), cropped)

        # Modo 3: Isolado + vinheta corrigida
        elif mode == "3":
            img_corrected = correct_vignetting(img, mask_binary)
            isolated = apply_mask_isolated(img_corrected, mask_binary)
            out_name = f"{img_path.stem}_isolated_vc.png"
            cv2.imwrite(str(out_isolated_vignette / out_name), isolated)

        # Modo 4: Recortado + vinheta corrigida
        elif mode == "4":
            img_corrected = correct_vignetting(img, mask_binary)
            cropped = apply_mask_cropped(img_corrected, mask_binary)
            out_name = f"{img_path.stem}_cropped_vc.png"
            cv2.imwrite(str(out_cropped_vignette / out_name), cropped)

        # Modo 5: Todos
        elif mode == "5":
            # Sem vinheta
            isolated = apply_mask_isolated(img, mask_binary)
            cv2.imwrite(str(out_isolated / f"{img_path.stem}_isolated.png"), isolated)

            cropped = apply_mask_cropped(img, mask_binary)
            cv2.imwrite(str(out_cropped / f"{img_path.stem}_cropped.png"), cropped)

            # Com vinheta corrigida
            img_corrected = correct_vignetting(img, mask_binary)

            isolated_vc = apply_mask_isolated(img_corrected, mask_binary)
            cv2.imwrite(str(out_isolated_vignette / f"{img_path.stem}_isolated_vc.png"), isolated_vc)

            cropped_vc = apply_mask_cropped(img_corrected, mask_binary)
            cv2.imwrite(str(out_cropped_vignette / f"{img_path.stem}_cropped_vc.png"), cropped_vc)

        processed += 1

    print(f"\n{'='*60}")
    print(f"Processamento concluído!")
    print(f"Imagens processadas: {processed}")
    print(f"Imagens ignoradas: {skipped}")

    # Mostra caminhos das pastas geradas
    if mode in {"1", "5"}:
        print(f"\nImagens isoladas: {out_isolated.resolve()}")
    if mode in {"2", "5"}:
        print(f"Imagens recortadas: {out_cropped.resolve()}")
    if mode in {"3", "5"}:
        print(f"Imagens isoladas (vinheta corrigida): {out_isolated_vignette.resolve()}")
    if mode in {"4", "5"}:
        print(f"Imagens recortadas (vinheta corrigida): {out_cropped_vignette.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()