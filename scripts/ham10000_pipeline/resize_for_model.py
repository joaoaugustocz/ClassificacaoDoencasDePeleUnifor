#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resize_for_model.py
- Redimensiona imagens para tamanhos padrão de modelos de classificação
- Processa imagens de HAM10000/masked_images_cropped_vignette_corrected
- Oferece opções: 224x224 (ResNet/EfficientNet) ou 299x299 (Inception/Xception)
- Salva em HAM10000/model_ready_<SIZE>/
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path):
    """Lista todas as imagens na pasta."""
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def resize_keep_aspect_ratio(img, target_size, bg_color=(235, 235, 235)):
    """
    Redimensiona mantendo aspect ratio e adiciona padding se necessário.

    Args:
        img: Imagem original (BGR)
        target_size: Tupla (altura, largura) do tamanho final
        bg_color: Cor do padding (BGR tuple)

    Returns:
        Imagem redimensionada com padding
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Calcula scale para caber dentro do target mantendo aspect ratio
    scale = min(target_w / w, target_h / h)

    # Novo tamanho mantendo proporção
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Redimensiona
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Cria canvas do tamanho final com fundo neutro
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # Centraliza a imagem redimensionada no canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def resize_stretch(img, target_size):
    """
    Redimensiona esticando a imagem (pode distorcer).

    Args:
        img: Imagem original (BGR)
        target_size: Tupla (altura, largura) do tamanho final

    Returns:
        Imagem redimensionada
    """
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def main():
    # Caminho fixo para o dataset processado
    base_folder = Path("HAM10000")
    input_folder = base_folder / "masked_images_cropped_vignette_corrected"

    if not input_folder.exists():
        print(f"Erro: Pasta '{input_folder}' não encontrada.")
        print("Execute primeiro apply_masks.py com modo 4 ou 5.")
        return

    # Menu de tamanhos
    print("\n" + "="*60)
    print("TAMANHOS PADRÃO PARA MODELOS")
    print("="*60)
    print("1. 224×224 - ResNet, EfficientNet, VGG, DenseNet")
    print("2. 299×299 - Inception v3, Xception")
    print("="*60)

    size_choice = input("\nEscolha o tamanho [1/2]: ").strip()
    if size_choice not in {"1", "2"}:
        print("Opção inválida.")
        return

    target_size = (224, 224) if size_choice == "1" else (299, 299)
    size_str = f"{target_size[0]}x{target_size[1]}"

    # Menu de método de redimensionamento
    print("\n" + "="*60)
    print("MÉTODO DE REDIMENSIONAMENTO")
    print("="*60)
    print("1. Manter proporção + padding (recomendado)")
    print("   - Não distorce a lesão")
    print("   - Adiciona barras cinzas se necessário")
    print()
    print("2. Esticar para preencher (stretch)")
    print("   - Pode distorcer a lesão")
    print("   - Preenche toda a imagem")
    print("="*60)

    method_choice = input("\nEscolha o método [1/2]: ").strip()
    if method_choice not in {"1", "2"}:
        print("Opção inválida.")
        return

    keep_aspect = (method_choice == "1")
    method_str = "padded" if keep_aspect else "stretched"

    # Lista imagens
    images = list_images(input_folder)
    print(f"\nEncontradas {len(images)} imagens.")

    # Cria diretório de saída
    output_folder = base_folder / f"model_ready_{size_str}_{method_str}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Processa cada imagem
    processed = 0
    skipped = 0

    print(f"\nRedimensionando para {size_str} ({method_str})...\n")

    for img_path in tqdm(images, desc="Processando"):
        # Carrega imagem
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"\nErro ao ler: {img_path.name}")
            skipped += 1
            continue

        # Redimensiona conforme método escolhido
        if keep_aspect:
            resized = resize_keep_aspect_ratio(img, target_size)
        else:
            resized = resize_stretch(img, target_size)

        # Mantém o nome original (sem sufixo adicional)
        out_name = img_path.name
        cv2.imwrite(str(output_folder / out_name), resized)

        processed += 1

    print(f"\n{'='*60}")
    print(f"Processamento concluído!")
    print(f"Imagens processadas: {processed}")
    print(f"Imagens ignoradas: {skipped}")
    print(f"\nTamanho: {size_str}")
    print(f"Método: {'Aspect ratio preservado com padding' if keep_aspect else 'Stretched'}")
    print(f"\nPasta de saída: {output_folder.resolve()}")
    print(f"{'='*60}")

    # Estatísticas de exemplo
    if processed > 0:
        sample_img = cv2.imread(str(output_folder / images[0].name))
        print(f"\nVerificação: Todas as imagens agora têm shape {sample_img.shape}")


if __name__ == "__main__":
    main()