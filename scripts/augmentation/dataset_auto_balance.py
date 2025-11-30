#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_auto_balance.py
Script interativo para:
1) Ler um dataset no formato atual (train/val/test -> classes -> subclasses opcionais)
2) Perguntar metas finais por classe ou por subclasse
3) Aplicar remocoes aleatorias + augmentations baseadas em augment_skin_images.py
4) Reorganizar o resultado em uma nova pasta com a mesma estrutura (train/val/test)
"""

from __future__ import annotations

import math
import random
import re
import shutil
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from augment_skin_images import IMG_EXTS, augment_pipeline
except ImportError as exc:  # pragma: no cover - ajuda se o script for chamado de fora do repo
    raise ImportError(
        "Nao foi possivel importar augment_skin_images. Execute este script a partir da raiz do projeto."
    ) from exc

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_MIN_AUG_DOWNSAMPLE = 0.15  # percentual minimo de amostras aumentadas ao reduzir uma classe


@dataclass
class SampleSpec:
    src: Path
    augmented: bool
    subclass_name: Optional[str]


@dataclass
class Bucket:
    class_name: str
    subclass_name: Optional[str]
    files: List[Path]
    subgroups: Dict[Optional[str], List[Path]]
    target_total: int = 0

    def label(self) -> str:
        return self.class_name if not self.subclass_name else f"{self.class_name}/{self.subclass_name}"


def prompt_path(prompt: str) -> Path:
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            print("Informe um caminho valido.")
            continue
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print("Caminho nao encontrado, tente novamente.")
            continue
        return path


def prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        resp = input(f"{message} {suffix} ").strip().lower()
        if not resp:
            return default
        if resp in {"s", "sim", "y", "yes"}:
            return True
        if resp in {"n", "nao", "nao", "no"}:
            return False
        print("Resposta invalida, digite s ou n.")


def prompt_int(message: str, default: Optional[int] = None, min_value: Optional[int] = None) -> int:
    while True:
        base = f"{message}"
        if default is not None:
            base += f" [{default}]"
        resp = input(base + ": ").strip()
        if not resp and default is not None:
            value = default
        else:
            if not resp:
                print("Digite um numero inteiro.")
                continue
            if not resp.isdigit():
                print("Digite um numero inteiro valido.")
                continue
            value = int(resp)
        if min_value is not None and value < min_value:
            print(f"O valor minimo permitido e {min_value}.")
            continue
        return value


def prompt_float(message: str, default: float, min_value: float, max_value: float) -> float:
    while True:
        resp = input(f"{message} [{default:.2f}]: ").strip().replace(",", ".")
        if not resp:
            return default
        try:
            value = float(resp)
        except ValueError:
            print("Digite um numero valido.")
            continue
        if not (min_value <= value <= max_value):
            print(f"Informe um valor entre {min_value} e {max_value}.")
            continue
        return value


def detect_split_dirs(dataset_root: Path) -> Dict[str, Path]:
    split_dirs: Dict[str, Path] = {}
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        key = child.name.lower()
        if key in SPLIT_NAMES:
            split_dirs[key] = child
    missing = [s for s in SPLIT_NAMES if s not in split_dirs]
    if missing:
        raise FileNotFoundError(
            f"Pasta do dataset deve conter subpastas {SPLIT_NAMES}. Faltando: {', '.join(missing)}"
        )
    return split_dirs


def list_image_files(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def list_image_files_recursive(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def scan_dataset(dataset_root: Path, split_dirs: Dict[str, Path]) -> Dict[str, Dict[str, List[Path]]]:
    class_data: Dict[str, Dict[str, List[Path]]] = {}
    for split_name, split_dir in split_dirs.items():
        for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            info = class_data.setdefault(class_dir.name, {"root": [], "subclasses": defaultdict(list)})

            root_files = list_image_files(class_dir)
            if root_files:
                info["root"].extend(root_files)

            for sub_dir in sorted([d for d in class_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                files = list_image_files_recursive(sub_dir)
                if files:
                    info["subclasses"][sub_dir.name].extend(files)
    return class_data


def summarize_classes(class_data: Dict[str, Dict[str, List[Path]]]) -> None:
    print("\nResumo encontrado:")
    for class_name in sorted(class_data.keys()):
        info = class_data[class_name]
        total = len(info["root"]) + sum(len(lst) for lst in info["subclasses"].values())
        subs = sorted(info["subclasses"].keys())
        subs_txt = ", ".join(subs) if subs else "sem subclasses"
        print(f"- {class_name}: {total} imagens ({subs_txt})")


def build_subgroups(info: Dict[str, List[Path]]) -> Dict[Optional[str], List[Path]]:
    subgroups: Dict[Optional[str], List[Path]] = {}
    if info["subclasses"]:
        for sub_name, files in info["subclasses"].items():
            subgroups[sub_name] = files.copy()
        if info["root"]:
            subgroups[None] = info["root"].copy()
    else:
        combined = info["root"].copy()
        subgroups[None] = combined
    return subgroups


def choose_buckets(class_data: Dict[str, Dict[str, List[Path]]]) -> List[Bucket]:
    buckets: List[Bucket] = []
    for class_name in sorted(class_data.keys()):
        info = class_data[class_name]
        has_subs = bool(info["subclasses"])
        files_all = info["root"].copy()
        for lst in info["subclasses"].values():
            files_all.extend(lst)
        if not files_all:
            continue

        use_subs = False
        if has_subs:
            subs_list = ", ".join(sorted(info["subclasses"].keys()))
            use_subs = prompt_yes_no(
                f"Deseja definir metas por subclasse dentro de {class_name}? ({subs_list})", default=False
            )

        if use_subs and has_subs:
            for sub_name, sub_files in sorted(info["subclasses"].items()):
                if not sub_files:
                    continue
                buckets.append(
                    Bucket(
                        class_name=class_name,
                        subclass_name=sub_name,
                        files=sub_files.copy(),
                        subgroups={sub_name: sub_files.copy()},
                    )
                )
        else:
            subgroups = build_subgroups(info)
            buckets.append(
                Bucket(
                    class_name=class_name,
                    subclass_name=None,
                    files=files_all,
                    subgroups=subgroups,
                )
            )
    return buckets


def prompt_targets(buckets: List[Bucket]) -> None:
    print("\nDefina as quantidades finais desejadas:")
    for bucket in buckets:
        current = len(bucket.files)
        bucket.target_total = prompt_int(
            f"Total para {bucket.label()} (atual {current})", default=current, min_value=0
        )


def parse_split_ratios() -> Dict[str, float]:
    raw = input(
        "\nInforme a proporcao train/val/test (ex: 70/15/15 ou 0.7,0.15,0.15). "
        "Pressione ENTER para usar 70/15/15: "
    ).strip()
    if not raw:
        ratios = list(DEFAULT_SPLIT_RATIOS)
    else:
        tokens = re.split(r"[,\s/;]+", raw)
        tokens = [t for t in tokens if t]
        if len(tokens) != 3:
            print("Entrada invalida, usando 70/15/15.")
            ratios = list(DEFAULT_SPLIT_RATIOS)
        else:
            try:
                values = [float(t.replace(",", ".")) for t in tokens]
            except ValueError:
                print("Entrada invalida, usando 70/15/15.")
                ratios = list(DEFAULT_SPLIT_RATIOS)
            else:
                total = sum(values)
                if total <= 0:
                    print("Soma zero, usando 70/15/15.")
                    ratios = list(DEFAULT_SPLIT_RATIOS)
                else:
                    ratios = [v / total for v in values]
    return dict(zip(SPLIT_NAMES, ratios))


def prompt_output_dir(dataset_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default = dataset_root.parent / f"{dataset_root.name}_balanced_{timestamp}"
    raw = input(f"\nPasta de saida (ENTER para {default}): ").strip().strip('"').strip("'")
    out_dir = Path(raw).expanduser().resolve() if raw else default
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise FileExistsError(f"A pasta de saida {out_dir} ja existe e nao esta vazia.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def distribute_equally(total: int, keys: Sequence[Optional[str]]) -> Dict[Optional[str], int]:
    if not keys:
        return {}
    base = total // len(keys)
    remainder = total - base * len(keys)
    result: Dict[Optional[str], int] = {}
    for idx, key in enumerate(keys):
        extra = 1 if idx < remainder else 0
        result[key] = base + extra
    return result


def allocate_split_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {split: total * ratios.get(split, 0.0) for split in SPLIT_NAMES}
    floored = {split: int(math.floor(val)) for split, val in raw.items()}
    remainder = total - sum(floored.values())
    if remainder > 0:
        ordering = sorted(
            SPLIT_NAMES,
            key=lambda split: (raw[split] - floored[split], ratios.get(split, 0.0)),
            reverse=True,
        )
        for split in ordering:
            if remainder == 0:
                break
            floored[split] += 1
            remainder -= 1
    return floored


def plan_group_samples(
    files: List[Path],
    target: int,
    subclass_name: Optional[str],
    min_aug_when_down: float,
) -> List[SampleSpec]:
    if target == 0:
        return []
    if not files:
        raise ValueError("Nao ha imagens suficientes para gerar este grupo.")

    available = len(files)
    entries: List[SampleSpec] = []
    if available > target:  # reducao
        aug_count = int(round(target * min_aug_when_down))
        if min_aug_when_down > 0 and target > 0 and aug_count == 0:
            aug_count = 1
        aug_count = min(aug_count, target)
        originals_needed = target - aug_count
        originals = random.sample(files, originals_needed) if originals_needed > 0 else []
        aug_sources = random.choices(files, k=aug_count) if aug_count > 0 else []
    elif available == target:
        originals = files.copy()
        aug_sources = []
    else:  # aumento
        originals = files.copy()
        aug_needed = target - available
        aug_sources = random.choices(files, k=aug_needed)

    entries.extend(SampleSpec(src=path, augmented=False, subclass_name=subclass_name) for path in originals)
    entries.extend(SampleSpec(src=path, augmented=True, subclass_name=subclass_name) for path in aug_sources)
    return entries


def plan_bucket(bucket: Bucket, min_aug_when_down: float) -> List[SampleSpec]:
    if bucket.target_total == 0:
        return []
    if bucket.subclass_name is None and len(bucket.subgroups) > 1:
        sub_order = sorted(bucket.subgroups.keys(), key=lambda x: "" if x is None else str(x))
        sub_targets = distribute_equally(bucket.target_total, sub_order)
        entries: List[SampleSpec] = []
        for sub_name in sub_order:
            entries.extend(
                plan_group_samples(bucket.subgroups[sub_name], sub_targets[sub_name], sub_name, min_aug_when_down)
            )
        return entries
    else:
        key = bucket.subclass_name if bucket.subclass_name is not None else next(iter(bucket.subgroups.keys()))
        files = bucket.subgroups[key]
        return plan_group_samples(files, bucket.target_total, key, min_aug_when_down)


def build_filename(src: Path, augmented: bool) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", src.stem) or "img"
    token = uuid.uuid4().hex[:8]
    prefix = "aug" if augmented else "orig"
    ext = src.suffix.lower() if src.suffix else ".png"
    return f"{stem}_{prefix}_{token}{ext}"


def write_dataset(
    output_root: Path,
    bucket_plans: List[Tuple[Bucket, Dict[str, List[SampleSpec]]]],
) -> None:
    total = sum(len(samples) for _, split_map in bucket_plans for samples in split_map.values())
    progress = tqdm(total=total, desc="Gerando dataset balanceado", unit="img")
    for bucket, split_map in bucket_plans:
        for split, samples in split_map.items():
            if not samples:
                continue
            for spec in samples:
                dest_dir = output_root / split / bucket.class_name
                if spec.subclass_name:
                    dest_dir /= spec.subclass_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / build_filename(spec.src, spec.augmented)
                if spec.augmented:
                    img = cv2.imread(str(spec.src), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise RuntimeError(f"Falha ao ler imagem para augmentation: {spec.src}")
                    aug_img = augment_pipeline(img)
                    if aug_img is None:
                        raise RuntimeError(f"Augmentation retornou None para {spec.src}")
                    if not cv2.imwrite(str(dest_file), aug_img):
                        raise RuntimeError(f"Falha ao salvar {dest_file}")
                else:
                    shutil.copy2(spec.src, dest_file)
                progress.update(1)
    progress.close()


def main() -> None:
    print("=== Balanceador automatico de datasets ===\n")
    dataset_root = prompt_path("Digite o caminho da pasta do dataset: ")
    split_dirs = detect_split_dirs(dataset_root)
    class_data = scan_dataset(dataset_root, split_dirs)
    if not class_data:
        print("Nenhuma classe encontrada. Verifique o caminho informado.")
        sys.exit(1)

    summarize_classes(class_data)
    buckets = choose_buckets(class_data)
    if not buckets:
        print("Nenhum bucket selecionado. Encerrando.")
        sys.exit(0)

    prompt_targets(buckets)
    ratios = parse_split_ratios()
    min_aug_when_down = prompt_float(
        "Percentual minimo de imagens aumentadas quando for necessario reduzir uma classe",
        default=DEFAULT_MIN_AUG_DOWNSAMPLE,
        min_value=0.0,
        max_value=0.9,
    )

    seed_raw = input("\nSeed aleatoria (ENTER para aleatorio): ").strip()
    if seed_raw:
        try:
            seed_value = int(seed_raw)
        except ValueError:
            print("Seed invalida, usando aleatoria.")
            seed_value = None
    else:
        seed_value = None
    if seed_value is not None:
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        random.seed()
        np.random.seed(None)

    output_root = prompt_output_dir(dataset_root)
    bucket_plans: List[Tuple[Bucket, Dict[str, List[SampleSpec]]]] = []

    for bucket in buckets:
        entries = plan_bucket(bucket, min_aug_when_down)
        if not entries:
            bucket_plans.append((bucket, {split: [] for split in SPLIT_NAMES}))
            continue
        random.shuffle(entries)
        split_counts = allocate_split_counts(len(entries), ratios)
        assigned: Dict[str, List[SampleSpec]] = {}
        cursor = 0
        for split in SPLIT_NAMES:
            need = split_counts.get(split, 0)
            assigned[split] = entries[cursor : cursor + need]
            cursor += need
        bucket_plans.append((bucket, assigned))

    write_dataset(output_root, bucket_plans)
    print(f"\nPronto! Dataset balanceado salvo em: {output_root}")


if __name__ == "__main__":
    main()
