#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_resnetrs50.py
=================

This script trains a ResNetRS‑50 model (the efficient ResNet variant) on a folder of
images using PyTorch and the timm library. It duplicates the training pipeline
from the reference notebook, including stratified splitting, data augmentation,
mixed‑precision training, early stopping, learning rate scheduling, and
evaluation. All outputs—training curves, confusion matrices, classification
report, and model weights—are saved into a results directory.

Dataset structure:
    /home/mateus/data/output-001/
        class_a/
        class_b/
        ...

If `output-001` contains `train/` and `validation/` subfolders, those
splits are used directly; otherwise, a stratified 80/20 split is performed.

Images are assumed to be RGB; they are resized to 224×224 for ResNetRS‑50.
"""

import os
import csv
import random
import warnings
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import timm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


class HierarchicalImageFolder(torch.utils.data.Dataset):
    """ImageFolder variant that treats the first-level subdirectories as class labels.

    Any deeper subdirectories are traversed recursively so that images nested in
    multiple levels still map back to their top-level class folder.
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        self._scan()

    def _scan(self):
        class_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class folders found in '{self.root}'.")
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.classes)
                self.classes.append(class_name)
            class_idx = self.class_to_idx[class_name]
            for file_path in sorted(class_dir.rglob('*')):
                if file_path.is_file() and file_path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((file_path, class_idx))
        if not self.samples:
            raise RuntimeError(f"No images found in '{self.root}'. Ensure valid extensions: {IMG_EXTENSIONS}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def main():
    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    use_channels_last = torch.cuda.is_available()
    print(f"Using device: {device} | Mixed precision: {use_amp}")

    # Use UNC path for dataset with explicit splits
    DATA_ROOT = Path(r'/home/mateus/data')
    if not (DATA_ROOT / 'train').exists() or not (DATA_ROOT / 'val').exists() or not (DATA_ROOT / 'test').exists():
        raise FileNotFoundError(
            f"Esperava encontrar as pastas 'train', 'validation' e 'test' dentro de {DATA_ROOT}; "
            f"verifique a estrutura do diretório."
        )
    model_name = 'resnetrs50'
    results_dir = Path(f'results/{model_name}')
    results_dir.mkdir(parents=True, exist_ok=True)

    input_size = 224
    val_resize = 256
    BATCH_SIZE = 200

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.RandomAutocontrast(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load explicit splits; stratified split is no longer performed
    train_ds = HierarchicalImageFolder(DATA_ROOT / 'train', transform=train_tfms)
    val_ds   = HierarchicalImageFolder(DATA_ROOT / 'val', transform=val_tfms)
    test_ds  = HierarchicalImageFolder(DATA_ROOT / 'test', transform=val_tfms)
    class_names = train_ds.classes

    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    base_weights = {'benignos': 12, 'malignos': 24, 'pre-malignos': 5}
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for name, idx in train_ds.class_to_idx.items():
        weight = base_weights.get(name.lower(), 1.0)
        class_weights[idx] = weight
        print(f"Class '{name}' (index {idx}) assigned weight {weight}.")
    class_weights = class_weights.to(device)

    num_workers = 8 if torch.cuda.is_available() else 0
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    # Load ResNetRS‑50 from timm
    model = timm.create_model('resnetrs50', pretrained=True)
    # Freeze parameters selectively to allow deeper fine-tuning
    for name, param in model.named_parameters():
        param.requires_grad = False
        if name.startswith('layer3') or name.startswith('layer4') or name.startswith('conv_head'):
            param.requires_grad = True
    # Adjust the classifier head for our number of classes
    in_features = None
    dropout_rate = 0.3

    if hasattr(model, "get_classifier"):
        in_features = model.get_classifier().in_features
        new_head = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes))
        if hasattr(model, "classifier"):
            model.classifier = new_head
        elif hasattr(model, "fc"):
            model.fc = new_head
    else:
        # Fallback for models without get_classifier
        in_features = model.fc.in_features if hasattr(model, 'fc') else model.classifier.in_features
        new_head = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes))
        if hasattr(model, 'fc'):
            model.fc = new_head
        else:
            model.classifier = new_head
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    layer3_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('layer3')]
    layer4_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('layer4')]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith('fc') or n.startswith('classifier') or n.startswith('conv_head'))]
    optimizer = torch.optim.Adam(
        [
            {"params": layer3_params, "lr": 3e-4, "weight_decay": 3e-5},
            {"params": layer4_params, "lr": 5e-4, "weight_decay": 3e-5},
            {"params": head_params,   "lr": 1e-3, "weight_decay": 3e-5},
        ]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    class EarlyStopping:
        def __init__(self, patience=100, min_delta=0.0, path=results_dir / 'best_model.pth'):
            self.patience = patience
            self.min_delta = min_delta
            self.path = Path(path)
            self.best = -float('inf')
            self.counter = 0
            self.best_state = None

        def step(self, metric, model):
            if metric > self.best + self.min_delta:
                self.best = metric
                self.counter = 0
                self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                return False
            self.counter += 1
            return self.counter >= self.patience

        def has_checkpoint(self):
            return self.best_state is not None

        def save_checkpoint(self):
            if self.best_state is not None:
                torch.save(self.best_state, self.path)

    # Early stopping keeps the best validation accuracy and stops after extended patience without improvement
    early_stopper = EarlyStopping(patience=100, min_delta=1e-4, path=results_dir / 'best_model.pth')

    tta_transforms = (
        lambda batch: torch.flip(batch, dims=[-1]),
        lambda batch: torch.flip(batch, dims=[-2]),
    )
    tta_factor = 1 + len(tta_transforms)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def run_epoch(model, loader, stage, train=True, use_tta=False):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        progress = tqdm(loader, desc=stage, leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            if use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    if use_tta and tta_transforms:
                        variants = [images]
                        for tta_fn in tta_transforms:
                            aug_images = tta_fn(images)
                            if use_channels_last:
                                aug_images = aug_images.contiguous(memory_format=torch.channels_last)
                            else:
                                aug_images = aug_images.contiguous()
                            variants.append(aug_images)
                        stacked = torch.cat(variants, dim=0)
                        if use_channels_last:
                            stacked = stacked.to(memory_format=torch.channels_last)
                        logits = model(stacked)
                        batch_size = images.shape[0]
                        logits = logits.reshape(tta_factor, batch_size, -1)
                        outputs = logits.mean(dim=0)
                    else:
                        outputs = model(images)
                    loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
            if total:
                progress.set_postfix(loss=total_loss / total, acc=total_correct / total)
        progress.close()
        avg_loss = total_loss / total
        accuracy = total_correct / total
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        f1 = f1_score(y_true, y_pred, average='macro')
        return avg_loss, accuracy, f1, y_true, y_pred

    def save_plot(history, key1, key2, title, ylabel, filename):
        plt.figure(figsize=(6, 4))
        plt.plot(history[key1], label=key1)
        plt.plot(history[key2], label=key2)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        outfile = results_dir / filename
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()

    def save_confusion_matrix(cm, classes, normalize, title, filename):
        if normalize:
            cm_to_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        else:
            cm_to_plot = cm
        plt.figure(figsize=(6, 4))
        plt.imshow(cm_to_plot, interpolation='nearest', aspect='auto')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm_to_plot.max() / 2.0
        for i, j in itertools.product(range(cm_to_plot.shape[0]), range(cm_to_plot.shape[1])):
            value = cm_to_plot[i, j]
            plt.text(j, i, format(value, fmt),
                     horizontalalignment="center",
                     color="white" if value > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(results_dir / filename)
        plt.close()

    EPOCHS = 700
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, stage=f"Train {epoch}", train=True)
        val_loss, val_acc, val_f1, y_true, y_pred = run_epoch(model, val_loader, stage=f"Val {epoch}", train=False, use_tta=True)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"[{epoch}/{EPOCHS}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | lr={current_lr:.6f}")
        epoch_bar.set_postfix(train_loss=tr_loss, val_acc=val_acc, lr=current_lr)
        if early_stopper.step(val_acc, model):
            print(f"Early stopping triggered at epoch {epoch}. Best val_acc: {early_stopper.best:.4f}")
            break
    epoch_bar.close()

    best_model_path = early_stopper.path
    if early_stopper.has_checkpoint():
        model.load_state_dict(early_stopper.best_state)
        early_stopper.save_checkpoint()
        print(f"Best model saved to {best_model_path}")
    else:
        torch.save(model.state_dict(), best_model_path)
        print(f"No early stopping checkpoint; saved final model to {best_model_path}")

    # Final evaluation on validation and test sets
    val_loss, val_acc, val_f1, y_true_val, y_pred_val = run_epoch(model, val_loader, stage="Val Eval", train=False, use_tta=True)
    print("\nValidation metrics (best model):")
    print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1 (macro): {val_f1:.4f}")
    report_val = classification_report(y_true_val, y_pred_val, target_names=class_names)
    print("\n[VAL] Classification report:\n", report_val)
    cm_val = confusion_matrix(y_true_val, y_pred_val)
    test_loss, test_acc, test_f1, y_true_test, y_pred_test = run_epoch(model, test_loader, stage="Test Eval", train=False, use_tta=True)
    print("\nTest metrics (best model):")
    print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1 (macro): {test_f1:.4f}")
    report_test = classification_report(y_true_test, y_pred_test, target_names=class_names)
    print("\n[TEST] Classification report:\n", report_test)
    cm_test = confusion_matrix(y_true_test, y_pred_test)

    save_plot(history, 'train_loss', 'val_loss', 'Learning curve (loss)', 'loss', 'loss_curve.png')
    save_plot(history, 'train_acc', 'val_acc', 'Accuracy curve', 'accuracy', 'accuracy_curve.png')
    # Confusion matrices for validation and test
    save_confusion_matrix(cm_val, class_names, normalize=False,
                          title='Matriz de Confusão (Bruta) - VAL', filename='confusion_matrix_val_raw.png')
    save_confusion_matrix(cm_val, class_names, normalize=True,
                          title='Matriz de Confusão (Normalizada) - VAL', filename='confusion_matrix_val_norm.png')
    save_confusion_matrix(cm_test, class_names, normalize=False,
                          title='Matriz de Confusão (Bruta) - TEST', filename='confusion_matrix_test_raw.png')
    save_confusion_matrix(cm_test, class_names, normalize=True,
                          title='Matriz de Confusão (Normalizada) - TEST', filename='confusion_matrix_test_norm.png')
    # Save history as CSV
    history_path = results_dir / 'history.csv'
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_acc'][i],
                history['val_acc'][i],
                history['val_f1'][i],
            ])
    # Save classification reports
    report_val_path = results_dir / 'classification_report_val.txt'
    with open(report_val_path, 'w') as f:
        f.write(report_val)
    report_test_path = results_dir / 'classification_report_test.txt'
    with open(report_test_path, 'w') as f:
        f.write(report_test)
    # Save confusion matrices as CSV
    np.savetxt(results_dir / 'cm_val_raw.csv', cm_val, delimiter=',', fmt='%d')
    np.savetxt(results_dir / 'cm_val_norm.csv', cm_val.astype(float) / cm_val.sum(axis=1, keepdims=True), delimiter=',', fmt='%.6f')
    np.savetxt(results_dir / 'cm_test_raw.csv', cm_test, delimiter=',', fmt='%d')
    np.savetxt(results_dir / 'cm_test_norm.csv', cm_test.astype(float) / cm_test.sum(axis=1, keepdims=True), delimiter=',', fmt='%.6f')
    # Summarize metrics in JSON
    import json
    summary = {
        'val_loss': float(val_loss), 'val_acc': float(val_acc), 'val_f1_macro': float(val_f1),
        'test_loss': float(test_loss), 'test_acc': float(test_acc), 'test_f1_macro': float(test_f1),
        'classes': class_names,
    }
    summary_path = results_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results saved in: {results_dir.resolve()}")


if __name__ == '__main__':
    main()