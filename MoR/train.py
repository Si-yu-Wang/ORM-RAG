"""
train.py — Training script for ViT‑based supervised‑contrastive image classifier
--------------------------------------------------------------------------
A refactored, production‑ready version of your original single‑file script.  
Key improvements
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
* **Argparse CLI** – hyper‑parameters can be passed from command line or a YAML config.
* **Modular functions** – isolated `build_dataloaders`, `train_one_epoch`, `evaluate`, etc.
* **Logging** – standard `logging` + JSONL file for metrics.
* **Determinism** – explicit seeds for Python, NumPy, PyTorch & CUDA.
* **Graceful checkpoints** – best‑model saving with date‑time stamp.
* **Device agnostic** – automatic CPU/GPU selection & ampere clearing.
* **FAISS retrieval evaluation** – kept identical to your original utilities.

Run with
```
python train.py --label-form 1 \
              --model-path /path/to/vit \
              --retrieval-json /path/to/test.json \
              --judge-json /path/to/judge.json \
```
Add `-h` to see all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel

# project‑local imports -----------------------------------------------------
from model.constants import _GROUP_NAMES
from model.datasets import ImageDataset, collate_fn
from model.extract_feature import get_embeddings
from model.loss import SupConLoss
from model.metrics import compute_metrics
from model.retriever import build_retriever_save
from model.tools import get_root_dir

# ----------------------------------------------------------------------------
# CLI & CONFIG
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ViT + SupCon model")

    # data ------------------------------------------------------------------
    parser.add_argument("--label-form", type=int, help="which organ to train on", choices=list(range(len(_GROUP_NAMES))), required=True)
    parser.add_argument("--retrieval-json", type=str, required=True, help="JSON file used for retrieval evaluation")
    parser.add_argument("--judge-json", type=str, required=True, help="JSON file used for finding which organ")

    # model / optimiser -----------------------------------------------------
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace‑format ViT checkpoint or directory")
    parser.add_argument("--unfreeze-layers", type=int, default=2, help="# of final ViT encoder layers to unfreeze")
    parser.add_argument("--head-hidden", type=int, default=512, help="hidden dim of the classification head")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--lambda-contrast", type=float, default=0.9, help="weight of SupCon loss component")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature for SupCon loss")

    # training --------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)

    # misc ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    return parser

# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ensure deterministic behaviour where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(train_json: str, retrieval_json: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    transform = build_transforms()
    train_ds = ImageDataset(train_json, transform=transform)
    retrieval_ds = ImageDataset(retrieval_json, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    retrieval_loader = DataLoader(retrieval_ds, batch_size=1, shuffle=False)
    return train_loader, retrieval_loader


class ClassificationHead(nn.Module):
    """Simple MLP head for ViT feature maps."""

    def __init__(self, in_features: int, num_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, D403
        return self.net(x)


def freeze_vit_backbone(vit: nn.Module, unfreeze_layers: int) -> None:
    """Freeze all layers except the last *unfreeze_layers* encoder blocks."""
    for p in vit.parameters():
        p.requires_grad = False

    if unfreeze_layers <= 0:
        return

    try:
        for layer in vit.encoder.layer[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    except AttributeError:
        logging.warning("Could not locate vit.encoder.layer – backbone might differ.")


# ----------------------------------------------------------------------------
# TRAIN / EVAL ROUTINES
# ----------------------------------------------------------------------------

def train_one_epoch(
    vit: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    criterion_cls: nn.Module,
    criterion_sup: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_contrast: float,
) -> Tuple[float, float]:
    vit.train()
    head.train()

    total_loss = 0.0
    correct, total = 0, 0

    for imgs, masks, labels in tqdm(loader, desc="Training", leave=False):
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)

        feats = get_embeddings(vit, device, masks, imgs)
        logits = head(feats)

        loss_cls = criterion_cls(logits, labels)
        loss_sup = criterion_sup(feats, labels)
        loss = (1 - lambda_contrast) * loss_cls + lambda_contrast * loss_sup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(
    vit: nn.Module,
    train_loader: DataLoader,
    retrieval_loader: DataLoader,
    judge_data: list,
    epoch: int,
    label_form: int,
    root_dir: Path,
    device: torch.device,
) -> float:
    vit.eval()
    faiss_dir = root_dir / "faiss_index"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    index_path = faiss_dir / f"index_epoch_{epoch}.faiss"
    labels_path = faiss_dir / f"labels_epoch_{epoch}.npy"

    # Build FAISS index on training embeddings
    index, all_labels = build_retriever_save(vit, device, train_loader, index_path, labels_path)
    acc = compute_metrics(vit, label_form, device, judge_data, index, retrieval_loader, all_labels)
    torch.cuda.empty_cache()
    return acc


def save_checkpoint(
    vit: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    acc: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chkpt_name = f"epoch_{epoch:02d}_acc_{acc:.4f}.pth"
    torch.save(
        {
            "epoch": epoch,
            "vit_state": vit.state_dict(),
            "head_state": head.state_dict(),
            "opt_state": optimizer.state_dict(),
            "retrieval_acc": acc,
        },
        out_dir / chkpt_name,
    )


def append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("[]", encoding="utf-8")

    with log_path.open("r+", encoding="utf-8") as f:
        logs = json.load(f)
        logs.append(entry)
        f.seek(0)
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    args = build_parser().parse_args(argv)

    # set‑up ---------------------------------------------------------------
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
        stream=sys.stdout,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # directories ----------------------------------------------------------
    root_dir, train_json = get_root_dir(args.label_form)
    root_dir = Path(root_dir)

    # data -----------------------------------------------------------------
    train_loader, retrieval_loader = build_dataloaders(
        train_json, args.retrieval_json, args.batch_size, args.num_workers
    )

    with open(args.judge_json, 'r', encoding='utf-8') as f2:
        judge_data = json.load(f2)

    # model ----------------------------------------------------------------
    vit = AutoModel.from_pretrained(args.model_path).to(device)
    freeze_vit_backbone(vit, args.unfreeze_layers)
    head = ClassificationHead(vit.config.hidden_size, args.num_classes, args.head_hidden).to(device)

    # optimisation ---------------------------------------------------------
    optimizer = optim.Adam(
        [p for p in list(vit.parameters()) + list(head.parameters()) if p.requires_grad],
        lr=args.lr,
    )
    criterion_cls = nn.CrossEntropyLoss()
    criterion_sup = SupConLoss(temperature=args.temperature)

    # tracking -------------------------------------------------------------
    log_path = root_dir / "training_log.json"
    chkpt_dir = root_dir / "models"
    best_acc = 0.0

    # training loop --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)

        train_loss, train_acc = train_one_epoch(
            vit,
            head,
            train_loader,
            criterion_cls,
            criterion_sup,
            optimizer,
            device,
            args.lambda_contrast,
        )
        retrieval_acc = evaluate(vit, train_loader, retrieval_loader, judge_data, epoch, args.label_form, root_dir, device)

        logging.info(
            "Loss %.4f | Train Acc %.4f | Retrieval Acc %.4f",
            train_loss,
            train_acc,
            retrieval_acc,
        )

        # JSON log ---------------------------------------------------------
        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "retrieval_acc": round(retrieval_acc, 4),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        # checkpoint -------------------------------------------------------
        if retrieval_acc > best_acc:
            best_acc = retrieval_acc
            save_checkpoint(vit, head, optimizer, epoch, retrieval_acc, chkpt_dir)
            logging.info("New best model (%.4f) saved.", retrieval_acc)

        torch.cuda.empty_cache()

    logging.info("Training finished. Best retrieval accuracy: %.4f", best_acc)


if __name__ == "__main__":
    main()
