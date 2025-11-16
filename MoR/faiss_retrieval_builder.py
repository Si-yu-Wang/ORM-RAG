"""faiss_retrieval_builder.py
--------------------------------
Build or load per‑group FAISS indices for the InternVL retrieval system and
compute retrieval metrics.

Usage (example):

```bash
python faiss_retrieval_builder.py \
    --model-path /public/Report-Ge/code/InternVL-wsy/internvl_chat/vit \
    --output-root /public/Report-Ge/code/ORM-RAG/internvl_chat/build_retrieval_faiss/EN_FAISS1 \
    --test-json /public/Report-Ge/code/InternVL-wsy/internvl_chat/datasets/all_organ_250316/test.json
```

A subsequent run with the same ``--output-root`` will reuse the saved indices.

The script deliberately avoids building indices for label groups **1** and **7**
(these correspond to pathological/duplicate categories in the dataset).
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss                      # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

# Local project utils (make sure these are import‑able)
from model.constants import CKPT_DICT,SKIPPED_GROUPS
from model.extract_feature import extract_embeddings
from model.tools import get_root_dir
from retriever.datasets import BBImageDataset
from retriever.metrics import compute_metrics_groups

# ---------------------------------------------------------------------------
# Helpers & configuration
# ---------------------------------------------------------------------------

def init_logger(verbose: bool = False) -> None:
    """Configure a simple console logger."""
    fmt = "%(asctime)s | %(levelname)s | %(message)s"  # noqa: W505
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=fmt, datefmt="%Y-%m-%d %H:%M:%S", level=level)


def load_finetuned_model(base_model: AutoModel, label_id: int, device: torch.device) -> AutoModel:
    """Load a fine‑tuned checkpoint onto *base_model* and return it (in‑place)."""
    ckpt_path = CKPT_DICT[int(label_id)]
    state = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(state["model_state_dict"], strict=False)
    base_model.eval()
    return base_model

# ---------------------------------------------------------------------------
# Core retriever logic
# ---------------------------------------------------------------------------

class RetrieverManager:
    """Builds and stores FAISS indices per label group."""

    def __init__(self, model_path: str, output_root: Path, device: torch.device):
        self.device = device
        self.output_root = output_root
        self.model_path = model_path

        self.base_model = AutoModel.from_pretrained(model_path).to(device)
        self.feature_dim = self.base_model.config.hidden_size

        # Containers
        self.group_indices: Dict[int, faiss.IndexFlatIP] = {}
        self.group_labels: Dict[int, List[int]] = {}
        self.group_reports: Dict[int, List[str]] = {}
        self.group_images: Dict[int, List[str]] = {}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def _init_dirs(self) -> None:
        os.makedirs(self.output_root / "indices", exist_ok=True)
        os.makedirs(self.output_root / "labels", exist_ok=True)
        os.makedirs(self.output_root / "reports", exist_ok=True)
        os.makedirs(self.output_root / "images", exist_ok=True)

    def build_or_load(self) -> None:
        """Load indices from disk if available; otherwise build them."""
        if self._indices_exist():
            logging.info("Cached FAISS indices found. Loading …")
            self._load()
        else:
            logging.info("No cached indices found. Building from scratch …")
            self._init_dirs()
            self._build()
            self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ---- disk IO -------------------------------------------------------

    def _indices_exist(self) -> bool:
        """Return *True* if *all* required files are present on disk."""
        for group_id in range(8):
            if group_id in SKIPPED_GROUPS:
                continue
            if not (self._index_path(group_id).exists() and self._labels_path(group_id).exists() and
                    self._reports_path(group_id).exists() and self._images_path(group_id).exists()):
                return False
        return True

    def _save(self) -> None:
        """Persist every group's FAISS index and associated metadata."""
        for gid in range(8):
            if gid in SKIPPED_GROUPS:
                continue
            faiss.write_index(self.group_indices[gid], str(self._index_path(gid)))
            np.save(str(self._labels_path(gid)), np.array(self.group_labels[gid]))
            with open(str(self._reports_path(gid)), "wb") as f:
                pickle.dump(self.group_reports[gid], f)
            with open(str(self._images_path(gid)), "wb") as f:
                pickle.dump(self.group_images[gid], f)
            logging.debug("Saved group %d to disk", gid)

    def _load(self) -> None:
        """Load all group indices and metadata into memory."""
        for gid in range(8):
            if gid in SKIPPED_GROUPS:
                continue
            self.group_indices[gid] = faiss.read_index(str(self._index_path(gid)))
            self.group_labels[gid] = np.load(self._labels_path(gid), allow_pickle=True).tolist()
            with open(self._reports_path(gid), "rb") as f:
                self.group_reports[gid] = pickle.load(f)
            with open(self._images_path(gid), "rb") as f:
                self.group_images[gid] = pickle.load(f)
            logging.debug("Loaded group %d from disk", gid)

    # ---- building ------------------------------------------------------

    def _build(self) -> None:
        """Iterate over the entire dataset, splitting samples into groups and
        feeding them into independent FAISS indices.
        """
        for gid in range(8):
            if gid in SKIPPED_GROUPS:
                continue
            # Init per‑group containers.
            self.group_indices[gid] = faiss.IndexFlatIP(self.feature_dim)
            self.group_labels[gid] = []
            self.group_reports[gid] = []
            self.group_images[gid] = []

        for label_id in range(8):
            if label_id in SKIPPED_GROUPS:
                continue
            _, bank_json = get_root_dir(label_id)
            dataset = BBImageDataset(bank_json)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            # Load fine‑tuned weights *once* for this label (multiple batches share it)
            model_ft = load_finetuned_model(self.base_model, label_id, self.device)

            for images, label, image_paths, report in tqdm(loader, desc=f"Group {label_id}"):
                images = images.to(self.device)
                emb = extract_embeddings(model_ft, images)               # (1, d)

                group_id = int(label.item()) // 2  # e.g. labels 0/1 -> group 0
                if group_id in SKIPPED_GROUPS:
                    continue

                # Update collections
                self.group_indices[group_id].add(emb.cpu().numpy())
                self.group_labels[group_id].append(label.item())
                self.group_reports[group_id].append(report[0])
                self.group_images[group_id].append(random.choice(image_paths))

    # ---- utility -------------------------------------------------------

    def _index_path(self, gid: int) -> Path:
        return self.output_root / "indices" / f"faiss_index_group_{gid}.index"

    def _labels_path(self, gid: int) -> Path:
        return self.output_root / "labels" / f"faiss_labels_group_{gid}.npy"

    def _reports_path(self, gid: int) -> Path:
        return self.output_root / "reports" / f"faiss_reports_group_{gid}.pkl"

    def _images_path(self, gid: int) -> Path:
        return self.output_root / "images" / f"faiss_images_group_{gid}.pkl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or load FAISS indices and compute retrieval metrics.")
    parser.add_argument("--model-path", required=True, help="Path to the base ViT checkpoint.")
    parser.add_argument("--output-root", required=True, help="Directory where indices / metadata are stored.")
    parser.add_argument("--test-json", required=True, help="JSON file used for evaluation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size (default: 1).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--eval", action="store_true", help="Please choose skipping evaluation after building/loading indices, if your data is not with labels")
    return parser.parse_args()


def main():
    args = parse_args()
    init_logger(args.verbose)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build / load indices ------------------------------------------------
    manager = RetrieverManager(args.model_path, output_root, device)
    manager.build_or_load()
    
    if args.eval:
        # Evaluation ----------------------------------------------------------
        test_ds = BBImageDataset(args.test_json,k=2)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        logging.info("Evaluating retrieval performance …")
        top1, top5, top10 = compute_metrics_groups(
            manager.base_model, manager.group_indices, test_dl, manager.group_labels, device
        )
        logging.info("Top‑1  accuracy = %.4f", top1)
        logging.info("Top‑5  accuracy = %.4f", top5)
        logging.info("Top‑10 accuracy = %.4f", top10)
    else:
        print("Skipping evaluation")


if __name__ == "__main__":
    main()
