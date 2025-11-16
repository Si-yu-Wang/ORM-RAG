#!/usr/bin/env python3
"""
faiss_retrieval_pipeline.py
------------------------------------
Refactored, production‑ready FAISS retrieval + evaluation script.

Major improvements over the original prototype:
* **Modular design** – clear classes/functions for loading indices, dataset handling, retrieval, and metrics.
* **Type hints & docstrings** – easier maintenance, IDE support, and static analysis.
* **No hidden globals** – state is encapsulated in class instances.
* **CLI interface** – paths, device, and batch‑size are configurable from the command line.
* **Structured logging** – replaces scattered `print` statements.
* **Graceful error handling** – early validation of resources + informative messages.
* **Cleaning up GPU usage** – `torch.no_grad()` + explicit `.eval()`.

Example usage:
```
python faiss_retrieval_pipeline.py \
    --model_path /public/Report-Ge/code/ORM-RAG/internvl_chat/vit \
    --faiss_root  /public/Report-Ge/code/ORM-RAG/internvl_chat/build_retrieval_faiss/EN_FAISS \
    --test_json   /public/Report-Ge/code/InternVL-wsy/internvl_chat/datasets/without_spine_and_kidney/test_en_删除肾脏异常和脊柱异常.json \
    --judge_json /public/Report-Ge/code/InternVL-wsy/internvl_chat/datasets/retrieval_data/judgement.json \
    --output_json /public/Report-Ge/code/ORM-RAG/en_test_with_retrieval_info_confidence.json \
    --distance_json /public/Report-Ge/code/ORM-RAG/en_test_distence_label_pair_confidence.json
```
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss                    # type: ignore
import numpy as np              # type: ignore
import torch
from torch.utils.data import Dataset
from transformers import AutoModel
from tqdm import tqdm

# Project‑specific imports
try:
    from model.tools import process_images, judge_label
    from model.extract_feature import extract_embeddings      
    from faiss_retrieval_builder import load_finetuned_model   
    from retriever.confidence import advanced_calculate_confidence_entropy
except ImportError as err:
    print("[E] Failed to import project modules – check PYTHONPATH.")
    raise err

###############################################################################
# Constants & type aliases
###############################################################################
from retriever.constants import NUM_GROUPS, INVALID_LABELS, LOGGER

# ---------------------------------------------------------------------------
# Helper dataclasses / containers
# ---------------------------------------------------------------------------

class RetrievalResult(Tuple[str, int, str, List[float], List[int], float]):
    """Return‑type shorthand for search results."""
    # Elements: report, label, image_path, distances, labels, confidence


###############################################################################
# FAISS group manager
###############################################################################

class RetrieverGroupManager:
    """Loads and keeps FAISS indices + metadata for all groups in memory."""

    def __init__(self, root: Path) -> None:
        self.indices: Dict[int, faiss.Index] = {}
        self.labels: Dict[int, np.ndarray] = {}
        self.reports: Dict[int, List[str]] = {}
        self.images: Dict[int, List[str]] = {}

        self._load_groups(root)

    # ---------------------------------------------------------------------
    def _load_groups(self, root: Path) -> None:
        index_dir   = root / "indices"
        labels_dir  = root / "labels"
        reports_dir = root / "reports"
        images_dir  = root / "images"

        for gid in range(NUM_GROUPS):
            if gid in {1, 7}:        # Skip unused groups, per original logic
                LOGGER.debug("Skipping group %d (unused)", gid)
                continue

            try:
                self.indices[gid] = faiss.read_index(str(index_dir / f"faiss_index_group_{gid}.index"))
                self.labels[gid]  = np.load(labels_dir / f"faiss_labels_group_{gid}.npy", allow_pickle=True)
                with open(reports_dir / f"faiss_reports_group_{gid}.pkl", "rb") as f_rep:
                    self.reports[gid] = pickle.load(f_rep)
                with open(images_dir / f"faiss_images_group_{gid}.pkl", "rb") as f_img:
                    self.images[gid] = pickle.load(f_img)
                LOGGER.info("Loaded FAISS data for group %d", gid)
            except FileNotFoundError as exc:
                LOGGER.error("Missing data for group %d: %s", gid, exc)
                raise

    # ---------------------------------------------------------------------
    def search(self, group_id: int, query: np.ndarray, top_k: int = 10) -> RetrievalResult:
        """Perform nearest‑neighbor search inside *one* group.

        Returns
        -------
        RetrievalResult : tuple
            (report, predicted_label, image_path, distances, label_list, confidence)
        """
        assert query.ndim == 2, "Query has to be 2‑D (batch, feat_dim)"
        retriever = self.indices[group_id]
        labels    = self.labels[group_id]
        reports   = self.reports[group_id]
        images    = self.images[group_id]

        distances, indices = retriever.search(query, top_k)
        distances = distances[0].astype(float).tolist()
        indices   = indices[0]

        label_list = [int(labels[i]) for i in indices]
        confidence = advanced_calculate_confidence_entropy(label_list)

        # Select best match (top‑1)
        best_idx = indices[0]
        return (
            reports[best_idx],              # report
            int(labels[best_idx]),          # predicted label
            images[best_idx],               # image path
            [round(d, 2) for d in distances],
            label_list,
            confidence,
        )

###############################################################################
# Metrics container
###############################################################################

class GroupMetrics:
    """Accumulates top‑1 accuracy & distance stats for each group."""

    def __init__(self) -> None:
        self.top1:   Dict[int, int] = {i: 0 for i in range(NUM_GROUPS)}
        self.total:  Dict[int, int] = {i: 0 for i in range(NUM_GROUPS)}
        self.dists:  Dict[int, Dict[str, List[dict]]] = {
            i: {"correct": [], "incorrect": []} for i in range(NUM_GROUPS)
        }

    # ---------------------------------------------------------------------
    def update(self, group_id: int, true_label: int, result: RetrievalResult) -> None:
        _, pred_label, _, dist_list, label_list, _ = result
        correct = pred_label == true_label
        key     = "correct" if correct else "incorrect"

        self.top1[group_id]  += int(correct)
        self.total[group_id] += 1

        self.dists[group_id][key].append({
            "true_label": true_label,
            "distances":  dist_list,
            "labels":     label_list,
        })

    # ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        acc = {
            gid: self.top1[gid] / self.total[gid]
            for gid in range(NUM_GROUPS)
            if self.total[gid] > 0
        }
        return {
            "accuracy": acc,
            "distance_stats": self.dists,
        }

###############################################################################
# Dataset wrapper
###############################################################################

class RetrievalDataset(Dataset):
    """JSON‑based dataset. Each item returns *raw* sample (no transforms)."""

    def __init__(self, json_path: Path):
        super().__init__()
        with open(json_path, "r", encoding="utf‑8") as f:
            self.samples: List[dict] = json.load(f)

    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:  # noqa: D401
        return self.samples[idx]

###############################################################################
# Core processing logic
###############################################################################

def flatten_sample(raw: dict) -> dict:
    """Re‑structure the nested fields created by torch DataLoader."""
    return {
        "sample_id": raw["sample_id"],
        "conversations": raw["conversations"],
        "image": raw.get("image", []),
        "retrieval_image": raw.get("retrieval_image", []),
        "retrieval_report": raw.get("retrieval_report", []),
    }


def run_inference(
    backbone: torch.nn.Module,
    retriever_mgr: RetrieverGroupManager,
    dataset: Dataset,
    judge_data:list,
    device: torch.device,
    conf_choice: bool = False
) -> Tuple[List[dict], dict]:
    """Main loop: iterate dataset, perform retrieval, accumulate metrics."""

    backbone.eval()
    metrics = GroupMetrics()
    updated_samples: List[dict] = []

    count=0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Retrieving"):
            images = sample["image"]
            true_labels, rag_dict = judge_label(judge_data, images)

            # One sample may contain multiple abnormalities
            relative_imgs: List[str]  = []
            relative_rpts: List[str]  = []
            max_conf_normal: float    = 0.0

            for true_label in true_labels:
                if true_label in INVALID_LABELS:
                    LOGGER.warning("Skipping invalid label %d in sample %s", true_label, sample["sample_id"])
                    continue

                group_id = true_label // 2
                group_model = load_finetuned_model(backbone, group_id, device).to(device).eval()

                img_tensor = process_images(rag_dict[true_label]).to(device)
                feat_vec   = extract_embeddings(group_model, img_tensor)

                result = retriever_mgr.search(group_id, feat_vec.cpu().numpy())
                metrics.update(group_id, true_label, result)

                report, pred_label, img_path, *_ , conf = result

                if conf_choice:
                    if conf > 1e-4:
                        if pred_label % 2 == 0:      
                            relative_imgs.append(img_path)
                            relative_rpts.append(report)
                        else:                          
                            if conf > max_conf_normal:
                                max_conf_normal = conf
                                normal_imgs=img_path
                                normal_rpts=report
                else:
                    if pred_label % 2 == 0:       
                        relative_imgs.append(img_path)
                        relative_rpts.append(report)
                    else:                          
                        if conf > max_conf_normal:
                            max_conf_normal = conf
                            normal_imgs=img_path
                            normal_rpts=report
            # If no abnormal images were found, use the best normal image
            if not relative_imgs and max_conf_normal > 0:
                relative_imgs.append(normal_imgs)
                relative_rpts.append(normal_rpts)

            # Append retrieval info into sample
            if relative_imgs:
                sample["retrieval_image"]  = relative_imgs
                sample["retrieval_report"] = relative_rpts

            updated_samples.append(flatten_sample(sample))

            count+=1
            if count % 10 == 0:
                break

    return updated_samples, metrics.to_dict()

###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAISS retrieval pipeline")
    parser.add_argument("--model_path",  type=Path, required=True, help="Path to VIT backbone")
    parser.add_argument("--faiss_root",  type=Path, required=True, help="Root dir that contains group FAISS data")
    parser.add_argument("--test_json",   type=Path, required=True, help="Test JSON file")
    parser.add_argument("--judge_json",   type=str, required=True, help="JSON file used for finding which organ")
    parser.add_argument("--output_json", type=Path, required=True, help="Path to save enriched JSON output")
    parser.add_argument("--distance_json", type=Path, required=True, help="Path to save distance / accuracy stats")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device override")
    parser.add_argument("--conf_choice", action="store_true", help="Use confidence thresholding for retrieval selection")
    return parser.parse_args()

###############################################################################
# Entry point
###############################################################################

def main() -> None:  # noqa: D401
    args = parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.judge_json, 'r', encoding='utf-8') as f2:
        judge_data = json.load(f2)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Using confidence thresholding: %s", args.conf_choice)

    # Load backbone once; downstream feature extractors are fine‑tuned clones
    backbone = AutoModel.from_pretrained(str(args.model_path)).to(device).eval()

    retriever_mgr = RetrieverGroupManager(args.faiss_root)
    dataset       = RetrievalDataset(args.test_json)

    enriched_samples, stats = run_inference(backbone, retriever_mgr, dataset, judge_data, device, args.conf_choice)

    # Persist outputs
    args.output_json.write_text(json.dumps(enriched_samples, ensure_ascii=False, indent=4))
    args.distance_json.write_text(json.dumps(stats, ensure_ascii=False, indent=4))

    LOGGER.info("Saved enriched samples to %s", args.output_json)
    LOGGER.info("Saved distance stats   to %s", args.distance_json)

    for gid, acc in stats["accuracy"].items():
        LOGGER.info("Group %d – Top‑1 accuracy: %.4f", gid, acc)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        sys.exit(1)
