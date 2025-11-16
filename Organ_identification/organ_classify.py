#!/usr/bin/env python3
"""Image‑level inference script for Swin Transformer classifiers.

This refactored script cleans up the original prototype by
organising configuration handling, logging, error management and I/O.
It is written as an executable module so it can be invoked
both from the command line and imported elsewhere.
"""
from __future__ import annotations

import argparse
import json
import logging

import sys
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

# Local project imports -------------------------------------------------------
from class_data import IMG_FORMATS  # file‑name extensions considered images
from config import get_config       # Swin configuration loader
from .constants import CLASS_NAME_LIST  # human‑readable class names
from models import build_model      # Swin network factory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Argument parsing & configuration
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None):
    """Parse CLI arguments and merge with default Swin configs."""
    parser = argparse.ArgumentParser("Swin Transformer inference utility", add_help=False)

    # --- Swin config file & overrides
    parser.add_argument(
        "--cfg",
        default="configs/swin_small_patch4_window7_224.yaml",
        type=str,
        metavar="FILE",
        help="Path to a Swin Transformer YAML config file.",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs="+",
        help="Override config options by adding 'KEY VALUE' pairs.",
    )

    # --- Generic parameters
    parser.add_argument("--data-dir", required=True, type=Path, help="Root directory that holds the image dataset.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a trained model checkpoint (\*.pth).")
    parser.add_argument(
        "--output",
        default=Path("inference_results.json"),
        type=Path,
        metavar="FILE",
        help="Where to save the JSON results",
    )
    parser.add_argument("--batch-size", default=1, type=int, help="Images processed per GPU batch (only 1 supported currently).")
    parser.add_argument("--gpu", default=0, type=int, help="Which CUDA device to use (‑1 for CPU).")

    # Swin‑specific tuning knobs (kept for completeness)
    parser.add_argument("--zip", action="store_true", help="Use a zipped dataset instead of folder dataset.")
    parser.add_argument(
        "--cache-mode",
        default="part",
        choices=["no", "full", "part"],
        help="Dataset caching strategy (irrelevant for inference).",
    )
    parser.add_argument("--amp‑opt‑level", default="O1", choices=["O0", "O1", "O2"], help="Mixed precision level.")
    parser.add_argument("--num-classes", default=len(CLASS_NAME_LIST), type=int, help="Number of classes in the model.")

    args, _ = parser.parse_known_args(argv)
    cfg = get_config(args)  # merge CLI with YAML
    return args, cfg

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def list_images(root: Path) -> List[Path]:
    """Recursively gather all image files in *root* with allowed extensions."""
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_FORMATS]


def load_model(cfg, checkpoint_path: Path, device: torch.device):
    """Build a Swin model and load weights from *checkpoint_path*."""
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    LOGGER.info("Loaded checkpoint '%s'", checkpoint_path)
    return model


@torch.inference_mode()
def predict_single(model: torch.nn.Module, img_path: Path, transform: T.Compose, device: torch.device) -> int:
    """Run inference on a single image and return the predicted class ID."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", img_path, exc)
        raise

    tensor = transform(img).unsqueeze(0).to(device)
    out = model(tensor)
    scores = torch.nn.functional.softmax(out, dim=1)
    _, pred_label = scores.topk(1, dim=1)
    return int(pred_label.item())

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    args, cfg = parse_args(argv)

    # GPU/CPU selection
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    # Data transformations (ImageNet statistics)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Build network and load checkpoint
    model = load_model(cfg, args.checkpoint, device)

    # Iterate through dataset
    image_paths = list_images(args.data_dir)
    if not image_paths:
        LOGGER.error("No images found under %s", args.data_dir)
        sys.exit(1)

    LOGGER.info("Found %d images — starting inference…", len(image_paths))

    results: Dict[str, Dict[str, int | str]] = {}
    for img_path in tqdm(image_paths):
        try:
            class_id = predict_single(model, img_path, transform, device)
            class_name = CLASS_NAME_LIST[class_id] if 0 <= class_id < len(CLASS_NAME_LIST) else "Unknown"
        except Exception:
            class_id = -1
            class_name = "Error"

        results[str(img_path)] = {"class_id": class_id, "class_name": class_name}

    # Save results
    with args.output.open("w", encoding="utf‑8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)
    LOGGER.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
