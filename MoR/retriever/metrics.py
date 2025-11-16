import contextlib
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import faiss                      # type: ignore

from model.extract_feature import extract_embeddings
from model.retriever import search_retriever


@torch.no_grad()  # ⬅ 避免梯度计算，节省显存&加速
def compute_metrics_groups(
    model: torch.nn.Module,
    group_indices: Dict[int, faiss.Index],
    test_dataloader: DataLoader,
    group_labels: Dict[int, List[int]],
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Compute retrieval metrics (Top-1, Top-5, Top-10 accuracy) for grouped indices using a given model and test dataloader.

    Args:
        model (torch.nn.Module): The neural network model used for feature extraction.
        group_indices (Dict[int, faiss.Index]): A dictionary mapping group IDs to their corresponding FAISS indices.
        test_dataloader (DataLoader): DataLoader providing test samples. Each batch should contain a single image and its label.
        group_labels (Dict[int, List[int]]): A dictionary mapping group IDs to lists of label indices corresponding to the FAISS indices.
        device (torch.device): The device (CPU or GPU) on which computation is performed.

    Returns:
        Tuple[float, float, float]: A tuple containing overall Top-1, Top-5, and Top-10 accuracy across all groups.

    Notes:
        - The function assumes each batch from the dataloader contains only one image.
        - Prints per-group metrics to stdout.
        - Uses torch.no_grad() to avoid gradient computation for efficiency.
    """
    group_stats = {
        gid: {"top1": 0, "top5": 0, "top10": 0, "total": 0}
        for gid in group_indices
    }
    overall = {"top1": 0, "top5": 0, "top10": 0, "total": 0}

    model.eval()  # 确保 eval 模式
    for images, true_label, *_ in test_dataloader:
        images = images.to(device, non_blocking=True)
        true_label = true_label.item()  # 数据集中每批仅 1 张图

        # -------- 特征提取 --------
        feat = extract_embeddings(model, images)        # (1, D)
        group_id = true_label // 2
        index = group_indices[group_id]

        # -------- 检索 --------
        _, retrieved_idx = search_retriever(index, feat.cpu().numpy())
        retrieved_labels = [group_labels[group_id][i] for i in retrieved_idx]

        # -------- 逐级命中统计 --------
        ranks_hit = {
            "top1": true_label in retrieved_labels[:1],
            "top5": true_label in retrieved_labels[:5],
            "top10": true_label in retrieved_labels[:10],
        }
        for k, hit in ranks_hit.items():
            group_stats[group_id][k] += int(hit)
            overall[k] += int(hit)
        group_stats[group_id]["total"] += 1
        overall["total"] += 1

    # -------- 结果计算 --------
    def safe_div(num: int, denom: int) -> float:
        return num / denom if denom else 0.0

    # 分组结果打印
    for gid, stat in group_stats.items():
        print(
            f"[Group {gid}] "
            f"Top-1: {safe_div(stat['top1'], stat['total']):.4f} | "
            f"Top-5: {safe_div(stat['top5'], stat['total']):.4f} | "
            f"Top-10: {safe_div(stat['top10'], stat['total']):.4f} | "
            f"Samples: {stat['total']}"
        )

    overall_top1 = safe_div(overall["top1"], overall["total"])
    overall_top5 = safe_div(overall["top5"], overall["total"])
    overall_top10 = safe_div(overall["top10"], overall["total"])
    return overall_top1, overall_top5, overall_top10
