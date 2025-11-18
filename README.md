# ORM-RAG
Organ-Aware Routing Mixture-of-Retrieval Augmented Generation for Fetal Ultrasound Reporting (AAAI-26)

> **Organ‑Retrieval‑Multimodal Retrieval‑Augmented Generation (ORM‑RAG)**
>
> The ORM-RAG (Organ Recognition Model with Retrieval-Augmented Generation) project is designed for medical image analysis. It focuses on organ classification, embedding training, and retrieval-based tasks for efficient similarity search. The project provides a modular workflow that includes data preparation, organ classification, embedding training, knowledge base construction, retrieval, and large model training/inference. It leverages Python scripts and shell scripts to automate tasks, making it suitable for researchers and developers working in medical imaging.

---

## 1. Repository Layout

```text
├── data/                        # Datasets & metadata (downloaded or custom)
│   ├── build_faiss_data.json    # data structures
│   ├── processed/               # Pre‑processed splits (train / val / test)
│   └── ...
├── docs/                        # Design notes & figures
├── internvl_chat/               # Upstream ViT checkpoints & helper scripts
|   ├── train.sh                     # LLM fine‑tuning launcher
|   ├── report_eval.sh               # LLM inference / evaluation script
│   └── ...
├── MoR/                         # Metric‑learning‑oriented Retrieval (MoR) module
|   ├── vit/                     # Vision‑Transformer backbone
│   ├── train.py                 # Embedding model training script
│   ├── faiss_retrieval_builder.py
│   ├── faiss_retrieval_pipeline.py
│   └── ...
├── Organ_identification/
|   ├── organ_classify.py            # Split multi‑organ datasets into organ folders
│   └── ...
├── requirements.txt             # Python dependencies
└── README.md                    # (this file)
```

---

## 2. Quick Start

### 2.1. Environment

```bash
# 1️⃣  Create and activate a fresh virtual environment
conda create -n orm_rag python=3.10 -y
conda activate orm_rag

# 2️⃣  Install required packages
pip install -r requirements.txt
```

### 2.2. Data Preparation

1. **Download** the official dataset archives to `data/` *or* place your own data using the same structure

2. **Organ split** (required once):

   ```bash
   python organ_classify.py
   ```

---

## 3. Train Organ‑Level Embeddings

Navigate to `MoR/` and run:

```bash
cd MoR
python train.py \
  --label-form 1 \
  --model-path /path/to/vit \
  --retrieval-json /path/to/test.json \
  --judge-json /path/to/judge.json
```

**Arguments**

| Flag               | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `--label-form`     | `num` → per‑organ labels .                                         |
| `--model-path`     | Folder containing ViT weights (`config.json`, `pytorch_model.bin`, etc.). |
| `--retrieval-json` | Validation set with organ labels.                                         |
| `--judge-json`     | Judgement file for metric evaluation.                                     |

Outputs (under `MoR/output/` by default):

- `model_best.pth` – Best checkpoint by validation loss.
- `embedding_cache.npy` – Frozen embeddings of all training images.

---

## 4. Build Knowledge Bases

Create a knowledge base for each organ to enable retrieval of the most relevant information. Finally, return the knowledge bases you have created.
```bash
python retrieval/faiss_retrieval_builder.py \
  --model-path /path/to/vit \
  --output-root /path/to/bank \
  --test-json /path/to/test.json
```

---

## 5. Retrieval Pipeline

Perform retrieval to obtain a dataset enriched with retrieval information.
```bash
python retrieval/faiss_retrieval_pipeline.py \
  --model_path   /path/to/vit \
  --faiss_root   /path/to/bank \
  --test_json    /path/to/test_ori.json \
  --judge_json   /path/to/judgement.json \
  --output_json  /path/to/test.json \
  --distance_json /path/to/distence.json
```

The script returns:

- `` – Each sample enriched with `topk_docs`, cosine distances & confidence.
- `` – Pair‑wise (retrieved‑vs‑ground‑truth) distance stats.

---

## 6. VLLM Training & Inference

### 6.1. Fine‑tune

```bash
bash train.sh
```

Adjust hyper‑parameters (`EPOCHS`, `LR`, etc.) inside the shell script. Logs and checkpoints are stored according to the `OUTPUT_DIR` variable.

### 6.2. Evaluate / Generate Report

```bash
bash report_eval.sh
```

The script loads the best checkpoint, runs inference on the evaluation split, and outputs a comprehensive CSV / JSON report.

---

## 8. Citation

If you build upon this work, please cite:

```bibtex
@misc{,
  title       = {Organ-Aware Routing Mixture-of-Retrieval Augmented Generation for Fetal Ultrasound Reporting},
  author      = {},
  year        = {2025},
  url         = {https://github.com/yourhandle/ORM-RAG}
}
```

---

## 9. License

[Apache 2.0](LICENSE) – free for academic and commercial use with attribution.

---

### Contact

Questions or PRs welcome!\
Wang · [siyuw8897@gmail.com](mailto\:siyuw8897@gmail.com) · Issues tab on GitHub

