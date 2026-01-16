# MedVQA on VQA-RAD (Baseline vs BLIP) — WOA7015 Alternative Assignment

This repository contains a WOA7015 course project implementing and comparing two approaches for **Medical Visual Question Answering (Med-VQA)** on the **VQA-RAD** dataset:

- **Baseline (discriminative)**: CNN–RNN VQA model (ResNet image encoder + GRU question encoder + fusion + answer heads)
- **BLIP (vision–language model)**: BLIP VQA model in **zero-shot** and **fine-tuned** settings

We analyze performance on:
- **Closed-ended questions (Yes/No)** → Accuracy
- **Open-ended questions (free-text)** → EM / BLEU / ROUGE-L (and optional mapped metrics if enabled)

This project runs on **Apple Silicon (M2 Pro) using PyTorch MPS** (no NVIDIA GPU required).

---

## Repository Structure

Current structure (as in this repo):

```
medvqa-vqarad/
  README.md
  requirements.txt
  data/
  checkpoints/
  runs/
  src/
    __init__.py
    prepare_data.py
    evaluate.py
    train_baseline.py
    train_blip.py
    datasets/
      vqarad.py
    models/
      baseline_cnn_rnn.py
      blip_vqa.py
    utils/
      device.py
      io.py
      metrics.py
      seed.py
      text.py
```

- `src/prepare_data.py` prepares dataset splits / data layout (if needed).
- `src/train_baseline.py` trains the baseline model.
- `src/train_blip.py` fine-tunes BLIP.
- `src/evaluate.py` evaluates baseline / BLIP on train/val/test splits.
- `data/` stores images and split files.
- `checkpoints/` stores trained weights.
- `runs/` stores evaluation results and training logs.

---

## Environment Setup

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation (VQA-RAD)

This repo expects the dataset under `data/`:

- Images: `data/raw/images/`
- Splits: `data/splits/{train,val,test}.json`

If your data is not in this format, use the preparation script:

```bash
python -m src.prepare_data --data_dir data
```

> Note: The exact behavior depends on how `prepare_data.py` is implemented in this repo.
> If you already have `data/raw/images/` and `data/splits/` ready, you can skip this step.

---

## Training

### A) Train Baseline (CNN–RNN)
Recommended for Apple Silicon (MPS):

```bash
python -m src.train_baseline --data_dir data --device auto --unfreeze_layer4
```

Common optional hyperparameters:
```bash
python -m src.train_baseline --data_dir data --device auto --unfreeze_layer4 \
  --stage1_epochs 12 --stage2_epochs 10 --lr 2e-4 --batch_size 64
```

Outputs:
- `checkpoints/baseline.pt`
- `checkpoints/baseline_maps.json`
- training history under `runs/`

---

### B) BLIP

#### 1) Zero-shot (no training)
See Evaluation below.

#### 2) Fine-tune BLIP (small setting)
A lightweight fine-tuning configuration (suitable for limited compute):

```bash
python -m src.train_blip --data_dir data --device auto --epochs 1 --batch_size 2 \
  --lr 5e-6 --max_train_samples 200 --max_val_samples 200
```

Output:
- `checkpoints/blip.pt`

---

## Evaluation

### 1) Evaluate Baseline
```bash
python -m src.evaluate --data_dir data --model baseline \
  --ckpt checkpoints/baseline.pt --split test --device auto
```

### 2) Evaluate BLIP (zero-shot)
```bash
python -m src.evaluate --data_dir data --model blip \
  --blip_mode zeroshot --split test --device auto
```

### 3) Evaluate BLIP (fine-tuned)
```bash
python -m src.evaluate --data_dir data --model blip \
  --blip_mode finetuned --ckpt checkpoints/blip.pt --split test --device auto
```

Evaluation results are written to `runs/` as JSON files.

---

## Final Results (Test Split)

Final test results used in the report (Apple M2 Pro, MPS):

- **Baseline (final)**: closed yes/no accuracy = **0.6304**
- **BLIP (zero-shot)**: closed yes/no accuracy = **0.5217**
- **BLIP (fine-tuned)**: closed yes/no accuracy = **0.6087**

Test set sizes (from evaluation logs):
- closed questions: 138
- open questions: 92

---

## Reproducibility Notes

- Use `--device auto` to select `mps` on Apple Silicon when available.
- Results may vary slightly due to randomness (seed) and small-sample fine-tuning settings.
- Open-ended metrics (EM/BLEU/ROUGE) are strict; semantically correct paraphrases can still score low.

---

## Recommended .gitignore (Important)

Do **not** commit large or private files (dataset/images/checkpoints). Create a `.gitignore` such as:

```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store

data/
checkpoints/
runs/

*.pt
*.zip
```

---

## GitHub Link (for the report)

After you create a GitHub repository and push this project:

```bash
git remote -v
```

The GitHub link to include in the report is the HTTPS repository page, e.g.:
`https://github.com/<username>/<repo>`
