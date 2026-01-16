# Med-VQA (VQA-RAD) Baseline vs BLIP — CPU-friendly project (PyCharm)

This project compares:
1) **Baseline**: ResNet18 (image) + GRU (question) + MLP fusion  
2) **BLIP**: `Salesforce/blip-vqa-base` (generative VQA)

Target analysis:
- **Closed** questions (**Yes/No**) → Accuracy
- **Open** questions (free-text) → Exact Match + BLEU + ROUGE-L

> Designed for **CPU-only** Macs (slow but workable). Use `--max_*_samples` to run quick experiments.

---

## 0. Put the dataset files

Copy into:
- `data/raw/VQA_RAD Dataset Public.json`
- `data/raw/VQA_RAD Image Folder.zip`

---

## 1. Create a venv and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

---

## 2. Prepare data (extract images + image-level split)

```bash
python -m src.prepare_data \
  --json_path data/raw/VQA_RAD\ Dataset\ Public.json \
  --image_zip data/raw/VQA_RAD\ Image\ Folder.zip \
  --out_dir data \
  --seed 42
```

Creates:
- `data/raw/images/`
- `data/splits/train.json`, `val.json`, `test.json`

Split is **image-level** to avoid leakage.

---

## 3. Train & evaluate Baseline

```bash
python -m src.train_baseline --data_dir data --epochs 8 --batch_size 32
python -m src.evaluate --data_dir data --model baseline --ckpt checkpoints/baseline.pt
```

CPU quick run:
```bash
python -m src.train_baseline --data_dir data --epochs 2 --batch_size 16 --max_train_samples 400
```

---

## 4. Evaluate BLIP (recommended on CPU)

Zero-shot (no finetune):
```bash
python -m src.evaluate --data_dir data --model blip --blip_mode zeroshot
```

Optional: finetune (slow on CPU; use small sample)
```bash
python -m src.train_blip --data_dir data --epochs 1 --batch_size 2 --max_train_samples 200
python -m src.evaluate --data_dir data --model blip --blip_mode finetuned --ckpt checkpoints/blip.pt
```

---

## 5. Outputs

Evaluation prints and saves JSON:
- `runs/results_*.json`

Metrics are separated for:
- Closed Yes/No Accuracy
- Open EM / BLEU / ROUGE-L
