import re
from difflib import SequenceMatcher

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision.transforms as T
from PIL import Image

from .utils.device import get_device
from .utils.io import read_json, write_json
from .utils.text import normalize_text, normalize_yesno
from .utils.metrics import accuracy, bleu_scores, rouge_l
from .datasets.vqarad import VQARADDataset
from .models.baseline_cnn_rnn import BaselineVQA
from .models.blip_vqa import load_blip, blip_generate

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _norm_for_match(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_to_answer_space(pred: str, answer_list, min_score: float = 0.25) -> str:
    """
    Map free-form pred to closest answer in answer_list using SequenceMatcher.
    If the best similarity is too low, return empty string.
    """
    p = _norm_for_match(pred)
    if not p:
        return ""

    best_a = ""
    best_s = -1.0
    for a in answer_list:
        a_n = _norm_for_match(a)
        if not a_n:
            continue
        s = SequenceMatcher(None, p, a_n).ratio()
        if s > best_s:
            best_s = s
            best_a = a

    return best_a if best_s >= min_score else ""


def eval_baseline(data_dir: Path, split: str, ckpt: str, device: str = "cpu"):
    items = read_json(str(data_dir / f"splits/{split}.json"))
    maps = read_json("checkpoints/baseline_maps.json")
    vocab = maps["vocab"]
    open_answer2id = maps["open_answer2id"]
    id2open = {int(v): k for k, v in open_answer2id.items()}

    images_dir = str(data_dir / "raw/images")

    # Use eval-time transform consistent with train_baseline (A: normalize)
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ds_c = VQARADDataset(
        items, images_dir, "baseline_closed",
        vocab=vocab, open_answer2id=open_answer2id, transform=tfm
    )
    ds_o = VQARADDataset(
        items, images_dir, "baseline_open",
        vocab=vocab, open_answer2id=open_answer2id, transform=tfm
    )

    model = BaselineVQA(vocab_size=len(vocab), open_num_classes=len(open_answer2id))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    closed_pairs = []
    open_preds, open_refs = [], []

    with torch.no_grad():
        for i in tqdm(range(len(ds_c)), desc="baseline closed"):
            img, q_ids, y, meta = ds_c[i]
            logits_c, _ = model(img.unsqueeze(0).to(device), q_ids.unsqueeze(0).to(device))
            p = int(logits_c.argmax(dim=1).item())
            pred = "yes" if p == 1 else "no"
            closed_pairs.append((pred, meta["answer"]))

        for i in tqdm(range(len(ds_o)), desc="baseline open"):
            img, q_ids, y, meta = ds_o[i]
            _, logits_o = model(img.unsqueeze(0).to(device), q_ids.unsqueeze(0).to(device))
            p = int(logits_o.argmax(dim=1).item())
            pred = id2open.get(p, "")
            open_preds.append(pred)
            open_refs.append(meta["answer"])

    closed_acc = accuracy(closed_pairs)
    open_em = accuracy(list(zip(open_preds, open_refs)))
    bleu = bleu_scores(open_preds, open_refs)
    rl = rouge_l(open_preds, open_refs)

    return {
        "model": "baseline",
        "split": split,
        "device": device,
        "closed_yesno_acc": closed_acc,
        "open_em": open_em,
        "open_bleu1": bleu["bleu1"],
        "open_bleu2": bleu["bleu2"],
        "open_bleu4": bleu["bleu4"],
        "open_rougeL": rl,
        "n_closed": len(closed_pairs),
        "n_open": len(open_preds),
    }

def eval_blip(data_dir: Path, split: str, mode: str, ckpt: str = "", device: str = "cpu"):
    items = read_json(str(data_dir / f"splits/{split}.json"))
    images_dir = data_dir / "raw/images"

    # Load baseline open answer space for fair mapping evaluation
    maps = read_json("checkpoints/baseline_maps.json")
    open_answer2id = maps.get("open_answer2id", {})
    answer_space = list(open_answer2id.keys())  # list of open answers

    processor, model = load_blip(device=device)

    if mode == "finetuned":
        if not ckpt:
            raise ValueError("finetuned mode requires --ckpt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)

    model.eval()

    closed_pairs = []

    open_preds, open_refs = [], []
    open_preds_mapped = []

    bs = 4
    for i in tqdm(range(0, len(items), bs), desc=f"blip {mode}"):
        batch = items[i:i + bs]
        images = [Image.open(images_dir / d["image_name"]).convert("RGB") for d in batch]
        questions = [d.get("question", "") for d in batch]
        preds = blip_generate(processor, model, images, questions, device=device, max_new_tokens=10)

        for d, pred in zip(batch, preds):
            a_type = normalize_text(d.get("answer_type", ""))
            gold = str(d.get("answer", ""))

            if a_type.startswith("closed"):
                g = normalize_yesno(gold)
                if g in ("yes", "no"):
                    closed_pairs.append((normalize_yesno(pred), g))


            elif a_type.startswith("open"):
                open_preds.append(pred)
                open_refs.append(gold)

                # mapped prediction into baseline answer space (fairer comparison)
                mapped = map_to_answer_space(pred, answer_space, min_score=0.25)
                open_preds_mapped.append(mapped)

    closed_acc = accuracy(closed_pairs)
    open_em = accuracy(list(zip(open_preds, open_refs)))
    bleu = bleu_scores(open_preds, open_refs)
    rl = rouge_l(open_preds, open_refs)

    open_em_mapped = accuracy(list(zip(open_preds_mapped, open_refs)))
    bleu_mapped = bleu_scores(open_preds_mapped, open_refs)
    rl_mapped = rouge_l(open_preds_mapped, open_refs)

    return {
        "model": f"blip_{mode}",
        "split": split,
        "device": device,
        "closed_yesno_acc": closed_acc,
        "open_em": open_em,
        "open_bleu1": bleu["bleu1"],
        "open_bleu2": bleu["bleu2"],
        "open_bleu4": bleu["bleu4"],
        "open_rougeL": rl,
        "n_closed": len(closed_pairs),
        "n_open": len(open_preds),

        "open_em_mapped": open_em_mapped,
        "open_bleu1_mapped": bleu_mapped["bleu1"],
        "open_bleu2_mapped": bleu_mapped["bleu2"],
        "open_bleu4_mapped": bleu_mapped["bleu4"],
        "open_rougeL_mapped": rl_mapped,

    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", required=True, choices=["baseline", "blip"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--blip_mode", default="zeroshot", choices=["zeroshot", "finetuned"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    Path("runs").mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.model == "baseline":
        if not args.ckpt:
            raise ValueError("baseline requires --ckpt")
        res = eval_baseline(data_dir, args.split, args.ckpt, device=device)
        out = Path("runs") / f"results_baseline_{args.split}.json"
    else:
        res = eval_blip(data_dir, args.split, args.blip_mode, ckpt=args.ckpt, device=device)
        out = Path("runs") / f"results_blip_{args.blip_mode}_{args.split}.json"

    write_json(res, str(out))
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
