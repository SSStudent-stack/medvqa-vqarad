import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .datasets.vqarad import VQARADDataset
from .models.baseline_cnn_rnn import BaselineVQA
from .utils.text import build_vocab, normalize_text
from .utils.seed import set_seed
from .utils.io import read_json, write_json
from .utils.device import get_device

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_open_answer_vocab(train_items: List[Dict[str, Any]]) -> Dict[str, int]:
    answers = []
    for d in train_items:
        if normalize_text(d.get("answer_type", "")).startswith("open"):
            a = normalize_text(d.get("answer", ""))
            if a:
                answers.append(a)
    uniq = sorted(set(answers))
    return {a:i for i,a in enumerate(uniq)}

def collate_baseline(batch):
    imgs, qids, ys, metas, tasks = [], [], [], [], []
    for img, q_ids, y, meta in batch:
        imgs.append(img)
        qids.append(q_ids)
        ys.append(y)
        metas.append(meta)
        tasks.append(meta["task"])
    return torch.stack(imgs), torch.stack(qids), torch.stack(ys), {"tasks": tasks, "metas": metas}

def set_trainable_for_stage(model: BaselineVQA, stage: int, unfreeze_layer4: bool):
    """
    stage 1: freeze cnn; train question+fusion+closed_head
    stage 2: train everything except maybe early cnn; optionally unfreeze layer4
    """
    # default: freeze all cnn
    for p in model.cnn.parameters():
        p.requires_grad = False

    if stage == 2 and unfreeze_layer4:
        # model.cnn is Sequential of resnet children without fc.
        # In resnet18 children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        # Here, model.cnn includes up to avgpool, so we unfreeze layer4 if it exists.
        # model.cnn[7] usually corresponds to layer4 for resnet18.
        try:
            layer4 = model.cnn[7]
            for p in layer4.parameters():
                p.requires_grad = True
        except Exception:
            # fallback: unfreeze last module
            for p in list(model.cnn.parameters())[-1:]:
                p.requires_grad = True

    # question & heads always train
    for p in model.emb.parameters():
        p.requires_grad = True
    for p in model.gru.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.closed_head.parameters():
        p.requires_grad = True
    for p in model.open_head.parameters():
        p.requires_grad = (stage == 2)

def make_optimizer(model: BaselineVQA, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)

def train_one_epoch(model, loader, optimizer, device, stage: int, open_loss_weight: float = 1.0):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_closed = correct_closed = 0
    total_open = correct_open = 0
    loss_sum = 0.0

    for img, q_ids, y, meta in tqdm(loader, desc=f"train(stage={stage})", leave=False):
        img = img.to(device)
        q_ids = q_ids.to(device)
        y = y.to(device)

        logits_closed, logits_open = model(img, q_ids)
        tasks = meta["tasks"]

        closed_idx = [i for i,t in enumerate(tasks) if t == "closed"]
        open_idx = [i for i,t in enumerate(tasks) if t == "open" and int(y[i].item()) >= 0]

        loss = None

        if closed_idx:
            y_c = y[closed_idx]
            loss_c = ce(logits_closed[closed_idx], y_c)
            loss = loss_c if loss is None else loss + loss_c
            pred_c = logits_closed[closed_idx].argmax(dim=1)
            correct_closed += int((pred_c == y_c).sum().item())
            total_closed += len(closed_idx)

        if stage == 2 and open_idx:
            y_o = y[open_idx]
            loss_o = ce(logits_open[open_idx], y_o) * open_loss_weight
            loss = loss_o if loss is None else loss + loss_o
            pred_o = logits_open[open_idx].argmax(dim=1)
            correct_open += int((pred_o == y_o).sum().item())
            total_open += len(open_idx)

        if loss is None:
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())

    return {
        "loss": loss_sum / max(1, len(loader)),
        "closed_acc": correct_closed / total_closed if total_closed else 0.0,
        "open_acc": correct_open / total_open if total_open else 0.0,
        "closed_n": total_closed,
        "open_n": total_open,
    }

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_closed = correct_closed = 0
    total_open = correct_open = 0

    for img, q_ids, y, meta in tqdm(loader, desc="val", leave=False):
        img = img.to(device); q_ids = q_ids.to(device); y = y.to(device)
        logits_closed, logits_open = model(img, q_ids)
        tasks = meta["tasks"]

        closed_idx = [i for i,t in enumerate(tasks) if t == "closed"]
        open_idx = [i for i,t in enumerate(tasks) if t == "open" and int(y[i].item()) >= 0]

        if closed_idx:
            y_c = y[closed_idx]
            pred_c = logits_closed[closed_idx].argmax(dim=1)
            correct_closed += int((pred_c == y_c).sum().item())
            total_closed += len(closed_idx)

        if open_idx:
            y_o = y[open_idx]
            pred_o = logits_open[open_idx].argmax(dim=1)
            correct_open += int((pred_o == y_o).sum().item())
            total_open += len(open_idx)

    return {
        "closed_acc": correct_closed / total_closed if total_closed else 0.0,
        "open_acc": correct_open / total_open if total_open else 0.0,
        "closed_n": total_closed,
        "open_n": total_open,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--stage1_epochs", type=int, default=4)
    ap.add_argument("--stage2_epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--open_loss_weight", type=float, default=1.0)

    ap.add_argument("--device", default="auto", choices=["auto","cpu","mps"])
    ap.add_argument("--unfreeze_layer4", action="store_true", help="Stage2 unfreeze resnet layer4 (recommended)")

    ap.add_argument("--max_q_len", type=int, default=32)
    ap.add_argument("--max_train_samples", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_items = read_json(str(data_dir/"splits/train.json"))
    val_items   = read_json(str(data_dir/"splits/val.json"))

    if args.max_train_samples and args.max_train_samples > 0:
        train_items = train_items[:args.max_train_samples]

    vocab = build_vocab([d.get("question","") for d in train_items], min_freq=1)
    open_answer2id = build_open_answer_vocab(train_items)

    Path("checkpoints").mkdir(exist_ok=True)
    write_json({"vocab": vocab, "open_answer2id": open_answer2id}, "checkpoints/baseline_maps.json")

    images_dir = str(data_dir/"raw/images")

    # A) transforms: normalize + light augmentation
    train_tfm = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.9, 1.0)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # datasets
    ds_closed = VQARADDataset(train_items, images_dir, "baseline_closed", vocab=vocab, max_q_len=args.max_q_len,
                             open_answer2id=open_answer2id, transform=train_tfm)
    ds_open   = VQARADDataset(train_items, images_dir, "baseline_open", vocab=vocab, max_q_len=args.max_q_len,
                             open_answer2id=open_answer2id, transform=train_tfm)

    class Mix(torch.utils.data.Dataset):
        def __init__(self, a, b): self.a=a; self.b=b
        def __len__(self): return len(self.a)+len(self.b)
        def __getitem__(self, i): return self.a[i] if i < len(self.a) else self.b[i-len(self.a)]

    train_mixed = Mix(ds_closed, ds_open)

    val_ds_closed = VQARADDataset(val_items, images_dir, "baseline_closed", vocab=vocab, max_q_len=args.max_q_len,
                                 open_answer2id=open_answer2id, transform=eval_tfm)
    val_ds_open   = VQARADDataset(val_items, images_dir, "baseline_open", vocab=vocab, max_q_len=args.max_q_len,
                                 open_answer2id=open_answer2id, transform=eval_tfm)
    val_mixed = Mix(val_ds_closed, val_ds_open)

    # loaders

    from torch.utils.data import WeightedRandomSampler

    # ----- Balanced sampler for closed yes/no -----
    closed_labels = []
    for i in range(len(ds_closed)):
        # ds_closed[i] -> (img, q_ids, y, meta)
        _, _, y, _ = ds_closed[i]
        closed_labels.append(int(y.item()))

    # count
    n_yes = sum(closed_labels)
    n_no = len(closed_labels) - n_yes
    # weights: inverse frequency
    w_yes = 1.0 / max(1, n_yes)
    w_no = 1.0 / max(1, n_no)
    sample_weights = [w_yes if lab == 1 else w_no for lab in closed_labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    closed_loader = DataLoader(
        ds_closed,
        batch_size=args.batch_size,
        sampler=sampler,  # <= sampler and shuffle are mutually exclusive
        shuffle=False,
        num_workers=0,
        collate_fn=collate_baseline
    )

    mixed_loader  = DataLoader(train_mixed, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_baseline)
    val_loader    = DataLoader(val_mixed, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_baseline)

    model = BaselineVQA(vocab_size=len(vocab), open_num_classes=len(open_answer2id)).to(device)

    Path("runs").mkdir(exist_ok=True)
    history = []
    best = -1.0

    # B) Stage 1: closed only
    set_trainable_for_stage(model, stage=1, unfreeze_layer4=False)
    opt = make_optimizer(model, args.lr)
    for epoch in range(1, args.stage1_epochs+1):
        tr = train_one_epoch(model, closed_loader, opt, device, stage=1)
        va = eval_epoch(model, val_loader, device)
        rec = {"stage": 1, "epoch": epoch, **tr, **va}
        history.append(rec)
        print(rec)
        score = va["closed_acc"] + va["open_acc"]
        if score > best:
            best = score
            torch.save(model.state_dict(), "checkpoints/baseline.pt")

    # Stage 2: mixed
    set_trainable_for_stage(model, stage=2, unfreeze_layer4=args.unfreeze_layer4)
    opt = make_optimizer(model, args.lr * 0.5)  # often helps stability
    for epoch in range(1, args.stage2_epochs+1):
        tr = train_one_epoch(model, mixed_loader, opt, device, stage=2, open_loss_weight=args.open_loss_weight)
        va = eval_epoch(model, val_loader, device)
        rec = {"stage": 2, "epoch": epoch, **tr, **va}
        history.append(rec)
        print(rec)
        score = va["closed_acc"] + va["open_acc"]
        if score > best:
            best = score
            torch.save(model.state_dict(), "checkpoints/baseline.pt")

    write_json(history, "runs/baseline_history_ab.json")
    print("Saved best baseline to checkpoints/baseline.pt")

if __name__ == "__main__":
    main()
