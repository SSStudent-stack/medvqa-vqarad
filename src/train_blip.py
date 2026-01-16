import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from torch.optim import AdamW


from .utils.seed import set_seed
from .utils.io import read_json, write_json
from .utils.device import get_device

def collate_blip(processor: BlipProcessor, batch):
    images, questions, answers, metas = [], [], [], []
    for img, q, a, meta in batch:
        images.append(img)
        questions.append(q)
        answers.append(a)
        metas.append(meta)

    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True)
    labels = processor(text=answers, return_tensors="pt", padding=True).input_ids
    inputs["labels"] = labels
    return inputs, metas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", default="Salesforce/blip-vqa-base")
    ap.add_argument("--max_train_samples", type=int, default=200)
    ap.add_argument("--max_val_samples", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_items = read_json(str(data_dir / "splits/train.json"))[:args.max_train_samples]
    val_items   = read_json(str(data_dir / "splits/val.json"))[:args.max_val_samples]
    images_dir  = data_dir / "raw/images"

    class DS(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            d = self.items[i]
            img = Image.open(images_dir / d["image_name"]).convert("RGB")
            return img, d.get("question",""), str(d.get("answer","")), {"answer_type": d.get("answer_type","")}

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForQuestionAnswering.from_pretrained(args.model_name).to(device)  # ✅ 用 device

    train_loader = DataLoader(
        DS(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_blip(processor, b),
    )
    val_loader = DataLoader(
        DS(val_items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_blip(processor, b),
    )

    opt = AdamW(model.parameters(), lr=args.lr)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    history = []
    best = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for inputs, _ in tqdm(train_loader, desc=f"train {epoch}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader, desc=f"val {epoch}"):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(**inputs)
                va_loss += float(out.loss.item())
        va_loss /= max(1, len(val_loader))

        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "device": device}
        history.append(rec)
        print(rec)

        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), "checkpoints/blip.pt")

    write_json(history, "runs/blip_history.json")
    print("Saved best BLIP to checkpoints/blip.pt")

if __name__ == "__main__":
    main()
