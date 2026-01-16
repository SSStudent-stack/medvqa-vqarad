import argparse
import random
import zipfile
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from .utils.seed import set_seed
from .utils.io import write_json, read_json

def extract_images(zip_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        jpgs = [n for n in names if n.lower().endswith(".jpg") and "__macosx" not in n.lower()]
        for n in tqdm(jpgs, desc="Extracting images"):
            filename = Path(n).name
            target = out / filename
            if target.exists():
                continue
            with z.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())

def image_level_split(items: List[Dict], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1):
    imgs = sorted({d["image_name"] for d in items})
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_imgs = set(imgs[:n_train])
    val_imgs = set(imgs[n_train:n_train+n_val])
    test_imgs = set(imgs[n_train+n_val:])
    train = [d for d in items if d["image_name"] in train_imgs]
    val   = [d for d in items if d["image_name"] in val_imgs]
    test  = [d for d in items if d["image_name"] in test_imgs]
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True)
    ap.add_argument("--image_zip", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    splits_dir = out_dir / "splits"
    images_dir = raw_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < 10:
        extract_images(args.image_zip, str(images_dir))

    items = read_json(args.json_path)
    train, val, test = image_level_split(items, seed=args.seed)

    write_json(train, splits_dir / "train.json")
    write_json(val,   splits_dir / "val.json")
    write_json(test,  splits_dir / "test.json")

    print("Done.")
    print(f"Unique images: {len(set(d['image_name'] for d in items))}")
    print(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")

if __name__ == "__main__":
    main()
