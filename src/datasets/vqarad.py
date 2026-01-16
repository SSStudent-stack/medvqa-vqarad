from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from ..utils.text import simple_tokenize, encode, normalize_text, normalize_yesno

@dataclass
class Sample:
    image_path: str
    question: str
    answer: str
    answer_type: str
    image_name: str

class VQARADDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        images_dir: str,
        mode: str,
        vocab: Optional[Dict[str, int]] = None,
        max_q_len: int = 32,
        open_answer2id: Optional[Dict[str, int]] = None,
        transform=None,
    ):
        self.images_dir = images_dir
        self.mode = mode
        self.vocab = vocab
        self.max_q_len = max_q_len
        self.open_answer2id = open_answer2id
        self.transform = transform

        self.samples: List[Sample] = []
        for d in items:
            a_type = normalize_text(d.get("answer_type", ""))
            a_type = "CLOSED" if a_type.startswith("closed") else ("OPEN" if a_type.startswith("open") else d.get("answer_type",""))

            img_name = d["image_name"]
            img_path = str(Path(images_dir) / img_name)
            q = d.get("question", "")
            a = d.get("answer", "")

            if mode == "baseline_closed":
                a2 = normalize_yesno(a)
                if a2 not in ("yes", "no"):
                    continue
                self.samples.append(Sample(img_path, q, a2, "CLOSED", img_name))

            elif mode == "baseline_open":
                if a_type != "OPEN":
                    continue
                a2 = normalize_text(a)
                self.samples.append(Sample(img_path, q, a2, "OPEN", img_name))

            elif mode == "blip":
                self.samples.append(Sample(img_path, q, str(a), a_type, img_name))

            else:
                raise ValueError(f"Unknown mode: {mode}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img_t = self.transform(img) if self.transform is not None else img

        if self.mode.startswith("baseline"):
            assert self.vocab is not None
            q_ids = torch.tensor(encode(simple_tokenize(s.question), self.vocab, self.max_q_len), dtype=torch.long)

            if self.mode == "baseline_closed":
                y = torch.tensor(1 if s.answer == "yes" else 0, dtype=torch.long)
                return img_t, q_ids, y, {"answer": s.answer, "question": s.question, "image_name": s.image_name, "task": "closed"}

            # baseline_open
            y_id = -1
            if self.open_answer2id is not None:
                y_id = self.open_answer2id.get(s.answer, -1)
            return img_t, q_ids, torch.tensor(y_id, dtype=torch.long), {"answer": s.answer, "question": s.question, "image_name": s.image_name, "task": "open"}

        # blip
        return img_t, s.question, s.answer, {"answer_type": s.answer_type, "image_name": s.image_name}
