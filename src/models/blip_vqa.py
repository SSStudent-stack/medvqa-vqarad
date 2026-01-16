from typing import List
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

def load_blip(model_name: str = "Salesforce/blip-vqa-base", device: str = "cpu"):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    return processor, model

@torch.no_grad()
def blip_generate(processor, model, images, questions: List[str], device: str = "cpu", max_new_tokens: int = 10):
    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True).to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out_ids, skip_special_tokens=True)
