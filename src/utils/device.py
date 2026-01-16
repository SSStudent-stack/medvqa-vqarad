import torch

def get_device(requested: str = "auto") -> str:
    """
    requested: auto | cpu | mps
    """
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    # auto
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
