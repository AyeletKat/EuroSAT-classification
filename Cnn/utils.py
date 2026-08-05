import random, os, numpy as np, torch

__all__ = ["set_seed", "accuracy"]

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a mini-batch (returns value in [0,1])."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()