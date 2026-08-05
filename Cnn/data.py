from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EuroSAT
from torchvision import transforms

from utils import set_seed

IMG_SIZE = 64            # EuroSAT is 64×64 already
N_WORKERS = 4

# Base directory is the directory containing this script
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"

# Basic transforms -----------------------------------------------------------
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_tf = transforms.ToTensor()

def get_loaders(
    root: str | Path = DATA_DIR,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders with the requested split sizes."""
    set_seed(seed)

    # Ensure data directory exists
    if isinstance(root, str):
        root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    full_ds = EuroSAT(root=root, download=True, transform=val_tf)
    n_total = len(full_ds)
    n_val   = int(val_fraction  * n_total)
    n_test  = int(test_fraction * n_total)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    # Augmentations only for the training subset
    train_ds.dataset.transform = train_tf

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True)

    return train_dl, val_dl, test_dl