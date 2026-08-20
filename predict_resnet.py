"""
predict_resnet.py
=================
Classify 64x64 EuroSAT-style RGB tiles with the fine-tuned ResNet18 from this
project (resnet_eurosat.pth).

The preprocessing here reproduces exactly the transform used when the network
was fine-tuned in models_code.py:

    transforms.Resize((64, 64))
    transforms.ToTensor()
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])

and the class order is the alphabetical folder order that
torchvision.datasets.ImageFolder produced during training. Both must match or
the predictions are meaningless, so neither should be changed.

Usage
-----
    python predict_resnet.py tile.png
    python predict_resnet.py tile1.png tile2.png --plot probs.png
    python predict_resnet.py "C:\\path\\to\\tiles\\*.png"
    python predict_resnet.py tile.png --weights "C:\\...\\resnet_eurosat.pth"

Run with no image arguments to classify every .png/.jpg in the current folder.
"""

import argparse
import glob
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# --------------------------------------------------------------------------
# Defaults -- edit these two lines if you move the files
# --------------------------------------------------------------------------
DEFAULT_WEIGHTS = r"C:\Users\ayele\Documents\ML_2025\ML_project\ML_Final_Project\resnet_eurosat.pth"

# ImageFolder sorts class directories alphabetically; this is that order.
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Identical to the fine-tuning transform in models_code.py
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------
def build_model(weights_path, device):
    """Rebuild the fine-tuned architecture and load the saved state_dict."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        sys.exit(f"Weights file not found:\n  {weights_path}")

    # No ImageNet download: the saved state_dict supplies every weight.
    try:
        model = resnet18(weights=None)            # torchvision >= 0.13
    except TypeError:
        model = resnet18(pretrained=False)        # older torchvision

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)

    # tolerate a checkpoint dict or a DataParallel prefix
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as err:
        sys.exit(
            "The checkpoint does not match a resnet18 with a 10-class head.\n"
            f"torch said:\n{err}"
        )

    return model.to(device).eval()


@torch.no_grad()
def predict(model, paths, device):
    """Return (probabilities [N, 10] tensor, list of PIL images actually used)."""
    images = [Image.open(p).convert("RGB") for p in paths]
    batch = torch.stack([TRANSFORM(im) for im in images]).to(device)
    return model(batch).softmax(dim=1).cpu(), images


def report(paths, probs, topk=3):
    for path, row in zip(paths, probs):
        conf, idx = row.sort(descending=True)
        print(f"\n{Path(path).name}")
        print(f"  -> {CLASSES[idx[0]]}  ({conf[0] * 100:.1f}%)")
        print(f"  top-{topk}:")
        for c, i in zip(conf[:topk], idx[:topk]):
            bar = "#" * int(round(c.item() * 40))
            print(f"    {CLASSES[i]:<22} {c * 100:5.1f}%  {bar}")


def save_plot(paths, probs, out_path):
    """Horizontal probability bars -- ready to drop into the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(paths)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.9 * n), squeeze=False)
    y = np.arange(len(CLASSES))
    for ax, path, row in zip(axes[:, 0], paths, probs):
        top = int(row.argmax())
        colours = ["#c8cdd6"] * len(CLASSES)
        colours[top] = "#2f6f4f"
        ax.barh(y, row.numpy() * 100, color=colours)
        ax.set_yticks(y)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("predicted probability (%)", fontsize=8)
        ax.set_title(f"{Path(path).name}  ->  {CLASSES[top]} "
                     f"({row[top] * 100:.1f}%)", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"\nwrote {out_path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*",
                    help="image files or globs; default = all images in the folder")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="path to resnet_eurosat.pth")
    ap.add_argument("--plot", metavar="OUT.png",
                    help="also save a probability bar chart")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    patterns = args.images or ["*.png", "*.jpg", "*.jpeg", "*.tif"]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)} |
                   {p for p in args.images if Path(p).is_file()})
    if not paths:
        sys.exit("No images found. Pass a filename, or run inside the folder "
                 "containing the tiles.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"weights: {args.weights}")

    model = build_model(args.weights, device)
    probs, images = predict(model, paths, device)

    for path, im in zip(paths, images):
        if im.size != (64, 64):
            print(f"note: {Path(path).name} is {im.size[0]}x{im.size[1]}, "
                  f"resized to 64x64 (same as training).")

    report(paths, probs, args.topk)
    if args.plot:
        save_plot(paths, probs, args.plot)


if __name__ == "__main__":
    main()