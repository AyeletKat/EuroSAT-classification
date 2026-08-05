import argparse, os, random, csv, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EuroSAT
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

# ---------------- reproducibility -----------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- directories ---------------------------------------
# Base directory is the directory containing this script
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CONF_MATS_DIR = BASE_DIR / "conf_mats"
CSV_PATH = BASE_DIR / "experiment_log.csv"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(CONF_MATS_DIR, exist_ok=True)

# ---------------- model ---------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10, use_bn: bool = False):
        super().__init__()
        def block(c_in, c_out, layers):
            layers.append(nn.Conv2d(c_in, c_out, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))

        feats = []
        block(3,   32, feats)   # 64→32
        block(32,  64, feats)   # 32→16
        block(64, 128, feats)   # 16→8
        self.features = nn.Sequential(*feats)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------- data ----------------------------------------------
def make_loaders(batch, workers=4):
    tf_eval  = transforms.ToTensor()
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    full = EuroSAT(root=DATA_DIR, download=True, transform=tf_eval)
    n_tr, n_val = int(0.70*len(full)), int(0.15*len(full))
    n_te = len(full) - n_tr - n_val
    tr, va, te = random_split(full, [n_tr, n_val, n_te],
                              generator=torch.Generator().manual_seed(SEED))
    tr.dataset.transform = tf_train
    return (
        DataLoader(tr, batch, shuffle=True,  num_workers=workers),
        DataLoader(va, batch, shuffle=False, num_workers=workers),
        DataLoader(te, batch, shuffle=False, num_workers=workers),
        full.classes                                   # ← class names
    )

# ---------------- helpers -------------------------------------------
def evaluate(model, loader, criterion, device, want_preds=False):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    all_y, all_hat = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            preds  = logits.argmax(1)
            loss_sum += loss.item() * xb.size(0)
            correct  += (preds == yb).sum().item()
            total    += xb.size(0)
            if want_preds:
                all_y.append(yb.cpu())
                all_hat.append(preds.cpu())
    if want_preds:
        return loss_sum / total, correct / total, torch.cat(all_y), torch.cat(all_hat)
    return loss_sum / total, correct / total

def save_conf_mat_png(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def append_csv(row, path=CSV_PATH):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["run_name", "lr", "batch", "bn", "test_loss", "test_acc"])
        writer.writerow(row)

# ---------------- main ----------------------------------------------
def main():
    ap = argparse.ArgumentParser("EuroSAT CNN trainer")
    ap.add_argument("--epochs",   type=int,   default=50)
    ap.add_argument("--patience", type=int,   default=5)
    ap.add_argument("--batch",    type=int,   default=64)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--bn",       action="store_true", help="enable BatchNorm")
    ap.add_argument("--quick",    action="store_true", help="1-epoch smoke test")
    ap.add_argument("--project",  default="eurosat")
    ap.add_argument("--entity",   default=None)
    ap.add_argument("--group",    default="grid")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()
    if args.quick:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, va_dl, te_dl, class_names = make_loaders(args.batch)
    model = SimpleCNN(use_bn=args.bn).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.CrossEntropyLoss()

    run = wandb.init(project=args.project, entity=args.entity,
                     group=args.group, name=args.run_name,
                     config=vars(args), save_code=True)
    run_name = wandb.run.name  # guaranteed unique legend label
    wandb.watch(model, log="all", log_freq=100)

    # x-axis
    wandb.define_metric("epoch")
    for k in ["train/loss_epoch","train/acc_epoch","val/loss","val/acc",
              "grad_norm","lr","test/loss","test/acc"]:
        wandb.define_metric(k, step_metric="epoch")
    wandb.define_metric("train/loss_step", step_metric="step")
    wandb.define_metric("train/acc_step",  step_metric="step")

    best, stale, gstep = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        # ---------- train ----------
        model.train()
        tloss, tcorrect, ttotal = 0., 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()

            preds = out.argmax(1)
            tloss    += loss.item() * xb.size(0)
            tcorrect += (preds == yb).sum().item()
            ttotal   += xb.size(0)

            wandb.log({"step": gstep,
                       "train/loss_step": loss.item(),
                       "train/acc_step":  (preds == yb).float().mean().item(),
                       "lr": opt.param_groups[0]["lr"]},
                       commit=False)
            gstep += 1

        tr_loss = tloss / ttotal
        tr_acc  = tcorrect / ttotal

        # ---------- val ----------
        val_loss, val_acc = evaluate(model, va_dl, crit, device)

        # grad-norm
        with torch.no_grad():
            gn = sum(p.grad.detach().pow(2).sum().item()
                     for p in model.parameters() if p.grad is not None) ** 0.5

        wandb.log({"epoch": epoch,
                   "train/loss_epoch": tr_loss,
                   "train/acc_epoch":  tr_acc,
                   "val/loss": val_loss,
                   "val/acc": val_acc,
                   "grad_norm": gn})

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc*100:.2f}% "
              f"| val {val_loss:.4f}/{val_acc*100:.2f}%")

        # early-stop
        if val_loss < best:
            best, stale = val_loss, 0
            checkpoint_path = CHECKPOINTS_DIR / "best.pt"
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop (no improvement {args.patience} epochs)")
                break

    # ---------- test & confusion-matrix ----------
    te_loss, te_acc, y_true, y_pred = evaluate(model, te_dl, crit, device, want_preds=True)
    wandb.log({"epoch": epoch, "test/loss": te_loss, "test/acc": te_acc})

    title = f"{run_name} | bs={args.batch}, lr={args.lr:g}, bn={args.bn}"
    png_path = CONF_MATS_DIR / f"{run_name}.png"
    save_conf_mat_png(y_true, y_pred, class_names, title, png_path)
    wandb.save(str(png_path))  # upload image artefact

    append_csv([run_name, args.lr, args.batch, args.bn, te_loss, te_acc])

    print(f"TEST | loss {te_loss:.4f} | acc {te_acc*100:.2f}%  → saved {png_path}")
    run.finish()

if __name__ == "__main__":
    main()
