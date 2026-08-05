import itertools, subprocess, pathlib, sys

learning_rates = [1e-3, 5e-4, 1e-4]
batch_sizes    = [32, 64]
batch_norms    = [False, True]

epochs, patience = 50, 5
train_py = pathlib.Path("Cnn/train.py").as_posix()

def launch(lr, bs, bn):
    name = f"bs{bs}_lr{lr:.0e}_{'BN' if bn else 'NoBN'}"
    cmd = [
        sys.executable, train_py,
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(lr),
        "--batch", str(bs),
        "--group", "grid",
        "--run_name", name,
    ]
    if bn:
        cmd.append("--bn")
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_default():
    """Run train.py with default parameters"""
    cmd = [sys.executable, train_py]
    print("▶ Running with default parameters:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # Run with default parameters only what you run if you start write python train.py without changing nothing
    run_default()
    
    for lr, bs, bn in itertools.product(learning_rates, batch_sizes, batch_norms):
        launch(lr, bs, bn)

if __name__ == "__main__":
    main()