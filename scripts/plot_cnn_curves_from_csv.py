"""One-off: plot train curves from experiments_cnn.csv and save to same run dir."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import csv
from scripts.train_mnist_cnn import plot_curves, judge_overfitting

run_dir = ROOT / "outputs" / "runs" / "20260208_155516_mnist_cnn"
csv_path = run_dir / "experiments_cnn.csv"
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        history["train_loss"].append(float(row["train_loss"]))
        history["train_acc"].append(float(row["train_acc"]))
        history["val_loss"].append(float(row["val_loss"]))
        history["val_acc"].append(float(row["val_acc"]))

msg = judge_overfitting(history)
plot_curves(history, run_dir / "train_curves.png", overfitting_msg=msg)
print("Overfitting:", msg)
print("Saved", run_dir / "train_curves.png")
