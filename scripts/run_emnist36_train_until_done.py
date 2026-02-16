"""
自動跑完 EMNIST 36-class CNN 訓練：最多 10 epochs，用 CPU。
若 OOM 或崩潰會自動換小 batch_size (64 -> 32 -> 16) 重跑，直到 10 epochs 完成。
從專案根目錄執行: python scripts/run_emnist36_train_until_done.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "runs"
STATUS_FILE = ROOT / "outputs" / "emnist36_train_status.txt"
TARGET_EPOCHS = 10
# MNIST CNN 預設 64；EMNIST 36 資料量約 6 倍，先試 10，再依序縮小
BATCH_SIZES = [10, 8, 4]


def write_status(msg):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(msg.strip() + "\n", encoding="utf-8")


def get_latest_emnist36_run():
    """回傳最新的 *_emnist36_cnn 目錄（依修改時間）。"""
    if not RUNS_DIR.exists():
        return None
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and "_emnist36_cnn" in d.name]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def count_epochs(run_dir):
    """experiments.csv 中資料行數（不含 header）= 已完成 epoch 數。"""
    csv_path = run_dir / "experiments.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)  # minus header


def main():
    for batch_size in BATCH_SIZES:
        write_status(f"runner: trying batch_size={batch_size} (CPU, {TARGET_EPOCHS} epochs)")
        print(f"\n>>> Trying batch_size={batch_size} (CPU, {TARGET_EPOCHS} epochs)", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_emnist_digits_letters.py"),
            "--model", "cnn",
            "--epochs", str(TARGET_EPOCHS),
            "--save_best",
            "--batch_size", str(batch_size),
            "--cpu",
        ]
        ret = subprocess.run(cmd, cwd=str(ROOT))
        run_dir = get_latest_emnist36_run()
        n = count_epochs(run_dir) if run_dir else 0
        print(f">>> Completed {n}/{TARGET_EPOCHS} epochs. Run dir: {run_dir}", flush=True)
        if n >= TARGET_EPOCHS and ret.returncode == 0:
            write_status(f"done: {n} epochs, run_dir={run_dir}")
            print(f">>> Done. Checkpoints in: {run_dir}", flush=True)
            return 0
        if ret.returncode != 0:
            write_status(f"runner: batch_size={batch_size} failed (exit {ret.returncode}), trying smaller batch_size.")
            print(f">>> Training exited with code {ret.returncode}, trying smaller batch_size.", flush=True)
    write_status("runner: all batch sizes tried; did not complete 10 epochs.")
    print(">>> All batch sizes tried; run did not complete 10 epochs.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
