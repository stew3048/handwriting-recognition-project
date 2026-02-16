"""
讀取指定資料夾內所有 .npy，輸出可讀的報告檔（.txt）。
用法：python scripts/view_npy_report.py --dir outputs/emnist36_eval_centerloss_64
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.datasets import EMNIST36_LABEL_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="outputs/emnist36_eval_centerloss_64", help="放 npy 的資料夾")
    args = p.parse_args()
    dir_path = Path(args.dir)
    if not dir_path.is_absolute():
        dir_path = ROOT / dir_path
    out_txt = dir_path / "npy內容報告.txt"
    lines = []
    lines.append("=" * 60)
    lines.append("NPY 檔案內容報告（可直接用記事本開此檔）")
    lines.append("=" * 60)

    # 2x2 易混字對的標籤對應（名稱 -> (row0, row1), (col0, col1)）
    pair_labels = {
        "O_0": (("O", "0"), ("O", "0")),
        "I_1": (("I", "1"), ("I", "1")),
        "Z_2": (("Z", "2"), ("Z", "2")),
        "S_5": (("S", "5"), ("S", "5")),
        "A_4": (("A", "4"), ("A", "4")),
        "G_6": (("G", "6"), ("G", "6")),
        "B_8": (("B", "8"), ("B", "8")),
    }

    for f in sorted(dir_path.glob("*.npy")):
        arr = np.load(f)
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"檔案: {f.name}")
        lines.append(f"形狀: {arr.shape}")
        lines.append("-" * 60)
        if "confusion_2x2" in f.name:
            # 2x2：列=真實類別，行=預測類別
            name = f.stem.replace("confusion_2x2_", "")
            if name in pair_labels:
                rlab, clab = pair_labels[name]
                lines.append(f"列（真實）: {rlab[0]}, {rlab[1]}")
                lines.append(f"行（預測）: {clab[0]}, {clab[1]}")
                lines.append("")
                lines.append(f"              預測{rlab[0]:>4}  預測{rlab[1]:>4}")
                lines.append(f"真實 {rlab[0]}    {int(arr[0,0]):6}   {int(arr[0,1]):6}")
                lines.append(f"真實 {rlab[1]}    {int(arr[1,0]):6}   {int(arr[1,1]):6}")
            else:
                lines.append(np.array2string(arr, prefix="  "))
        elif "confusion_matrix" in f.name and arr.ndim == 2 and arr.shape[0] == 36:
            lines.append("36x36 混淆矩陣（列=真實類別，行=預測類別）")
            lines.append("類別對應: 0-9 為數字 0-9，10-35 為 A-Z")
            # 只印前幾行或摘要，否則太長
            lines.append("")
            lines.append("左上 10x10（數字 0-9）：")
            lines.append(np.array2string(arr[:10, :10].astype(int), prefix="  "))
            lines.append("")
            lines.append("對角線（各類正確數）前 12 個: " + str([int(arr[i,i]) for i in range(12)]))
        else:
            lines.append(np.array2string(arr, prefix="  "))

    lines.append("")
    lines.append("=" * 60)
    text = "\n".join(lines)
    out_txt.write_text(text, encoding="utf-8")
    print(f"已寫入: {out_txt}")
    print(text)


if __name__ == "__main__":
    main()
