"""
產生 robustness 總結表格：三模型 × 7情境（clean + 6種corruption s3）的 accuracy 比較。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_BASE = ROOT / "outputs" / "robustness"

# 情境名稱（英文，避免字體問題）
SCENARIOS = {
    "clean": "Clean",
    "translation": "Translation",
    "rotation": "Rotation",
    "scale": "Scale",
    "shear": "Shear",
    "perspective": "Perspective",
    "elastic": "Elastic",
}

MODELS = ["mlp", "cnn", "resnet"]
MODEL_NAMES = {"mlp": "MLP", "cnn": "CNN", "resnet": "ResNet"}


def load_accuracy(model, scenario):
    """載入 accuracy。scenario='clean' 或 corruption type（取s3）。"""
    if scenario == "clean":
        path = OUT_BASE / model / "clean" / "metrics.json"
    else:
        path = OUT_BASE / model / scenario / "s3" / "metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["accuracy"]


def create_summary_table():
    """產生總結表格 PNG。"""
    # 收集數據
    data = {}
    for model in MODELS:
        data[model] = {}
        for scenario in SCENARIOS.keys():
            acc = load_accuracy(model, scenario)
            data[model][scenario] = acc

    # 建立表格數據
    table_data = []
    headers = ["情境"] + [MODEL_NAMES[m] for m in MODELS]
    
    for scenario in SCENARIOS.keys():
        row = [SCENARIOS[scenario]]
        for model in MODELS:
            acc = data[model][scenario]
            row.append(f"{acc*100:.2f}%")
        table_data.append(row)

    # 繪製表格
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    
    # 樣式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 標題行
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # 數據行：依 accuracy 著色（越高越綠）
    for i, scenario in enumerate(SCENARIOS.keys(), 1):
        for j, model in enumerate(MODELS, 1):
            acc = data[model][scenario]
            # 綠色漸層：0.0 (紅) -> 1.0 (綠)
            green = acc
            red = 1.0 - acc
            table[(i, j)].set_facecolor((red, green, 0.3))
            if acc < 0.5:
                table[(i, j)].set_text_props(weight="bold", color="white")
    
    # 情境欄
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor("#E0E0E0")
        table[(i, 0)].set_text_props(weight="bold")
    
    plt.title("Robustness Summary: 3 Models × 7 Scenarios (S3 Worst Case)", fontsize=14, weight="bold", pad=20)
    
    # 總結文字
    summary_text = []
    summary_text.append("Summary:")
    
    # 找出每個情境的最佳模型
    for scenario in SCENARIOS.keys():
        best_model = max(MODELS, key=lambda m: data[m][scenario])
        best_acc = data[best_model][scenario]
        summary_text.append(f"  {SCENARIOS[scenario]}: {MODEL_NAMES[best_model]} best ({best_acc*100:.2f}%)")
    
    # 整體平均
    avg_acc = {m: np.mean([data[m][s] for s in SCENARIOS.keys()]) for m in MODELS}
    best_overall = max(MODELS, key=lambda m: avg_acc[m])
    summary_text.append(f"\nOverall Average: {MODEL_NAMES[best_overall]} most robust ({avg_acc[best_overall]*100:.2f}%)")
    summary_text.append(f"  MLP: {avg_acc['mlp']*100:.2f}% | CNN: {avg_acc['cnn']*100:.2f}% | ResNet: {avg_acc['resnet']*100:.2f}%")
    
    summary_str = "\n".join(summary_text)
    fig.text(0.5, 0.02, summary_str, ha="center", fontsize=9, wrap=True)
    
    plt.tight_layout()
    output_path = OUT_BASE / "robustness_summary_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Summary table saved to: {output_path}")
    print("\n" + summary_str)


if __name__ == "__main__":
    create_summary_table()
