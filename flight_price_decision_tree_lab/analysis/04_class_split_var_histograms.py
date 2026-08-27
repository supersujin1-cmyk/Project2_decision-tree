"""
필요 라이브러리: pandas, matplotlib
실행: python analysis/04_class_split_var_histograms.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
class(Economy/Business)로 나눈 뒤, 각 그룹 안에서 설명변수들의 분포를
(범주형은 막대그래프, 수치형은 히스토그램) 하나의 그림에 모아 outputs/figures/에 저장한다.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/flight_prices.csv"
FIG_DIR = "outputs/figures"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops"]
NUM_COLS = ["duration", "days_left"]

df = pd.read_csv(RAW_PATH)


def plot_class_subset(sub_df, class_name, color):
    n_cols = 4
    n_rows = int(np.ceil((len(CAT_COLS) + len(NUM_COLS)) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(CAT_COLS):
        counts = sub_df[col].value_counts().sort_values(ascending=False)
        ax = axes[i]
        ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor="white")
        ax.set_title(col)
        ax.set_ylabel("count")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    for j, col in enumerate(NUM_COLS):
        ax = axes[len(CAT_COLS) + j]
        ax.hist(sub_df[col], bins=40, color=color, edgecolor="white")
        ax.set_title(col)
        ax.set_ylabel("count")

    for k in range(len(CAT_COLS) + len(NUM_COLS), len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"class = {class_name} (n={len(sub_df)}) - Explanatory Variable Distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG_DIR}/06_var_histograms_{class_name.lower()}.png", dpi=120)
    plt.close(fig)


plot_class_subset(df[df["class"] == "Economy"], "Economy", "#4C72B0")
plot_class_subset(df[df["class"] == "Business"], "Business", "#DD8452")

print("class별 설명변수 히스토그램 저장 완료 (outputs/figures/06_var_histograms_economy.png, 06_var_histograms_business.png)")
