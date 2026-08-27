"""
필요 라이브러리: pandas, matplotlib, numpy
실행: python analysis/03_price_histograms.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
price 분포를 다양한 방식(로그축, class/stops별 분리, 수치형 변수)으로 시각화해 outputs/figures/에 저장한다.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/flight_prices.csv"
FIG_DIR = "outputs/figures"

df = pd.read_csv(RAW_PATH)

# 1. price 히스토그램 - 로그축 (오른쪽 꼬리 분포 확인용)
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["price"], bins=50, color="#4C72B0", edgecolor="white")
ax.set_xscale("log")
ax.set_xlabel("price (log scale)")
ax.set_ylabel("count")
ax.set_title("Price Distribution (log x-axis)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_price_histogram_logscale.png", dpi=120)
plt.close(fig)

# 2. price 히스토그램 - class별 겹쳐그리기
fig, ax = plt.subplots(figsize=(7, 5))
for cls, color in [("Economy", "#4C72B0"), ("Business", "#DD8452")]:
    ax.hist(df.loc[df["class"] == cls, "price"], bins=50, alpha=0.6, label=cls, color=color, edgecolor="white")
ax.set_xlabel("price")
ax.set_ylabel("count")
ax.set_title("Price Distribution by Class")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_price_histogram_by_class.png", dpi=120)
plt.close(fig)

# 3. price 히스토그램 - class별 서브플롯 (각자 스케일)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
for ax, cls, color in zip(axes, ["Economy", "Business"], ["#4C72B0", "#DD8452"]):
    ax.hist(df.loc[df["class"] == cls, "price"], bins=50, color=color, edgecolor="white")
    ax.set_title(f"class = {cls}")
    ax.set_xlabel("price")
    ax.set_ylabel("count")
fig.suptitle("Price Distribution by Class (separate scales)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_price_histogram_by_class_subplots.png", dpi=120)
plt.close(fig)

# 4. price 히스토그램 - stops별 겹쳐그리기
fig, ax = plt.subplots(figsize=(7, 5))
colors = {"zero": "#4C72B0", "one": "#DD8452", "two_or_more": "#55A868"}
for stop_val, color in colors.items():
    ax.hist(df.loc[df["stops"] == stop_val, "price"], bins=50, alpha=0.5, label=stop_val, color=color, edgecolor="white")
ax.set_xlabel("price")
ax.set_ylabel("count")
ax.set_title("Price Distribution by Stops")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_price_histogram_by_stops.png", dpi=120)
plt.close(fig)

# 5. duration 히스토그램
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["duration"], bins=50, color="#55A868", edgecolor="white")
ax.set_xlabel("duration (hours)")
ax.set_ylabel("count")
ax.set_title("Duration Distribution")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_duration_histogram.png", dpi=120)
plt.close(fig)

# 6. days_left 히스토그램
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["days_left"], bins=np.arange(1, 51) - 0.5, color="#C44E52", edgecolor="white")
ax.set_xlabel("days_left")
ax.set_ylabel("count")
ax.set_title("Days Left Distribution")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_days_left_histogram.png", dpi=120)
plt.close(fig)

print("히스토그램 6종 저장 완료 (outputs/figures/05_*.png)")
