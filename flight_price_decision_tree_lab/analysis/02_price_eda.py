"""
필요 라이브러리: pandas, matplotlib
실행: python analysis/02_price_eda.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
data/raw/flight_prices.csv를 읽어 price 요약통계/히스토그램과
설명변수별 price 관계(요약통계, 히스토그램, 산포도)를 outputs/에 저장한다.
표본추출 난수: random_state=42, 산포도는 5000행 샘플링.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAW_PATH = "data/raw/flight_prices.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"
RANDOM_STATE = 42
SAMPLE_N = 5000

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]

df = pd.read_csv(RAW_PATH)
print(f"제외 열: Unnamed: 0 (식별자), flight (airline과 중복 정보 + 고유값 {df['flight'].nunique()}개 고카디널리티)")

# 1. price 전체 요약통계
price_summary = df["price"].describe()
price_summary.to_csv(f"{TABLE_DIR}/02_price_summary.csv", header=["price"])
print("\n=== price 요약통계 ===")
print(price_summary)

# 2. price 히스토그램
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["price"], bins=50, color="#4C72B0", edgecolor="white")
ax.set_xlabel("price")
ax.set_ylabel("count")
ax.set_title("Price Distribution")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/02_price_histogram.png", dpi=120)
plt.close(fig)

# 3. 범주형 설명변수별 price 요약통계 + boxplot
for col in CAT_COLS:
    grp = df.groupby(col)["price"].agg(["count", "mean", "median", "std", "min", "max"])
    grp = grp.sort_values("mean", ascending=False)
    grp.to_csv(f"{TABLE_DIR}/03_price_by_{col}.csv")
    print(f"\n=== price by {col} ===")
    print(grp)

    categories = grp.index.tolist()
    data_by_cat = [df.loc[df[col] == c, "price"].values for c in categories]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_by_cat, labels=categories, showfliers=False)
    ax.set_xlabel(col)
    ax.set_ylabel("price")
    ax.set_title(f"Price by {col}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_price_by_{col}.png", dpi=120)
    plt.close(fig)

# 4. 수치형 설명변수별 산포도 + 상관계수
corr_rows = []
sample_df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
for col in NUM_COLS:
    corr = df[col].corr(df["price"])
    corr_rows.append({"variable": col, "corr_with_price": corr})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample_df[col], sample_df["price"], alpha=0.3, s=10, color="#55A868")
    ax.set_xlabel(col)
    ax.set_ylabel("price")
    ax.set_title(f"price vs {col} (sample n={SAMPLE_N})")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_price_vs_{col}.png", dpi=120)
    plt.close(fig)

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(f"{TABLE_DIR}/04_price_numeric_corr.csv", index=False)
print("\n=== 수치형 변수와 price의 상관계수 ===")
print(corr_df)
