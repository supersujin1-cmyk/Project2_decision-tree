"""
필요 라이브러리: pandas, numpy
실행: python analysis/05_first_split_rss.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
회귀 트리의 첫 분할 후보를 정하기 위해, 각 설명변수별 최적 분할과
그때의 잔차제곱합(RSS = sum((y - group_mean)^2), 좌/우 그룹 합)을 계산한다.
식별자(Unnamed: 0)와 airline과 중복되는 flight는 후보에서 제외한다.
원본 파일을 수정하지 않는다.
"""
import itertools

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/flight_prices.csv"
TABLE_DIR = "outputs/tables"

CAT_COLS = ["class", "airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops"]
NUM_COLS = ["duration", "days_left"]

df = pd.read_csv(RAW_PATH)
y_all = df["price"].astype("float64")  # int64 제곱합이 오버플로우하므로 float64로 변환


def numeric_best_split(x, y):
    tmp = pd.DataFrame({"x": x, "y": y})
    grp = tmp.groupby("x")["y"].agg(n="count", s="sum", sq=lambda v: (v ** 2).sum())
    grp = grp.sort_index()
    n_total, sum_total, sq_total = grp["n"].sum(), grp["s"].sum(), grp["sq"].sum()
    cum_n = grp["n"].cumsum().values
    cum_s = grp["s"].cumsum().values
    cum_sq = grp["sq"].cumsum().values
    xs = grp.index.values

    best_rss, best_thr = np.inf, None
    for i in range(len(xs) - 1):
        n_l, n_r = cum_n[i], n_total - cum_n[i]
        if n_l == 0 or n_r == 0:
            continue
        s_l, s_r = cum_s[i], sum_total - cum_s[i]
        sq_l, sq_r = cum_sq[i], sq_total - cum_sq[i]
        rss = (sq_l - s_l ** 2 / n_l) + (sq_r - s_r ** 2 / n_r)
        if rss < best_rss:
            best_rss, best_thr = rss, (xs[i] + xs[i + 1]) / 2
    return f"<= {best_thr:.3f}", best_rss


def categorical_best_split(x, y):
    tmp = pd.DataFrame({"x": x, "y": y})
    grp = tmp.groupby("x")["y"].agg(n="count", s="sum", sq=lambda v: (v ** 2).sum())
    levels = grp.index.tolist()
    n_total, sum_total, sq_total = grp["n"].sum(), grp["s"].sum(), grp["sq"].sum()
    k = len(levels)

    best_rss, best_subset = np.inf, None
    for r in range(1, k):
        for combo in itertools.combinations(levels, r):
            sub = grp.loc[list(combo)]
            n_l, s_l, sq_l = sub["n"].sum(), sub["s"].sum(), sub["sq"].sum()
            n_r, s_r, sq_r = n_total - n_l, sum_total - s_l, sq_total - sq_l
            if n_l == 0 or n_r == 0:
                continue
            rss = (sq_l - s_l ** 2 / n_l) + (sq_r - s_r ** 2 / n_r)
            if rss < best_rss:
                best_rss, best_subset = rss, combo
    return f"in {best_subset}", best_rss


tss = float(((y_all - y_all.mean()) ** 2).sum())

rows = []
for col in CAT_COLS:
    rule, rss = categorical_best_split(df[col], y_all)
    rows.append({"variable": col, "type": "categorical", "split_rule": rule,
                 "rss": rss, "sse_reduction": tss - rss})

for col in NUM_COLS:
    rule, rss = numeric_best_split(df[col], y_all)
    rows.append({"variable": col, "type": "numeric", "split_rule": rule,
                 "rss": rss, "sse_reduction": tss - rss})

result = pd.DataFrame(rows).sort_values("rss").reset_index(drop=True)
result.to_csv(f"{TABLE_DIR}/05_first_split_candidates.csv", index=False)

print(f"전체 잔차제곱합 TSS (분할 전) = {tss:,.0f}\n")
print(result.to_string(index=False))
print(f"\n=> 첫 분할 추천: {result.iloc[0]['variable']} ({result.iloc[0]['split_rule']})")
