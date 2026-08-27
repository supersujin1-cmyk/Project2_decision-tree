"""
필요 라이브러리: pandas, numpy, scikit-learn, matplotlib
실행: python analysis/08_decision_tree_baseline.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
범주형 변수는 원-핫 인코딩, 수치형 변수(duration, days_left)는 그대로 사용해
의사결정나무(회귀) 모델을 학습하고 테스트 성능을 기준 모델과 비교한다.
하이퍼파라미터: max_depth=2, min_samples_leaf=500, random_state=42
data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

TRAIN_PATH = "data/processed/flight_prices_train.csv"
TEST_PATH = "data/processed/flight_prices_test.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]

RANDOM_STATE = 42
MAX_DEPTH = 2
MIN_SAMPLES_LEAF = 500

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["price"]
y_test = test_df["price"]

X_train = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = pd.get_dummies(test_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # 학습 기준 열로 맞춤 (테스트에만 있는 범주 유입 방지)

print(f"입력변수 개수 (원-핫 인코딩 후): {X_train.shape[1]}개")

model = DecisionTreeRegressor(
    max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

# 기준 모델 (직전 단계 결과값)
baseline = pd.DataFrame([
    {"모델": "평균 기준 모델", "MAE": 19763.694116, "RMSE": 22721.251817, "R2": -0.000001},
    {"모델": "중앙값 기준 모델", "MAE": 16155.283653, "RMSE": 26420.570832, "R2": -0.352136},
    {"모델": "의사결정나무(depth=2)", "MAE": mae, "RMSE": rmse, "R2": r2},
])
baseline["MAE_개선율(%, vs 평균모델)"] = (1 - baseline["MAE"] / 19763.694116) * 100
baseline["RMSE_개선율(%, vs 평균모델)"] = (1 - baseline["RMSE"] / 22721.251817) * 100
baseline.to_csv(f"{TABLE_DIR}/08_tree_vs_baseline_metrics.csv", index=False, encoding="utf-8-sig")

print("\n=== 테스트 성능 비교 ===")
print(baseline.to_string(index=False))

# 나무 그림
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(model, feature_names=X_train.columns, filled=True, rounded=True,
          fontsize=9, ax=ax, precision=1)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/08_decision_tree.png", dpi=150)
plt.close(fig)

# 텍스트 규칙
rules_text = export_text(model, feature_names=list(X_train.columns), decimals=2)
print("\n=== 트리 규칙 (텍스트) ===")
print(rules_text)
