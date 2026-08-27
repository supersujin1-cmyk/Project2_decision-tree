"""
필요 라이브러리: pandas, scikit-learn
실행: python analysis/09_depth_comparison.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
max_depth를 2, 4, 6으로 바꿔가며 의사결정나무(회귀)를 학습하고
학습/테스트 MAE, RMSE, R2를 비교해 outputs/tables/에 저장한다.
min_samples_leaf=500, random_state=42는 고정.
data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

TRAIN_PATH = "data/processed/flight_prices_train.csv"
TEST_PATH = "data/processed/flight_prices_test.csv"
TABLE_DIR = "outputs/tables"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]

RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 500
DEPTHS = [2, 4, 6]

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["price"]
y_test = test_df["price"]

X_train = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = pd.get_dummies(test_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

rows = []
for depth in DEPTHS:
    model = DecisionTreeRegressor(
        max_depth=depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    rows.append({
        "max_depth": depth,
        "리프 개수": model.get_n_leaves(),
        "학습_MAE": mean_absolute_error(y_train, pred_train),
        "학습_RMSE": mean_squared_error(y_train, pred_train) ** 0.5,
        "학습_R2": r2_score(y_train, pred_train),
        "테스트_MAE": mean_absolute_error(y_test, pred_test),
        "테스트_RMSE": mean_squared_error(y_test, pred_test) ** 0.5,
        "테스트_R2": r2_score(y_test, pred_test),
    })

result = pd.DataFrame(rows)
result["학습-테스트_MAE_차이"] = result["테스트_MAE"] - result["학습_MAE"]
result.to_csv(f"{TABLE_DIR}/09_depth_comparison_metrics.csv", index=False, encoding="utf-8-sig")

print(result.to_string(index=False))
