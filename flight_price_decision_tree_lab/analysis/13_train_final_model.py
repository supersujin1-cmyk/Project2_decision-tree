"""
필요 라이브러리: pandas, scikit-learn, joblib
실행: python analysis/13_train_final_model.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
대시보드에서 사용할 최종 모델(max_depth=8, min_samples_leaf=200, random_state=42)을 학습하고
테스트 성능을 출력한 뒤, 모델과 대시보드에 필요한 메타데이터(입력 열 목록, 범주형 선택지,
수치형 변수 범위)를 outputs/models/에 저장한다.
data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

TRAIN_PATH = "data/processed/flight_prices_train.csv"
TEST_PATH = "data/processed/flight_prices_test.csv"
MODEL_DIR = "outputs/models"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]

RANDOM_STATE = 42
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 200

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["price"]
y_test = test_df["price"]

X_train = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = pd.get_dummies(test_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model = DecisionTreeRegressor(
    max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
rmse = mean_squared_error(y_test, pred_test) ** 0.5
r2 = r2_score(y_test, pred_test)

print(f"모델: max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, random_state={RANDOM_STATE}")
print(f"리프 개수: {model.get_n_leaves()}")
print(f"테스트 MAE: {mae:,.2f}")
print(f"테스트 RMSE: {rmse:,.2f}")
print(f"테스트 R2: {r2:.4f}")

import os
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, f"{MODEL_DIR}/tree_depth8_leaf200.joblib")

metadata = {
    "cat_cols": CAT_COLS,
    "num_cols": NUM_COLS,
    "feature_columns": X_train.columns.tolist(),
    "categories": {col: sorted(train_df[col].dropna().unique().tolist()) for col in CAT_COLS},
    "numeric_ranges": {
        col: {"min": float(train_df[col].min()), "max": float(train_df[col].max()),
              "median": float(train_df[col].median())}
        for col in NUM_COLS
    },
    "test_metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
    "hyperparameters": {"max_depth": MAX_DEPTH, "min_samples_leaf": MIN_SAMPLES_LEAF, "random_state": RANDOM_STATE},
}
with open(f"{MODEL_DIR}/model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n모델 및 메타데이터 저장 완료: {MODEL_DIR}/tree_depth8_leaf200.joblib, {MODEL_DIR}/model_metadata.json")
