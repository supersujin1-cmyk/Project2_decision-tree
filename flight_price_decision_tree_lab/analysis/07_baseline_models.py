"""
필요 라이브러리: pandas, numpy, scikit-learn
실행: python analysis/07_baseline_models.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
학습 데이터의 price 평균/중앙값을 테스트 데이터 전체에 예측하는 두 기준 모델을 만들고
테스트 MAE, RMSE, R^2를 계산해 outputs/tables/에 저장한다.
data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "data/processed/flight_prices_train.csv"
TEST_PATH = "data/processed/flight_prices_test.csv"
TABLE_DIR = "outputs/tables"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["price"]
y_test = test_df["price"]

train_mean = y_train.mean()
train_median = y_train.median()

pred_mean = np.full(len(y_test), train_mean)
pred_median = np.full(len(y_test), train_median)

rows = []
for name, pred in [("평균 기준 모델", pred_mean), ("중앙값 기준 모델", pred_median)]:
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    rows.append({"모델": name, "예측값": pred[0], "MAE": mae, "RMSE": rmse, "R2": r2})

result = pd.DataFrame(rows)
result.to_csv(f"{TABLE_DIR}/07_baseline_metrics.csv", index=False, encoding="utf-8-sig")

print(f"학습 데이터 price 평균: {train_mean:,.2f}")
print(f"학습 데이터 price 중앙값: {train_median:,.2f}\n")
print(result.to_string(index=False))
