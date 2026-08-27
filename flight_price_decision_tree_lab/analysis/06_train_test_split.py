"""
필요 라이브러리: pandas, scikit-learn
실행: python analysis/06_train_test_split.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
data/raw/flight_prices.csv를 학습/테스트로 분할해 data/processed/에 저장한다.
- Unnamed: 0 은 식별자이므로 row_id로 이름만 바꿔 보존한다 (입력변수로는 사용하지 않음).
- flight 는 airline과 중복 정보 + 고카디널리티(1561종)라 제외한다.
- class 비율을 유지하도록 stratify=class 적용.
- random_state=42, test_size=0.2로 고정해 재현 가능하게 한다.
원본 파일을 수정하지 않는다.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/flight_prices.csv"
PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(RAW_PATH)

df = df.rename(columns={"Unnamed: 0": "row_id"})
df = df.drop(columns=["flight"])
print(f"제외 열: flight (airline과 중복 정보 + 고유값 1561개 고카디널리티)")
print(f"보존(입력 제외) 열: row_id (식별자, 그룹 분할/추적용)")

train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["class"]
)

train_df.to_csv(f"{PROCESSED_DIR}/flight_prices_train.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(f"{PROCESSED_DIR}/flight_prices_test.csv", index=False, encoding="utf-8-sig")

print(f"\n전체 {len(df)}건 -> 학습 {len(train_df)}건 / 테스트 {len(test_df)}건 (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
print("\nclass 비율 확인 (stratify 적용 결과)")
print(pd.DataFrame({
    "전체": df["class"].value_counts(normalize=True).round(4),
    "학습": train_df["class"].value_counts(normalize=True).round(4),
    "테스트": test_df["class"].value_counts(normalize=True).round(4),
}))
