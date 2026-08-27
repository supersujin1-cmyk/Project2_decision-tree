"""
필요 라이브러리: pandas
실행: python analysis/01_eda_raw.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
data/raw/flight_prices.csv를 읽어 기초 탐색 결과를 출력한다.
원본 파일을 수정하지 않는다.
"""
import pandas as pd

RAW_PATH = "data/raw/flight_prices.csv"

df = pd.read_csv(RAW_PATH)

print("=== 1. 데이터 크기 ===")
print(f"행: {df.shape[0]}, 열: {df.shape[1]}")

print("\n=== 2. 열별 데이터 타입 ===")
print(df.dtypes)

print("\n=== 3. 상위 5개 행 ===")
print(df.head(5).to_string())

print("\n=== 4. 결측치 개수 (열별) ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"결측_건수": missing, "결측_비율(%)": missing_pct})
print(missing_table[missing_table["결측_건수"] > 0])
if missing.sum() == 0:
    print("결측치 없음")

print("\n=== 5. 중복 행 ===")
print(f"전체 열 기준 완전 중복: {df.duplicated().sum()}건")
id_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
for c in id_cols:
    print(f"{c} 기준 중복: {df[c].duplicated().sum()}건")

print("\n=== 6. 범주형 열 고유값 개수 ===")
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    print(f"\n[{col}] 고유값 {df[col].nunique()}개")
    print(df[col].value_counts(dropna=False))

print("\n=== 7. 수치형 열 기술통계 ===")
print(df.describe().to_string())
