"""
필요 라이브러리: streamlit, pandas, numpy, joblib, scikit-learn
실행: streamlit run dashboard/app.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)

max_depth=8, min_samples_leaf=200 의사결정나무(회귀) 모델로 항공권 가격을 예측하는 대시보드.
모델과 메타데이터는 analysis/13_train_final_model.py 실행 결과(outputs/models/)를 그대로 읽는다.
원본 데이터를 수정하지 않는다.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.tree import _tree

MODEL_PATH = "outputs/models/tree_depth8_leaf200.joblib"
METADATA_PATH = "outputs/models/model_metadata.json"

st.set_page_config(page_title="항공권 가격 예측", page_icon="✈️", layout="wide")
st.title("✈️ 항공권 가격 예측 대시보드")
st.caption("의사결정나무(회귀) 모델 · max_depth=8, min_samples_leaf=200, random_state=42")

if not (os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)):
    st.error(
        "학습된 모델 파일을 찾을 수 없습니다.\n\n"
        "먼저 다음 명령을 실행하세요:\n\n"
        "`python analysis/13_train_final_model.py`"
    )
    st.stop()

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

CAT_COLS = meta["cat_cols"]
NUM_COLS = meta["num_cols"]
FEATURE_COLUMNS = meta["feature_columns"]
CATEGORIES = meta["categories"]
NUM_RANGES = meta["numeric_ranges"]
TEST_METRICS = meta["test_metrics"]

COL_LABELS = {
    "class": "좌석 등급",
    "airline": "항공사",
    "source_city": "출발 도시",
    "destination_city": "도착 도시",
    "departure_time": "출발 시간대",
    "arrival_time": "도착 시간대",
    "stops": "경유 횟수",
    "duration": "비행 소요시간(시간)",
    "days_left": "출발까지 남은 일수",
}

# ---------------- 입력 영역 ----------------
st.subheader("1. 항공권 조건 입력")
col1, col2, col3 = st.columns(3)

with col1:
    class_val = st.selectbox(COL_LABELS["class"], CATEGORIES["class"])
    airline_val = st.selectbox(COL_LABELS["airline"], CATEGORIES["airline"])
    stops_val = st.selectbox(COL_LABELS["stops"], CATEGORIES["stops"])

with col2:
    source_val = st.selectbox(COL_LABELS["source_city"], CATEGORIES["source_city"])
    dest_options = [c for c in CATEGORIES["destination_city"] if c != source_val]
    destination_val = st.selectbox(COL_LABELS["destination_city"], dest_options)
    departure_val = st.selectbox(COL_LABELS["departure_time"], CATEGORIES["departure_time"])

with col3:
    arrival_val = st.selectbox(COL_LABELS["arrival_time"], CATEGORIES["arrival_time"])
    duration_val = st.slider(
        COL_LABELS["duration"],
        min_value=float(round(NUM_RANGES["duration"]["min"], 1)),
        max_value=float(round(NUM_RANGES["duration"]["max"], 1)),
        value=float(round(NUM_RANGES["duration"]["median"], 1)),
        step=0.1,
    )
    days_left_val = st.slider(
        COL_LABELS["days_left"],
        min_value=int(NUM_RANGES["days_left"]["min"]),
        max_value=int(NUM_RANGES["days_left"]["max"]),
        value=int(NUM_RANGES["days_left"]["median"]),
        step=1,
    )

raw_input = {
    "class": class_val, "airline": airline_val, "source_city": source_val,
    "destination_city": destination_val, "departure_time": departure_val,
    "arrival_time": arrival_val, "stops": stops_val,
    "duration": duration_val, "days_left": days_left_val,
}

# ---------------- 인코딩 & 예측 ----------------
input_df = pd.DataFrame([raw_input])
input_encoded = pd.get_dummies(input_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
input_encoded = input_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

predicted_price = model.predict(input_encoded)[0]

st.subheader("2. 예측 결과")
st.metric("예상 항공권 가격", f"{predicted_price:,.0f}")
st.caption(
    "원본 데이터(`flight_prices.csv`)의 가격 단위를 그대로 사용합니다(인도 국내선 데이터로, 통화는 인도 루피로 추정됩니다). "
    f"이 모델의 테스트 데이터 기준 평균 오차(MAE)는 약 {TEST_METRICS['MAE']:,.0f}이므로, "
    "예측값은 참고용이며 실제 가격과 차이가 있을 수 있습니다."
)

# ---------------- 설명: 전체 변수 중요도 ----------------
st.subheader("3. 어떤 설명변수가 고려되었는가 (전체 모델 기준)")

importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
grouped_importance = {}
for fname, imp in importances.items():
    matched = None
    for col in CAT_COLS:
        if fname.startswith(col + "_"):
            matched = col
            break
    if matched is None:
        matched = fname  # 수치형은 열 이름 그대로
    grouped_importance[matched] = grouped_importance.get(matched, 0) + imp

importance_df = pd.DataFrame({
    "설명변수": [COL_LABELS.get(k, k) for k in grouped_importance.keys()],
    "중요도": list(grouped_importance.values()),
}).sort_values("중요도", ascending=False).set_index("설명변수")

st.bar_chart(importance_df)
st.caption(
    "중요도는 트리가 분할 기준으로 각 변수를 얼마나 자주, 얼마나 효과적으로 사용했는지를 나타냅니다 "
    "(0~1 사이 값이며 전체 합은 1). 값이 클수록 모델이 가격을 예측하는 데 그 변수를 더 많이 활용했다는 뜻입니다."
)

# ---------------- 설명: 이번 입력의 결정 경로 ----------------
st.subheader("4. 이번 예측이 나온 이유 (결정 경로)")


def to_readable_condition(fname):
    for col in CAT_COLS:
        if fname.startswith(col + "_"):
            return col, fname[len(col) + 1:]
    return fname, None


tree_ = model.tree_
feature_names = FEATURE_COLUMNS
row_encoded = input_encoded.iloc[0]

node = 0
step_no = 1
while tree_.feature[node] != _tree.TREE_UNDEFINED:
    feat_idx = tree_.feature[node]
    thr = tree_.threshold[node]
    fname = feature_names[feat_idx]
    val = row_encoded[fname]
    col, category = to_readable_condition(fname)
    go_left = val <= thr

    if category is not None:
        label = COL_LABELS.get(col, col)
        answer_yes = raw_input[col] == category
        question = f"**{label}**이(가) '{category}' 인가요?"
        actual = f"입력값: {raw_input[col]}"
    else:
        label = COL_LABELS.get(fname, fname)
        answer_yes = go_left
        question = f"**{label}**이(가) {thr:.2f} 이하인가요?"
        actual = f"입력값: {raw_input[fname]:.2f}"

    ans_text = "예" if answer_yes else "아니오"
    branch_text = "왼쪽" if go_left else "오른쪽"
    st.markdown(f"{step_no}. {question}  ({actual}) → **{ans_text}** → {branch_text} 가지로 이동")

    node = tree_.children_left[node] if go_left else tree_.children_right[node]
    step_no += 1

leaf_value = tree_.value[node][0][0]
st.success(f"최종 도착한 리프의 평균 가격: {leaf_value:,.0f} (= 위 예측 결과)")

st.caption(
    "이 순서는 학습된 트리가 뿌리(첫 질문)부터 리프(최종 예측)까지 실제로 거친 분기 조건입니다. "
    "즉 지금 입력한 조건이 왜 이 가격으로 예측됐는지를 그대로 보여줍니다."
)
