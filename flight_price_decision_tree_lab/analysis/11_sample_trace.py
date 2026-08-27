"""
필요 라이브러리: pandas, scikit-learn, matplotlib
실행: python analysis/11_sample_trace.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
테스트 데이터에서 Economy 1건, Business 1건을 무작위로 뽑아
학습된 max_depth=4 트리를 뿌리부터 리프까지 따라가며
각 분기 조건 충족 여부와 최종 예측값-실제값 차이를 계산/시각화한다.
min_samples_leaf=500, random_state=42는 고정 (샘플 추출도 random_state=42).
data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree

matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"  # 한글 글리프 깨짐 방지 (macOS)
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_PATH = "data/processed/flight_prices_train.csv"
TEST_PATH = "data/processed/flight_prices_test.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]
DISPLAY_COLS = ["class", "airline", "source_city", "destination_city",
                 "departure_time", "arrival_time", "stops", "duration", "days_left", "price"]

RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 500
MAX_DEPTH = 4

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["price"]
X_train = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)

model = DecisionTreeRegressor(
    max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)
tree_ = model.tree_
feature_names = X_train.columns.tolist()


def to_readable_condition(fname):
    for col in CAT_COLS:
        if fname.startswith(col + "_"):
            category = fname[len(col) + 1:]
            return col, category
    return fname, None


def trace_sample(row_raw, row_encoded):
    node = 0
    steps = []
    while tree_.feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = tree_.feature[node]
        thr = tree_.threshold[node]
        fname = feature_names[feat_idx]
        val = row_encoded[fname]
        col, category = to_readable_condition(fname)

        go_left = val <= thr
        if category is not None:
            # 범주형(원-핫): 이 값이 해당 카테고리이면 답 "예" -> 오른쪽, 아니면 "아니오" -> 왼쪽
            condition_text = f"{col} == '{category}' ?"
            actual_text = f"실제 {col} = '{row_raw[col]}'"
            answer_yes = row_raw[col] == category
        else:
            # 수치형: 값이 기준 이하이면 답 "예" -> 왼쪽, 초과이면 "아니오" -> 오른쪽
            condition_text = f"{fname} <= {thr:.2f} ?"
            actual_text = f"실제 {fname} = {row_raw[fname]:.2f}"
            answer_yes = go_left

        branch = "왼쪽" if go_left else "오른쪽"
        steps.append({
            "condition": condition_text,
            "actual": actual_text,
            "answer_yes": answer_yes,
            "branch": branch,
        })
        node = tree_.children_left[node] if go_left else tree_.children_right[node]

    predicted = tree_.value[node][0][0]
    return steps, predicted


def get_sample(class_name):
    sub = test_df[test_df["class"] == class_name].sample(n=1, random_state=RANDOM_STATE)
    row_raw = sub.iloc[0]
    row_encoded = pd.get_dummies(sub[CAT_COLS + NUM_COLS], columns=CAT_COLS)
    row_encoded = row_encoded.reindex(columns=feature_names, fill_value=0).iloc[0]
    return row_raw, row_encoded


samples = {}
for cls in ["Economy", "Business"]:
    row_raw, row_encoded = get_sample(cls)
    steps, predicted = trace_sample(row_raw, row_encoded)
    actual = row_raw["price"]
    samples[cls] = {"row_raw": row_raw, "steps": steps, "predicted": predicted, "actual": actual}

    print(f"\n=== {cls} 샘플 ===")
    print(row_raw[DISPLAY_COLS].to_string())
    print(f"\n[결정 경로]")
    for i, s in enumerate(steps, 1):
        ans = "예" if s["answer_yes"] else "아니오"
        print(f"{i}. 질문: {s['condition']}  |  {s['actual']}  |  답: {ans}  |  이동: {s['branch']} 가지")
    print(f"\n리프 예측 가격: {predicted:,.2f}")
    print(f"실제 가격: {actual:,.0f}")
    print(f"실제-예측 차이: {actual - predicted:,.2f}")

# 결과 표 저장
summary_rows = []
for cls, d in samples.items():
    summary_rows.append({
        "class": cls,
        "실제_가격": d["actual"],
        "예측_가격": d["predicted"],
        "차이(실제-예측)": d["actual"] - d["predicted"],
    })
pd.DataFrame(summary_rows).to_csv(f"{TABLE_DIR}/11_sample_trace_summary.csv", index=False, encoding="utf-8-sig")

# 시각화: 두 항공권의 결정 경로를 박스-화살표 흐름도로 표현
fig, axes = plt.subplots(1, 2, figsize=(11, 9))
colors = {"Economy": "#4C72B0", "Business": "#DD8452"}

for ax, cls in zip(axes, ["Economy", "Business"]):
    d = samples[cls]
    steps = d["steps"]
    n_boxes = len(steps) + 1  # 분기 박스 + 최종 리프 박스
    box_h = 1.0
    gap = 0.5
    total_h = n_boxes * box_h + (n_boxes - 1) * gap
    y_top = total_h

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, total_h + 1)
    ax.axis("off")
    ax.set_title(f"{cls} 항공권 (실제가격 {d['actual']:,.0f})", fontsize=12, fontweight="bold")

    y = y_top
    for i, s in enumerate(steps):
        ans = "예" if s["answer_yes"] else "아니오"
        box_color = "#C7E9C0" if s["answer_yes"] else "#FDD0A2"
        text = f"{s['condition']}\n{s['actual']}\n답: {ans} -> {s['branch']} 가지"
        box = mpatches.FancyBboxPatch((0.5, y - box_h), 9, box_h,
                                       boxstyle="round,pad=0.05", linewidth=1,
                                       edgecolor="black", facecolor=box_color)
        ax.add_patch(box)
        ax.text(5, y - box_h / 2, text, ha="center", va="center", fontsize=8.5)
        ax.annotate("", xy=(5, y - box_h - gap + 0.05), xytext=(5, y - box_h),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        y -= (box_h + gap)

    leaf_box = mpatches.FancyBboxPatch((0.5, y - box_h), 9, box_h,
                                        boxstyle="round,pad=0.05", linewidth=1.5,
                                        edgecolor="black", facecolor=colors[cls], alpha=0.5)
    ax.add_patch(leaf_box)
    ax.text(5, y - box_h / 2,
            f"리프 예측가격: {d['predicted']:,.0f}\n실제가격: {d['actual']:,.0f}\n차이: {d['actual']-d['predicted']:,.0f}",
            ha="center", va="center", fontsize=9, fontweight="bold")

fig.suptitle("depth=4 트리 결정 경로 추적 (Economy vs Business 샘플)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIG_DIR}/11_sample_trace_path.png", dpi=140)
plt.close(fig)

print("\n결정 경로 시각화 저장 완료: outputs/figures/11_sample_trace_path.png")
