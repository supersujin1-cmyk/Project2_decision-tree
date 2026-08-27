"""
필요 라이브러리: pandas, scikit-learn, matplotlib
실행: python analysis/10_depth_trees_plot.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)
max_depth=4, 6 의사결정나무(회귀)를 학습해 나무 그림(PNG)과 텍스트 규칙을 저장한다.
min_samples_leaf=500, random_state=42는 고정.
data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

TRAIN_PATH = "data/processed/flight_prices_train.csv"
FIG_DIR = "outputs/figures"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]

RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 500
DEPTHS = [4, 6]

train_df = pd.read_csv(TRAIN_PATH)
y_train = train_df["price"]
X_train = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)

for depth in DEPTHS:
    model = DecisionTreeRegressor(
        max_depth=depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    fig_w = 6 * depth
    fig_h = 3 * depth
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(model, feature_names=X_train.columns, filled=True, rounded=True,
              fontsize=8, ax=ax, precision=1)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_decision_tree_depth{depth}.png", dpi=150)
    plt.close(fig)

    rules_text = export_text(model, feature_names=list(X_train.columns), decimals=2)
    print(f"=== depth={depth} 리프 개수: {model.get_n_leaves()} ===")
    print(rules_text)
    print()

print("나무 그림 저장 완료: outputs/figures/10_decision_tree_depth4.png, depth6.png")
