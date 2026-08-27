"""
필요 라이브러리: pandas, scikit-learn, matplotlib
실행: python analysis/12_depth_leaf_comparison.py (프로젝트 루트인 flight_price_decision_tree_lab 기준)

두 가지 실험을 수행한다.
[실험 A] max_depth = 2, 4, 6, 8, 12, None(제한없음) 비교. min_samples_leaf=500 고정.
[실험 B] min_samples_leaf = 500, 200, 100, 50, 20 비교. max_depth=None(제한없음) 고정
         (max_depth를 고정해두지 않으면 min_samples_leaf 변화가 트리에 영향을 주지 못하는
          경우가 많아, 리프 크기 효과를 보기 위해 깊이 제한을 풀었다).
random_state=42 고정. data/processed/의 학습·테스트 분할(random_state=42)을 그대로 사용한다.
원본 파일을 수정하지 않는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_PATH = "data/processed/flight_prices_train.csv"
TEST_PATH = "data/processed/flight_prices_test.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

CAT_COLS = ["airline", "source_city", "destination_city",
            "departure_time", "arrival_time", "stops", "class"]
NUM_COLS = ["duration", "days_left"]

RANDOM_STATE = 42

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
y_train = train_df["price"]
y_test = test_df["price"]

X_train = pd.get_dummies(train_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = pd.get_dummies(test_df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


def evaluate(model):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "리프_개수": model.get_n_leaves(),
        "학습_MAE": mean_absolute_error(y_train, pred_train),
        "테스트_MAE": mean_absolute_error(y_test, pred_test),
        "학습_RMSE": mean_squared_error(y_train, pred_train) ** 0.5,
        "테스트_RMSE": mean_squared_error(y_test, pred_test) ** 0.5,
        "학습_R2": r2_score(y_train, pred_train),
        "테스트_R2": r2_score(y_test, pred_test),
    }


def plot_train_test(result, x_col, x_labels, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_pairs = [("MAE", "학습_MAE", "테스트_MAE"),
                     ("RMSE", "학습_RMSE", "테스트_RMSE"),
                     ("R2", "학습_R2", "테스트_R2")]
    x_pos = range(len(x_labels))
    for ax, (metric_name, train_col, test_col) in zip(axes, metric_pairs):
        ax.plot(x_pos, result[train_col], marker="o", label="학습(train)", color="#4C72B0")
        ax.plot(x_pos, result[test_col], marker="o", label="테스트(test)", color="#DD8452")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===== 실험 A: max_depth 비교 (min_samples_leaf=500 고정) =====
DEPTHS = [2, 4, 6, 8, 12, None]
depth_labels = [str(d) if d is not None else "제한없음" for d in DEPTHS]

rows_a = []
for depth in DEPTHS:
    model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=500, random_state=RANDOM_STATE)
    metrics = evaluate(model)
    rows_a.append({"max_depth": depth_labels[DEPTHS.index(depth)], **metrics})

result_a = pd.DataFrame(rows_a)
result_a.to_csv(f"{TABLE_DIR}/12_depth_sweep_metrics.csv", index=False, encoding="utf-8-sig")
print("=== 실험 A: max_depth 비교 (min_samples_leaf=500 고정) ===")
print(result_a.to_string(index=False))

plot_train_test(result_a, "max_depth", depth_labels,
                 "max_depth에 따른 학습 vs 테스트 오차 (min_samples_leaf=500)",
                 f"{FIG_DIR}/12_depth_sweep_train_test.png")

# ===== 실험 B: min_samples_leaf 비교 (max_depth=None 고정) =====
LEAF_SIZES = [500, 200, 100, 50, 20]

rows_b = []
for leaf_size in LEAF_SIZES:
    model = DecisionTreeRegressor(max_depth=None, min_samples_leaf=leaf_size, random_state=RANDOM_STATE)
    metrics = evaluate(model)
    rows_b.append({"min_samples_leaf": leaf_size, **metrics})

result_b = pd.DataFrame(rows_b)
result_b.to_csv(f"{TABLE_DIR}/12_leaf_sweep_metrics.csv", index=False, encoding="utf-8-sig")
print("\n=== 실험 B: min_samples_leaf 비교 (max_depth=None 고정) ===")
print(result_b.to_string(index=False))

plot_train_test(result_b, "min_samples_leaf", [str(v) for v in LEAF_SIZES],
                 "min_samples_leaf에 따른 학습 vs 테스트 오차 (max_depth=제한없음)",
                 f"{FIG_DIR}/12_leaf_sweep_train_test.png")

print("\n그림 저장 완료: outputs/figures/12_depth_sweep_train_test.png, 12_leaf_sweep_train_test.png")
