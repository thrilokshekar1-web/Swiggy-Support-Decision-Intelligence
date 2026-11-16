import csv
import math
import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

# 1. LOAD DATA
data = np.genfromtxt("data/records.csv", delimiter=",", names=True)

feature_names = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score",
    "reopened_flag",
    "self_service_first_flag"
]

X = np.column_stack([data[n] for n in feature_names])
y = np.asarray(data["escalated"], dtype=int)

# 2. EXACT STRATIFIED SPLIT: INTEGER ARITHMETIC
negative_idx = np.where(y == 0)[0]
positive_idx = np.where(y == 1)[0]

n_negative = (7 * len(negative_idx)) // 10
n_positive = (7 * len(positive_idx)) // 10

train_idx = np.sort(np.concatenate([
    negative_idx[:n_negative],
    positive_idx[:n_positive]
]))

test_idx = np.setdiff1d(np.arange(len(y)), train_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Training rows:", len(train_idx))
print("Training positives:", int(y_train.sum()))
print("Test rows:", len(test_idx))
print("Test positives:", int(y_test.sum()))

# 3. STANDARDISE USING TRAINING DATA
mean_X = X_train.mean(axis=0)
sd_X = X_train.std(axis=0, ddof=1)

Z_train = (X_train - mean_X) / sd_X
Z_test = (X_test - mean_X) / sd_X

# 4. ENTROPY AND GINI
def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = np.mean(labels)
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1-p) * math.log2(1-p)

def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = np.mean(labels)
    return 1 - p*p - (1-p)*(1-p)

root_entropy = entropy(y_train)
root_gini = gini(y_train)

# 5. FOUR CANDIDATE SPLITS
conditions_train = [
    X_train[:, 5] == 1,
    X_train[:, 6] == 1,
    X_train[:, 2] > 0.5,
    X_train[:, 0] <= 24
]

conditions_test = [
    X_test[:, 5] == 1,
    X_test[:, 6] == 1,
    X_test[:, 2] > 0.5,
    X_test[:, 0] <= 24
]

split_names = ["S1", "S2", "S3", "S4"]
split_rows = []
info_gains = []
gini_gains = []

for name, condition in zip(split_names, conditions_train):
    yes = y_train[condition]
    no = y_train[~condition]

    n_yes = len(yes)
    n_no = len(no)

    ig = (
        root_entropy
        - n_yes / len(y_train) * entropy(yes)
        - n_no / len(y_train) * entropy(no)
    )

    gg = (
        root_gini
        - n_yes / len(y_train) * gini(yes)
        - n_no / len(y_train) * gini(no)
    )

    info_gains.append(ig)
    gini_gains.append(gg)

    split_rows.append([
        name, n_yes, int(yes.sum()),
        n_no, int(no.sum()), ig, gg
    ])

best_info_index = int(np.argmax(info_gains))
best_gini_index = int(np.argmax(gini_gains))

best_split_by_info_gain = split_names[best_info_index]
best_split_by_gini_gain = split_names[best_gini_index]

# 6. DECISION STUMP
best_condition_train = conditions_train[best_info_index]
best_condition_test = conditions_test[best_info_index]

stump_rate_yes = float(y_train[best_condition_train].mean())
stump_rate_no = float(y_train[~best_condition_train].mean())

stump_probability = np.where(
    best_condition_test,
    stump_rate_yes,
    stump_rate_no
)

# 7. LOGISTIC REGRESSION: 1000 UPDATES
def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))

weights = np.zeros(7)
bias = 0.0

for _ in range(1000):
    p = sigmoid(Z_train @ weights + bias)
    error = p - y_train
    weights -= 0.1 * (Z_train.T @ error) / len(y_train)
    bias -= 0.1 * np.mean(error)

logistic_probability = sigmoid(Z_test @ weights + bias)

# 8. KNN: USE THE K SELECTED IN PART 2
with open("outputs/part2_knn_cv.csv", newline="") as f:
    knn_cv_rows = list(csv.DictReader(f))

best_knn_row = min(
    knn_cv_rows,
    key=lambda r: (-float(r["mean_cv_f1"]), int(r["K"]))
)
best_k = int(best_knn_row["K"])

knn_probability = []

for query in Z_test:
    distances = np.sum((Z_train - query) ** 2, axis=1)
    nearest = np.argsort(distances, kind="stable")[:best_k]
    knn_probability.append(float(y_train[nearest].mean()))

knn_probability = np.asarray(knn_probability)

# 9. EVALUATION
def evaluate(predictions):
    predictions = np.asarray(predictions, dtype=int)

    tp = int(np.sum((predictions == 1) & (y_test == 1)))
    fp = int(np.sum((predictions == 1) & (y_test == 0)))
    tn = int(np.sum((predictions == 0) & (y_test == 0)))
    fn = int(np.sum((predictions == 0) & (y_test == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return [tp, fp, tn, fn, precision, recall, f1]

# KNN positive when at least (K+1)/2 neighbours are positive
logistic_pred = (logistic_probability >= 0.3).astype(int)
knn_pred = (
    knn_probability >= (best_k + 1) / (2 * best_k)
).astype(int)
stump_pred = (stump_probability >= 0.3).astype(int)

hard_vote_pred = (
    logistic_pred + knn_pred + stump_pred >= 2
).astype(int)

soft_probability = (
    logistic_probability + knn_probability + stump_probability
) / 3

soft_vote_pred = (soft_probability >= 0.3).astype(int)

voting_rows = []

for name, pred in [
    ("logistic", logistic_pred),
    ("knn", knn_pred),
    ("stump", stump_pred),
    ("hard_vote", hard_vote_pred),
    ("soft_vote", soft_vote_pred)
]:
    voting_rows.append([name] + evaluate(pred))

# 10. ADABOOST: EXACT VALUES FROM ASSIGNMENT
boost_y = np.array([1, 1, 1, -1, -1, -1, 1, -1])

h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

boost_weights = np.ones(8) / 8
adaboost_rows = []
boost_results = {}
alphas = []

for round_number, hypothesis in enumerate([h1, h2], start=1):
    before = boost_weights.copy()

    error = float(np.sum(before[hypothesis != boost_y]))
    alpha = 0.5 * math.log((1-error) / error)

    unnormalised = (
        before * np.exp(-alpha * boost_y * hypothesis)
    )

    boost_weights = unnormalised / np.sum(unnormalised)
    alphas.append(alpha)

    boost_results[f"round{round_number}_error"] = error
    boost_results[f"round{round_number}_alpha"] = alpha

    for i in range(8):
        adaboost_rows.append([
            round_number,
            i + 1,
            before[i],
            unnormalised[i],
            boost_weights[i]
        ])

ensemble_score = alphas[0] * h1 + alphas[1] * h2
ensemble_prediction = np.sign(ensemble_score)
ensemble_accuracy = float(np.mean(ensemble_prediction == boost_y))

# 11. OUT-OF-BAG FRACTION
oob_fraction = (1 - 1/560) ** 560

# 12. MLP PARAMETERS
mlp_rows = [
    ["mlp_hidden1", 784 * 16 + 16],
    ["mlp_hidden2", 16 * 16 + 16],
    ["mlp_output", 16 * 10 + 10]
]
mlp_total = sum(r[1] for r in mlp_rows)

# 13. CNN PARAMETERS
cnn_rows = [
    ["cnn_conv1", 5 * 5 * 1 * 6 + 6],
    ["cnn_conv2", 5 * 5 * 6 * 16 + 16],
    ["cnn_dense1", 400 * 32 + 32],
    ["cnn_dense2", 32 * 10 + 10]
]
cnn_total = sum(r[1] for r in cnn_rows)

# 14. FULLY CONNECTED VS CONVOLUTIONAL LAYER
fc_layer_params = 128 * 128 * 3 * 1024 + 1024
conv_layer_params = 3 * 3 * 3 * 64 + 64

# 15. RNN PARAMETERS
rnn_W_xh = 128 * 64
rnn_W_hh = 128 * 128
rnn_W_hy = 16 * 128
rnn_b_h = 128
rnn_b_y = 16

rnn_total = (
    rnn_W_xh + rnn_W_hh + rnn_W_hy
    + rnn_b_h + rnn_b_y
)

# 16. RESULTS FILE
results = {
    "root_entropy": root_entropy,
    "root_gini": root_gini,
    "best_split_by_info_gain": best_split_by_info_gain,
    "best_split_by_gini_gain": best_split_by_gini_gain,
    "stump_rate_yes": stump_rate_yes,
    "stump_rate_no": stump_rate_no,
    "round1_error": boost_results["round1_error"],
    "round1_alpha": boost_results["round1_alpha"],
    "round2_error": boost_results["round2_error"],
    "round2_alpha": boost_results["round2_alpha"],
    "ensemble_accuracy": ensemble_accuracy,
    "oob_fraction": oob_fraction,
    "mlp_total": mlp_total,
    "cnn_total": cnn_total,
    "fc_layer_params": fc_layer_params,
    "conv_layer_params": conv_layer_params,
    "rnn_W_xh": rnn_W_xh,
    "rnn_W_hh": rnn_W_hh,
    "rnn_W_hy": rnn_W_hy,
    "rnn_b_h": rnn_b_h,
    "rnn_b_y": rnn_b_y,
    "rnn_total": rnn_total
}

with open("outputs/part4_results.txt", "w") as f:
    for key, value in results.items():
        if isinstance(value, (float, np.floating)):
            f.write(f"{key} = {value:.4f}\n")
        else:
            f.write(f"{key} = {value}\n")

# 17. WRITE CSV FILES
def write_csv(path, header, rows, float_columns):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            output_row = []
            for i, value in enumerate(row):
                if i in float_columns:
                    output_row.append(f"{float(value):.4f}")
                else:
                    output_row.append(value)
            writer.writerow(output_row)

write_csv(
    "outputs/part4_splits.csv",
    ["split", "n_yes", "positives_yes", "n_no",
     "positives_no", "info_gain", "gini_gain"],
    split_rows, {5, 6}
)

write_csv(
    "outputs/part4_voting.csv",
    ["model", "tp", "fp", "tn", "fn",
     "precision", "recall", "f1"],
    voting_rows, {5, 6, 7}
)

write_csv(
    "outputs/part4_adaboost.csv",
    ["round", "point", "weight_before",
     "weight_unnormalised", "weight_after"],
    adaboost_rows, {2, 3, 4}
)

write_csv(
    "outputs/part4_architecture.csv",
    ["layer", "parameters"],
    mlp_rows + cnn_rows, set()
)

# 18. DISPLAY RESULTS
print("\nPART 4 COMPLETED")
print("Total result keys:", len(results))
print()

for key, value in results.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key} = {value:.4f}")
    else:
        print(f"{key} = {value}")

print("\nSaved files:")
for filename in [
    "outputs/part4_results.txt",
    "outputs/part4_splits.csv",
    "outputs/part4_voting.csv",
    "outputs/part4_adaboost.csv",
    "outputs/part4_architecture.csv"
]:
    print(filename)
