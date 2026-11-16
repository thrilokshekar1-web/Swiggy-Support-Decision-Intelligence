
import csv
import os
import math
import numpy as np

os.makedirs("outputs", exist_ok=True)

# Load dataset
with open("data/records.csv", "r") as f:
    rows = list(csv.DictReader(f))

def get_col(name):
    return np.array([float(r[name]) for r in rows])

X = np.column_stack([
    get_col("customer_tenure_months"),
    get_col("earlier_ticket_count_90d"),
    get_col("sla_consumed_ratio"),
    get_col("account_value_inr_thousands"),
    get_col("satisfaction_score"),
    get_col("reopened_flag"),
    get_col("self_service_first_flag")
])

y = get_col("escalated")
n = len(y)

# --------------------------------------------------
# 2.1 STRATIFIED 70:30 SPLIT
# --------------------------------------------------

train_indices = []
test_indices = []

for cls in [0, 1]:
    indices = np.where(y == cls)[0]
    cut = (7 * len(indices)) // 10

    train_indices.extend(indices[:cut])
    test_indices.extend(indices[cut:])

train_indices = np.sort(train_indices)
test_indices = np.sort(test_indices)

X_train_raw = X[train_indices]
X_test_raw = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

train_mean = np.mean(X_train_raw, axis=0)
train_sd = np.std(X_train_raw, axis=0, ddof=1)

X_train = (X_train_raw - train_mean) / train_sd
X_test = (X_test_raw - train_mean) / train_sd

results = {}

results["train_rows"] = len(y_train)
results["train_positives"] = int(np.sum(y_train))
results["test_rows"] = len(y_test)
results["test_positives"] = int(np.sum(y_test))

# --------------------------------------------------
# LOGISTIC REGRESSION FUNCTIONS
# --------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


def train_logistic(X, y, eta, updates, record_epochs=None):
    w = np.zeros(X.shape[1])
    b = 0.0

    losses = {}

    if record_epochs is not None and 0 in record_epochs:
        losses[0] = bce(y, sigmoid(X @ w + b))

    for epoch in range(1, updates + 1):
        probabilities = sigmoid(X @ w + b)
        error = probabilities - y

        dw = (X.T @ error) / len(y)
        db = np.mean(error)

        w -= eta * dw
        b -= eta * db

        if record_epochs is not None and epoch in record_epochs:
            losses[epoch] = bce(y, sigmoid(X @ w + b))

    return w, b, losses


# --------------------------------------------------
# 2.2 MAIN LOGISTIC MODEL
# --------------------------------------------------

epochs = [0, 1, 10, 100, 1000]

w, b, losses = train_logistic(
    X_train, y_train, 0.1, 1000, epochs
)

for epoch in epochs:
    results[f"loss_epoch_{epoch}"] = losses[epoch]

for i, value in enumerate(w, start=1):
    results[f"weight_X{i}"] = value

results["bias"] = b

train_prob = sigmoid(X_train @ w + b)
test_prob = sigmoid(X_test @ w + b)

results["test_bce"] = bce(y_test, test_prob)

# --------------------------------------------------
# 2.3 LEARNING RATE EFFECT
# --------------------------------------------------

for eta in [0.01, 0.1, 1.0]:
    _, _, lr_losses = train_logistic(
        X_train, y_train, eta, 200, [200]
    )

    results[f"loss_epoch_200_eta_{eta}"] = lr_losses[200]

# --------------------------------------------------
# 2.4 ODDS RATIOS
# --------------------------------------------------

for i, value in enumerate(w, start=1):
    results[f"odds_ratio_X{i}"] = math.exp(value)

# --------------------------------------------------
# CONFUSION MATRIX AND METRICS
# --------------------------------------------------

def evaluate(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)

    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


# --------------------------------------------------
# 2.5 TWO THRESHOLDS
# --------------------------------------------------

for tau in [0.5, 0.3]:
    metrics = evaluate(y_test, test_prob, tau)

    for key, value in metrics.items():
        results[f"tau_{tau}_{key}"] = value

results["blind_accuracy"] = np.mean(y_test == 0)

# --------------------------------------------------
# 2.6 THRESHOLD SWEEP
# --------------------------------------------------

threshold_rows = []

for tau in np.arange(0.1, 1.0, 0.1):
    metrics = evaluate(y_test, test_prob, round(tau, 1))

    tpr = metrics["recall"]
    fpr = (
        metrics["fp"] / (metrics["fp"] + metrics["tn"])
        if metrics["fp"] + metrics["tn"] else 0
    )

    threshold_rows.append([
        round(tau, 1),
        metrics["tp"],
        metrics["fp"],
        metrics["tn"],
        metrics["fn"],
        tpr,
        fpr
    ])

with open("outputs/part2_thresholds.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])

    for row in threshold_rows:
        writer.writerow([
            row[0], row[1], row[2], row[3], row[4],
            f"{row[5]:.4f}", f"{row[6]:.4f}"
        ])

# --------------------------------------------------
# 2.7 K-NEAREST NEIGHBOURS
# --------------------------------------------------

def knn_predict(X_reference, y_reference, X_query, k):
    predictions = []
    probabilities = []

    for query in X_query:
        distances = np.sqrt(
            np.sum((X_reference - query)**2, axis=1)
        )

        order = np.argsort(distances, kind="stable")
        neighbours = order[:k]

        positive_count = int(np.sum(y_reference[neighbours]))
        probability = positive_count / k

        probabilities.append(probability)
        predictions.append(
            1 if positive_count >= (k + 1) // 2 else 0
        )

    return np.array(predictions), np.array(probabilities)


def f1_score(y_true, predictions):
    tp = np.sum((y_true == 1) & (predictions == 1))
    fp = np.sum((y_true == 0) & (predictions == 1))
    fn = np.sum((y_true == 1) & (predictions == 0))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )


knn_rows = []

for k in range(1, 22, 2):
    fold_scores = []

    positions = np.arange(len(y_train))

    for fold in range(5):
        validation = positions[positions % 5 == fold]
        reference = positions[positions % 5 != fold]

        predictions, _ = knn_predict(
            X_train[reference],
            y_train[reference],
            X_train[validation],
            k
        )

        score = f1_score(y_train[validation], predictions)
        fold_scores.append(score)

    mean_f1 = np.mean(fold_scores)
    knn_rows.append([k, mean_f1])

best_k = max(knn_rows, key=lambda item: (item[1], -item[0]))[0]

with open("outputs/part2_knn_cv.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "mean_cv_f1"])

    for k, score in knn_rows:
        writer.writerow([k, f"{score:.4f}"])

results["knn_best_k"] = best_k

knn_predictions, knn_probabilities = knn_predict(
    X_train, y_train, X_test, best_k
)

knn_metrics = evaluate(y_test, knn_probabilities, 0.5)

for key, value in knn_metrics.items():
    if key != "specificity":
        results[f"knn_{key}"] = value

# --------------------------------------------------
# 2.8 COST-BASED THRESHOLD
# --------------------------------------------------

def calculate_cost(y_true, probabilities, threshold):
    metrics = evaluate(y_true, probabilities, threshold)
    return 50 * metrics["fp"] + 400 * metrics["fn"]


best_tau = None
best_cost = float("inf")

for k in range(1, 20):
    tau = k / 20
    cost = calculate_cost(y_train, train_prob, tau)

    if cost < best_cost:
        best_cost = cost
        best_tau = tau

results["cost_tau"] = best_tau
results["cost_train_at_tau"] = best_cost
results["cost_test_at_tau"] = calculate_cost(
    y_test, test_prob, best_tau
)
results["cost_test_at_0.5"] = calculate_cost(
    y_test, test_prob, 0.5
)
results["cost_test_flag_nobody"] = 400 * int(np.sum(y_test))

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

with open("outputs/part2_results.txt", "w") as f:
    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            f.write(f"{key} = {value}\n")

        elif isinstance(value, str):
            f.write(f"{key} = {value}\n")

        else:
            f.write(f"{key} = {value:.4f}\n")

print("PART 2 COMPLETED SUCCESSFULLY")
print("Total result keys:", len(results))

print("\nRESULTS:\n")

for key, value in results.items():
    if isinstance(value, (int, np.integer)):
        print(f"{key} = {value}")
    elif isinstance(value, str):
        print(f"{key} = {value}")
    else:
        print(f"{key} = {value:.4f}")

print("\nSaved files:")
print("outputs/part2_results.txt")
print("outputs/part2_thresholds.csv")
print("outputs/part2_knn_cv.csv")
