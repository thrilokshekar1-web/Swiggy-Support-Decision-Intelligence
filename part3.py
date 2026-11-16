import csv
import os
import numpy as np

ROOT = "/content/Swiggy-Support-Decision-Intelligence"
DATA = os.path.join(ROOT, "data", "records.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# Load the dataset
with open(DATA, newline="") as f:
    rows = list(csv.DictReader(f))

X = np.array([
    [float(r["customer_tenure_months"]),
     float(r["earlier_ticket_count_90d"]),
     float(r["sla_consumed_ratio"]),
     float(r["account_value_inr_thousands"]),
     float(r["satisfaction_score"])]
    for r in rows
], dtype=float)

y = np.array([int(r["escalated"]) for r in rows])
value = np.array([float(r["resolution_minutes"]) for r in rows])

n = len(X)
means = X.mean(axis=0)
sds = X.std(axis=0, ddof=1)
Z = (X - means) / sds

results = {}

# --------------------------------------------------
# TASK 3.1: K-MEANS FROM SCRATCH
# --------------------------------------------------
final_labels = {}

for k in range(2, 7):
    step = n // k
    centroids = Z[np.arange(k) * step].copy()
    labels = np.full(n, -1, dtype=int)

    while True:
        distances = np.sum(
            (Z[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2
        )
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels.copy()

        for j in range(k):
            members = Z[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)

    wcss = float(np.sum((Z - centroids[labels]) ** 2))
    results[f"wcss_k{k}"] = wcss

    if k == 3:
        final_labels[3] = labels.copy()

# Elbow rule
elbow_k = None
for k in (3, 4, 5):
    previous_drop = results[f"wcss_k{k-1}"] - results[f"wcss_k{k}"]
    current_drop = results[f"wcss_k{k}"] - results[f"wcss_k{k+1}"]
    if current_drop < 0.5 * previous_drop:
        elbow_k = k
        break

results["elbow_k"] = elbow_k

# --------------------------------------------------
# TASK 3.2: PROFILE THE THREE SEGMENTS
# --------------------------------------------------
labels = final_labels[3]
cluster_info = []

for cluster_id in range(3):
    idx = np.where(labels == cluster_id)[0]
    cluster_info.append({
        "id": cluster_id,
        "indices": idx,
        "mean_x3": float(X[idx, 2].mean())
    })

cluster_info.sort(key=lambda item: item["mean_x3"])

segment_labels = np.empty(n, dtype=object)
segment_rows = []

for position, cluster in enumerate(cluster_info):
    segment = ["A", "B", "C"][position]
    idx = cluster["indices"]
    segment_labels[idx] = segment

    results[f"size_segment_{segment}"] = len(idx)

    segment_rows.append([
        segment,
        len(idx),
        *X[idx].mean(axis=0),
        y[idx].mean(),
        value[idx].mean()
    ])

with open(os.path.join(OUT, "part3_segments.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "segment", "size", "mean_X1", "mean_X2", "mean_X3",
        "mean_X4", "mean_X5", "outcome_rate", "mean_value"
    ])
    for row in segment_rows:
        writer.writerow([
            row[0], row[1],
            *[f"{v:.4f}" for v in row[2:]]
        ])

# --------------------------------------------------
# TASK 3.3: HIERARCHICAL CLUSTERING
# --------------------------------------------------
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

names = ["E1", "E2", "E3", "E4", "E5", "E6"]
hierarchy_rows = []

for linkage in ("single", "complete", "average"):
    clusters = [[i] for i in range(6)]
    heights = []

    for merge_no in range(1, 6):
        best_distance = float("inf")
        best_pair = None

        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                distances = [
                    D[i, j]
                    for i in clusters[a]
                    for j in clusters[b]
                ]

                if linkage == "single":
                    distance = min(distances)
                elif linkage == "complete":
                    distance = max(distances)
                else:
                    distance = sum(distances) / len(distances)

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (a, b)

        a, b = best_pair
        cluster_a = clusters[a]
        cluster_b = clusters[b]

        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        name_a = "+".join(names[i] for i in sorted(cluster_a))
        name_b = "+".join(names[i] for i in sorted(cluster_b))

        hierarchy_rows.append([
            linkage, merge_no, name_a, name_b, best_distance
        ])
        heights.append(best_distance)

        merged = sorted(cluster_a + cluster_b)

        for index in sorted((a, b), reverse=True):
            clusters.pop(index)
        clusters.append(merged)
        clusters.sort(key=lambda c: min(c))

    gaps = np.diff(heights)
    largest_gap_index = int(np.argmax(gaps))
    clusters_at_cut = 6 - (largest_gap_index + 1)

    results[f"{linkage}_clusters_at_cut"] = clusters_at_cut

with open(os.path.join(OUT, "part3_hierarchical.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "linkage", "merge_no", "cluster_a", "cluster_b", "height"
    ])
    for row in hierarchy_rows:
        writer.writerow([
            row[0], row[1], row[2], row[3], f"{row[4]:.4f}"
        ])

# --------------------------------------------------
# TASK 3.4: PCA FROM SCRATCH
# --------------------------------------------------
# Z is already standardised. Centre it again as specified.
Z_centered = Z - Z.mean(axis=0)

covariance = (Z_centered.T @ Z_centered) / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)

order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Apply the required sign convention.
for j in range(eigenvectors.shape[1]):
    largest_index = np.argmax(np.abs(eigenvectors[:, j]))
    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1

eigenvalue_sum = float(eigenvalues.sum())
explained = eigenvalues / eigenvalue_sum
cumulative = np.cumsum(explained)

results["eigenvalue_sum"] = eigenvalue_sum

for j in range(5):
    results[f"eigenvalue_{j+1}"] = eigenvalues[j]
    results[f"explained_ratio_{j+1}"] = explained[j]
    results[f"cumulative_ratio_{j+1}"] = cumulative[j]

results["components_for_80pct"] = int(
    np.argmax(cumulative >= 0.8) + 1
)

W = eigenvectors[:, :2]
projection = Z_centered @ W

for j in range(5):
    results[f"pc1_loading_X{j+1}"] = W[j, 0]
    results[f"pc2_loading_X{j+1}"] = W[j, 1]

for row_index in range(3):
    results[f"row{row_index+1}_pc1"] = projection[row_index, 0]
    results[f"row{row_index+1}_pc2"] = projection[row_index, 1]

for segment in ("A", "B", "C"):
    idx = np.where(segment_labels == segment)[0]
    results[f"mean_pc1_segment_{segment}"] = projection[idx, 0].mean()

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
with open(os.path.join(OUT, "part3_results.txt"), "w") as f:
    for key, val in results.items():
        if isinstance(val, (int, np.integer)):
            f.write(f"{key} = {val}\n")
        else:
            f.write(f"{key} = {float(val):.4f}\n")

print("PART 3 COMPLETED")
print("Total result keys:", len(results))
print()
for key, val in results.items():
    if isinstance(val, (int, np.integer)):
        print(f"{key} = {val}")
    else:
        print(f"{key} = {float(val):.4f}")

print("\nSaved:")
print("outputs/part3_results.txt")
print("outputs/part3_segments.csv")
print("outputs/part3_hierarchical.csv")
