
import csv
import os
import math
import numpy as np

# Load dataset
with open("data/records.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

def column(name):
    return np.array([float(row[name]) for row in rows])

n = len(rows)

# Variables
X1 = column("customer_tenure_months")
X2 = column("earlier_ticket_count_90d")
X3 = column("sla_consumed_ratio")
X4 = column("account_value_inr_thousands")
X5 = column("satisfaction_score")
X6 = column("reopened_flag")
X7 = column("self_service_first_flag")
y = column("resolution_minutes")

results = {}

# Task 1.1: Estimation and confidence interval
mean_y = np.mean(y)
sd_y = np.sqrt(np.sum((y - mean_y)**2) / (n - 1))
se_y = sd_y / np.sqrt(n)

ci_low = mean_y - 1.96 * se_y
ci_high = mean_y + 1.96 * se_y

results["n"] = n
results["mean_value"] = mean_y
results["sd_value"] = sd_y
results["se_value"] = se_y
results["ci95_low"] = ci_low
results["ci95_high"] = ci_high

# Task 1.2: Hypothesis testing
z_stat = (mean_y - 8.0) / se_y
reject = abs(z_stat) > 1.96

results["z_stat"] = z_stat
results["reject_h0"] = "yes" if reject else "no"

# Task 1.3: Required sample size
E = 0.5
n_required = math.ceil((1.96 * sd_y / E)**2)

results["n_required"] = n_required

# Task 1.4: Simple closed-form regression
x_mean = np.mean(X4)
x_sd = np.sqrt(np.sum((X4 - x_mean)**2) / (n - 1))

cov_xy = np.sum((X4 - x_mean) * (y - mean_y)) / (n - 1)
var_x = np.sum((X4 - x_mean)**2) / (n - 1)

slope = cov_xy / var_x
intercept = mean_y - slope * x_mean

fitted = intercept + slope * X4
residuals = y - fitted

tss = np.sum((y - mean_y)**2)
ess = np.sum((fitted - mean_y)**2)
rss = np.sum(residuals**2)

r2 = 1 - rss / tss
rmse = np.sqrt(rss / n)
mae = np.mean(np.abs(residuals))

results["reg_slope"] = slope
results["reg_intercept"] = intercept
results["tss"] = tss
results["ess"] = ess
results["rss"] = rss
results["tss_equals_ess_plus_rss"] = (
    "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
)
results["r2"] = r2
results["rmse"] = rmse
results["mae"] = mae

# Task 1.5: Gradient descent
z = (X4 - x_mean) / x_sd

m = 0.0
c = 0.0
eta = 0.1

def cost(m, c):
    error = y - (m * z + c)
    return np.mean(error**2)

results["gd_J0"] = cost(m, c)

gd_costs = {}

for iteration in range(1, 501):
    error = y - (m * z + c)

    dm = -(2 / n) * np.sum(z * error)
    dc = -(2 / n) * np.sum(error)

    m -= eta * dm
    c -= eta * dc

    if iteration in [1, 10, 100, 500]:
        gd_costs[iteration] = cost(m, c)

results["gd_J1"] = gd_costs[1]
results["gd_J10"] = gd_costs[10]
results["gd_J100"] = gd_costs[100]
results["gd_J500"] = gd_costs[500]

gd_slope = m / x_sd
gd_intercept = c - m * x_mean / x_sd

results["gd_slope"] = gd_slope
results["gd_intercept"] = gd_intercept
results["gd_slope_gap"] = abs(gd_slope - slope)

# Task 1.6: Funnel check
order = np.argsort(fitted, kind="stable")

lower_indices = order[:400]
upper_indices = order[400:]

resid_sd_lower_raw = np.std(residuals[lower_indices], ddof=1)
resid_sd_upper_raw = np.std(residuals[upper_indices], ddof=1)

results["resid_sd_lower_raw"] = resid_sd_lower_raw
results["resid_sd_upper_raw"] = resid_sd_upper_raw
results["resid_sd_ratio_raw"] = (
    resid_sd_upper_raw / resid_sd_lower_raw
)

# Log-log regression
log_x = np.log(X4)
log_y = np.log(y)

log_x_mean = np.mean(log_x)
log_y_mean = np.mean(log_y)

log_slope = (
    np.sum((log_x - log_x_mean) * (log_y - log_y_mean))
    / np.sum((log_x - log_x_mean)**2)
)

log_intercept = log_y_mean - log_slope * log_x_mean

log_fitted = log_intercept + log_slope * log_x
log_residuals = log_y - log_fitted

log_tss = np.sum((log_y - log_y_mean)**2)
log_rss = np.sum(log_residuals**2)

log_r2 = 1 - log_rss / log_tss
log_rmse = np.sqrt(log_rss / n)

results["loglog_intercept"] = log_intercept
results["loglog_slope"] = log_slope
results["loglog_r2"] = log_r2
results["loglog_rmse"] = log_rmse

log_order = np.argsort(log_fitted, kind="stable")

log_lower = log_order[:400]
log_upper = log_order[400:]

resid_sd_lower_log = np.std(
    log_residuals[log_lower], ddof=1
)
resid_sd_upper_log = np.std(
    log_residuals[log_upper], ddof=1
)

results["resid_sd_lower_log"] = resid_sd_lower_log
results["resid_sd_upper_log"] = resid_sd_upper_log
results["resid_sd_ratio_log"] = (
    resid_sd_upper_log / resid_sd_lower_log
)

# Task 1.7: Multiple regression
# Order: constant, ln(X4), X1, X2, X3, X5, X6, X7

design = np.column_stack([
    np.ones(n),
    np.log(X4),
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
])

beta = np.linalg.solve(
    design.T @ design,
    design.T @ log_y
)

mr_fitted = design @ beta
mr_residuals = log_y - mr_fitted

mr_tss = np.sum((log_y - np.mean(log_y))**2)
mr_rss = np.sum(mr_residuals**2)

mr_r2 = 1 - mr_rss / mr_tss
p = 7

mr_adj_r2 = (
    1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
)

loglog_adj_r2 = (
    1 - (1 - log_r2) * (n - 1) / (n - 1 - 1)
)

for i, value in enumerate(beta):
    results[f"mr_beta_{i}"] = value

results["mr_r2"] = mr_r2
results["mr_adj_r2"] = mr_adj_r2
results["loglog_adj_r2"] = loglog_adj_r2

# Save results
os.makedirs("outputs", exist_ok=True)

with open("outputs/part1_results.txt", "w") as f:
    for key, value in results.items():
        if isinstance(value, str):
            f.write(f"{key} = {value}\n")
        elif isinstance(value, (int, np.integer)):
            f.write(f"{key} = {value}\n")
        else:
            f.write(f"{key} = {value:.4f}\n")

print("PART 1 COMPLETED SUCCESSFULLY")
print("Total result keys:", len(results))
print("\nRESULTS:\n")

for key, value in results.items():
    if isinstance(value, str):
        print(f"{key} = {value}")
    elif isinstance(value, (int, np.integer)):
        print(f"{key} = {value}")
    else:
        print(f"{key} = {value:.4f}")

print("\nSaved: outputs/part1_results.txt")
