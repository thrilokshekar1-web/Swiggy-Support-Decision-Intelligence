Domain: Support

## How to run

The project uses Python and NumPy. From the repository root, install the dependency and run the scripts in order:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The scripts write their result files and CSV outputs into the `outputs/` directory. The supplied data generator must remain unchanged.

## Part 1 — Statistical analysis and regression

### R1.1 — Confidence interval and hypothesis test

The dataset contains 800 records. The sample mean is 8.9382, with a standard deviation of 9.1723 and standard error of 0.3243. The 95% confidence interval is [8.3026, 9.5738]. The calculated z-statistic is 2.8930, so the null hypothesis is rejected at the 5% significance level. The required sample size for the specified precision is 1293, which is larger than the available 800 records.

### R1.2 — Linear regression and error decomposition

The fitted linear regression has slope 0.6701 and intercept -3.6598. Its R² is 0.5196, RMSE is 6.3537, and MAE is 3.7636. The total sum of squares is 67221.4285, explained sum of squares is 34925.6088, and residual sum of squares is 32295.8197. The identity TSS = ESS + RSS holds. The model explains approximately 51.96% of the observed variation.

### R1.3 — Gradient descent

Gradient descent reduces the objective from 163.9178 at iteration 0 to 40.3698 at iteration 500. The estimated slope and intercept agree with the analytical regression solution; the slope gap is 0.0000. This indicates convergence to the same fitted parameters.

### R1.4 — Residual analysis

For the raw-scale model, the ratio of the upper-half residual standard deviation to the lower-half value is 3.7525, indicating substantial variation between the two groups. For the log-log model, the ratio is 1.0396, much closer to 1. The log transformation therefore produces more balanced residual spread in this comparison.

### R1.5 — Multiple regression comparison

The multiple regression model achieves R² = 0.7425 and adjusted R² = 0.7403. The log-log model has adjusted R² = 0.6561. The multiple regression therefore explains more variation after accounting for model complexity in this dataset. Its coefficients are reported in `part1_results.txt`.

## Part 2 — Classification models

### R2.1 — Learning-rate comparison

After 200 iterations, the losses are 0.5091 for learning rate 0.01, 0.3776 for 0.1, and 0.3703 for 1.0. The larger learning rates make faster progress in this experiment. A rate that is too small can converge slowly, while an excessively large rate can overshoot or become unstable. The tested rate of 1.0 gives the lowest loss among these three experiments.

### R2.2 — Logistic regression interpretation

The three largest absolute standardized logistic coefficients are X3 (0.9013), X1 (0.5554), and X2 (0.5505). X3 has the largest positive odds ratio, 2.4629. A one-standard-deviation increase in X3 is associated with approximately 2.46 times the odds of the positive outcome, holding the other predictors constant.

### R2.3 — Threshold and class-imbalance effects

The blind majority-class accuracy is 0.7875, while logistic regression at threshold 0.5 achieves 0.8042. The dataset is imbalanced, so accuracy alone does not describe positive-class detection well. Reducing the threshold to 0.3 increases recall from 0.4706 to 0.7451, while precision decreases from 0.5455 to 0.4634. This reflects the trade-off between finding more positive cases and generating more false positives.

### R2.4 — Logistic regression versus KNN

At threshold 0.3, logistic regression achieves F1 = 0.5714 and recall = 0.7451. The selected KNN model (K = 5) achieves F1 = 0.3696 and recall = 0.3333. Logistic regression performs better on both measures on the test set.

### R2.5 — Recommended threshold

For the specified cost rule, the selected training threshold is 0.15. Its training cost is 12150 and test cost is 6050. By comparison, logistic regression at threshold 0.5 has test cost 11800, while flagging nobody has cost 20400. The threshold is selected using training probabilities to avoid choosing it based on test-set outcomes. Logistic regression at 0.15 is the cost-oriented operating point for this experiment.

## Part 3 — Clustering and dimensionality reduction

### R3.1 — K-means and elbow selection

The WCSS values for K = 2 through 6 are 2262.6753, 1602.8471, 1494.7732, 1323.7157, and 1278.1691. The successive reductions are 659.8282, 108.0739, 171.0575, and 45.5466. The reduction from K = 3 to K = 4 is less than half the reduction from K = 2 to K = 3. The specified elbow rule therefore selects K = 3.

### R3.2 — Segment interpretation

The segments are ordered by increasing mean X3:

| Segment | Size | Mean X3 | Outcome rate | Mean value |
|---|---:|---:|---:|---:|
| A | 322 | 0.2250 | 1.86% | 3.5976 |
| B | 255 | 0.4534 | 23.53% | 17.1248 |
| C | 223 | 0.7118 | 46.64% | 7.2883 |

Segment A has the lowest outcome rate and the longest average tenure (47.5028). It is a candidate for routine support and retention activities. Segment B has the highest mean account value (28.7192) and a 23.53% outcome rate, so proactive support may be appropriate. Segment C has the highest mean X3 (0.7118), lowest satisfaction (38.5215), and highest outcome rate (46.64%). It should receive priority for intervention and escalation review.

### R3.3 — Hierarchical clustering

At the largest-gap cut, single linkage produces 4 clusters, while complete and average linkage each produce 2. Single linkage can join groups through short pairwise connections, creating chaining. Complete linkage considers the farthest pair, so it tends to keep groups more compact. Average linkage uses average pairwise distance and gives an intermediate criterion. These different distance definitions explain the different cuts.

### R3.4 — PCA interpretation

The first principal component explains 52.27% of variance and the second explains 23.66%. Three components are needed to exceed 80% cumulative explained variance (86.16%). PC1 has its largest positive loading on X1 (0.5317) and largest-magnitude negative loading on X3 (-0.5187), with X5 also contributing positively (0.4418). The mean PC1 scores are 1.7773 for A, -0.7831 for B, and -1.6709 for C. This separates the segments along a direction strongly associated with tenure and SLA consumption.

## Part 4 — Decision trees, ensembles, and neural-network sizing

### R4.1 — Decision-tree split

The root entropy is 0.7462 and root Gini impurity is 0.3347. Split S4 (X1 <= 24) gives the highest information gain and Gini gain among the four candidates. The resulting stump has positive rates of 0.4574 on the yes branch and 0.0887 on the no branch. A deeper tree could overfit a small dataset by creating narrow groups that capture noise. Limiting depth, requiring minimum samples per leaf, and pruning can help control this risk.

### R4.2 — Model comparison and voting

On the test set, logistic regression has F1 = 0.5714 and recall = 0.7451. The decision stump has F1 = 0.5532 and recall = 0.7647. KNN has F1 = 0.3696 and recall = 0.3333. Hard voting achieves F1 = 0.5344, while soft voting achieves F1 = 0.5507. Combining models does not automatically improve performance: the voting results do not exceed logistic regression's F1 in this experiment.

### R4.3 — AdaBoost weight update

The first-round error is 0.2500 and alpha is 0.5493. After the first update, misclassified points 3 and 7 each have normalized weight 0.2500, for a combined weight of 0.5000. Correctly classified points each have weight 0.0833. The second-round error is 0.1667 and alpha is 0.8047. Since the second alpha is larger, the second weak learner receives greater influence in the ensemble. This reflects its lower weighted error.

### R4.4 — Neural-network parameter sizing

**MLP architecture**

| Layer | Shape | Parameters |
|---|---|---:|
| Input to hidden 1 | 784 → 16 | 12560 |
| Hidden 1 to hidden 2 | 16 → 16 | 272 |
| Hidden 2 to output | 16 → 10 | 170 |
| **Total** | | **13002** |

**Model A — CNN option 2**

For a 28 × 28 × 1 input, the layer shapes and parameter counts are:

| Layer | Output shape | Parameters |
|---|---|---:|
| Conv1 (5 × 5, 6 filters, padding 2) | 28 × 28 × 6 | 156 |
| MaxPool1 (2 × 2) | 14 × 14 × 6 | 0 |
| Conv2 (5 × 5, 16 filters, no padding) | 10 × 10 × 16 | 2416 |
| MaxPool2 (2 × 2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 | 400 → 32 | 12832 |
| Dense2 | 32 → 10 | 330 |
| **Total** | | **15734** |

For 10-class classification, a typical output uses softmax with categorical cross-entropy (or logits with cross-entropy). ReLU is commonly used in hidden layers. Deep networks can suffer from vanishing gradients; skip connections provide shorter paths for gradient flow.

**Fully connected versus convolutional layer**

For an input of shape 128 × 128 × 3 and 1024 output units, the fully connected layer has 50,332,672 parameters. A convolutional layer with 64 filters of size 3 × 3 has 1,792 parameters. Convolution uses local connectivity and shared weights, making it substantially more parameter-efficient for image data.

**RNN parameter shapes**

For input size 64, hidden size 128, and output size 16:

- W_xh: 128 × 64 = 8192 parameters
- W_hh: 128 × 128 = 16384 parameters
- W_hy: 16 × 128 = 2048 parameters
- b_h: 128 parameters
- b_y: 16 parameters
- Total: 26768 parameters

The recurrent weights are shared across time steps, so the parameter count does not increase with sequence length.

## Model-choice memo

**To: Support Operations Head**

I recommend using logistic regression as the first structured-data decision model, with a threshold selected according to the operational cost rule. In this experiment, the selected threshold is 0.15, giving a test cost of 6050 compared with 11800 at threshold 0.5 and 20400 when no cases are flagged. At threshold 0.3, logistic regression achieves F1 = 0.5714 and recall = 0.7451, outperforming KNN (F1 = 0.3696, recall = 0.3333). The decision stump achieves slightly higher recall (0.7647) but lower F1 (0.5532). Voting does not improve F1 over logistic regression.

The threshold should be monitored using operational feedback because false negatives and false positives have different costs. These results are based on a generated dataset and should not be treated as proof of production performance. Validate on real support records before deployment.

Model A (CNN) and Model B (RNN) should not be prioritized ahead of the structured-data baseline. Model A is relevant if image data becomes available, while Model B is relevant if sequential support data becomes available. Either should be built only if a separate evaluation demonstrates operational value.

## Limitations

1. The dataset is generated and may not represent the distribution or complexity of real customer-support cases.
2. The test set contains only 240 records and 51 positive outcomes, so performance estimates may vary with a different sample.
3. The classification threshold depends on the assumed false-positive and false-negative costs. Actual business costs may differ.
4. Correlation and regression results do not establish causal relationships between predictors and outcomes.
5. Clustering and PCA are descriptive methods; their segments and components require validation with operational stakeholders.
6. The neural-network sections calculate parameter counts and architecture shapes; they do not demonstrate trained neural-network performance on this dataset.
7. Models should be monitored for data drift, changing customer behaviour, and unequal error rates across relevant customer groups.
