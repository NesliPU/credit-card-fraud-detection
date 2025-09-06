# Credit Card Fraud Detection Project

This is a small project where I tried to build a model that can spot fraudulent credit card transactions. The dataset comes from Kaggle
and contains ~285,000 transactions, out of which only 492 are frauds (~0.17%).

That huge imbalance makes the problem interesting: the model has to learn how to catch rare frauds.

  - In the data set, there are:
    - About 285k rows (one for each transaction)
    - 31 columns
        - PCA-anonymized features labeled as V1-V28
        - Time and Amount
        - Target (Class, 0= legit and 1=fraud)
# First Act
  - Initial model: Logistic regression: For legit transactions, it worked almost at 100% accuracy (with a 0.5 threshold), for fraud, it did poorly at 67%. Accuracy looked perfect, but it was misleading as the data is imbalanced towards the legitimate cases.
  - In order to handle the class imbalance, I added class_weight="balanced", which assigned a higher weight to fraud, i.e., every time it misclassifies a fraud, the loss/error is multiplied by a bigger weight. This increased the fraud detection to ~92%, but the precision dropped due to increased false alarms.
  - Random Forest model: Trained a random forest with 200 trees. I obtained high recall and improved precision compared to weighted logistic regression, with an AUC of around 0.98.
  - Feature Importance: Random Forest highlighted V14, V12, V17, and Amount as the most important signals for detecting fraud.
  
  - Results
      - ROC Curves
  ![Alt text](images/roc_curve.png)
      - Feature Importance
  ![Alt text](images/feature_importance.png)

- What I learned:
  - Accuracy is useless on imbalanced data. Precision, recall, and AUC tell the real story.
  - Logistic Regression is a solid baseline, but it struggles with rare classes.
  - Random Forests are much more powerful here, and feature importances help explain what dves fraud detection.
  - There’s always a trade-off: higher recall means catching more frauds but annoying more customers with false alarms.
    
- Tech Stack
   - Python, pandas, scikit-learn, matplotlib
   - Models: Logistic Regression, Random Forest
   - Evaluation: Precision, Recall, F1-score, ROC/AUC

# Second Act:

- Fine-tuning RF model: I tuned a Random Forest using `GridSearchCV` with a stratified 5-fold split (keeps the fraud/legit ratio consistent in each fold).
- **Base model:** `RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)`
- **Search space (16 combos; 5 folds → 80 fits):**
  - `n_estimators`: [100, 200]
  - `max_depth`: [8, 12]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
  - `max_features`: ["sqrt"]
- **Scoring:** optimized for **ROC AUC**; also tracked **Average Precision (PR-AUC)**.
- After the search, `GridSearchCV` refit the best model on the full training set and I evaluated on the held-out test set.

### Why this matters
- Grid search removes guesswork by testing all parameter combinations on multiple folds.
- Stratified CV avoids “unlucky” splits with too few fraud cases.
- Reporting both ROC AUC and PR-AUC is important for highly imbalanced data.

## Results (Tuned Random Forest)

- Test **ROC AUC**: ~0.98  
- Test **PR-AUC (Average Precision)**: ~0.80  
- Full precision/recall/F1 by class is printed in the notebook’s `classification_report`.

Plots saved in `images/`:
- ROC Curve — Tuned Random Forest  
  `images/roc_curve_best_rf.png`  
  ![ROC Curve — Tuned Random Forest](images/roc_curve_best_rf.png)

- Precision–Recall Curve — Tuned Random Forest  
  `images/precision_recall_best_rf.png`  
  ![Precision–Recall Curve — Tuned Random Forest](images/precision_recall_best_rf.png)

### Interpretation (brief)
- **ROC AUC ~0.98** indicates excellent ranking ability (fraud vs. non-fraud).
- **PR-AUC ~0.80** shows strong performance on the rare positive class (fraud), which is the metric that matters most when positives are scarce.

### How to reproduce
1. Run the “Fine-tuning RF (GridSearchCV)” cell in the notebook.  
2. The cell prints the best hyperparameters and test metrics.  
3. It also writes the plots above to `images/` so they render in this README.

### Cost-sensitive Thresholding 
A false-negative (FN) is costly, and a false-positive (FP) is not preferable in the custormer-front with too many false fraud alerts.
I assigned cost of a FN=100 and FP=5, then let the model run between the 0.0-1.0 threshold range to find the optimal value. 

### Results
- **Best threshold:** 0.47  
- **Minimum cost:** 1,545  
- **Confusion matrix at this threshold:** [[56815 49] [ 13 85]]
This means:
- 85 frauds were correctly detected,  
- only 13 frauds were missed,  
- 49 legitimate transactions were falsely flagged,  
- 56,815 legitimate transactions were correctly passed through.
- 
### Cost vs. Threshold Plot
The plot below shows how total business cost changes with the decision threshold, with the optimal point marked in red.

![Cost vs Threshold](images/cost_vs_threshold.png)

### Why this matters
Instead of using the default 0.5 cutoff, cost-sensitive tuning shows how to choose a threshold that **minimizes financial loss** under realistic assumptions.  
This illustrates that the model is not only accurate but also *aligned with business objectives*.

### Business Impact
By lowering the decision threshold from 0.50 to 0.47, the tuned Random Forest reduced the expected fraud cost from over 2,000 to 1,545 — a **~25% cost reduction** under realistic assumptions (missed fraud = 100, false alarm = 5).

### Executive Summary
The tuned Random Forest is not just a strong model statistically (ROC AUC ~0.98, PR-AUC ~0.80), but also operationally valuable.  
By explicitly weighing the higher cost of missed fraud against the smaller cost of false alarms, the model strikes a balance that would save money in production while catching 87% of fraud cases.  
This demonstrates how data science decisions translate into business outcomes.
