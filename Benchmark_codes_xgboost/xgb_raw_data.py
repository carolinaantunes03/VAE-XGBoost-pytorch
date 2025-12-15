
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter
from itertools import cycle
from time import time

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, make_scorer
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import precision_recall_curve


from xgboost import XGBClassifier


# Configuration
r_seed = 42
num_cv = 5
num_runs = 30
np.random.seed(r_seed)

print(f"Random seed: {r_seed}")
print(f"Cross-validation: {num_cv}-fold x {num_runs} runs")


# Load the raw BRCA expression data with labels
raw_path = "../counts_data/counts_data_with_label/TCGA_BRCA_VSTnorm_count_expr_clinical_data.txt"
og_data = pd.read_csv(raw_path, sep="\t", index_col=0)
og_data = og_data.dropna(axis='columns')
print("Raw data dimensions:", og_data.shape)


# Prepare features and labels
df_raw = og_data.set_index("Ensembl_ID")
X = df_raw.iloc[:, :-1].values
y = df_raw["response_group"].values
class_names = np.unique(y)
print("Classes:", class_names)

# Normalize features (same as in VAE preprocessing)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# Define metrics
def binary_class_roc_auc_score(y_true, y_score, average="weighted"):
    return roc_auc_score(y_true, y_score, average=average)

binaryclass_score = make_scorer(binary_class_roc_auc_score, needs_threshold=True)


# Handle class imbalance
counter = Counter(y)
print("Class distribution:", counter)
if len(counter) == 2:
    scale_pos_weight = counter[0] / counter[1]
else:
    scale_pos_weight = 1
print("scale_pos_weight:", scale_pos_weight)


# Hyperparameter tuning (similar to VAE experiment)
param_test_loop1 = {
    'learning_rate': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    'n_estimators': [i for i in range(1, 40)],
    'booster': ['gbtree'],
    'verbosity': [0],
    'random_state': [r_seed]
}

cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)

gsearch_loop1 = GridSearchCV(
    estimator=XGBClassifier(booster='gbtree', scale_pos_weight=scale_pos_weight),
    param_grid=param_test_loop1,
    scoring="roc_auc",
    n_jobs=-1,
    cv=cv,
    verbose=10
)
gsearch_loop1.fit(X, y)

print("Best parameters:", gsearch_loop1.best_params_)
print("Best AUC (CV):", gsearch_loop1.best_score_)



# Repeated CV runs for performance estimation
num_runs = 30
all_auc = []
all_auprc = []
all_conf_mats = []

for run in range(num_runs):
    print(f"\n===== Run {run+1}/{num_runs} =====")
    cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed + run, shuffle=True)

    xgb = XGBClassifier(
        **gsearch_loop1.best_params_,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    # Predict probabilities with cross-validation
    y_xgb_prob = cross_val_predict(xgb, X, y, cv=cv, method='predict_proba', n_jobs=-1)
    y_xgb_pred = np.argmax(y_xgb_prob, axis=1) if y_xgb_prob.shape[1] > 1 else (y_xgb_prob[:, 1] > 0.5).astype(int)

    # ---- Compute AUCROC ----
    auc_val = roc_auc_score(y, y_xgb_prob[:, 1])
    all_auc.append(auc_val)

    # ---- Compute AUPRC ----
    precision, recall, _ = precision_recall_curve(y, y_xgb_prob[:, 1])
    auprc_val = auc(recall, precision)
    all_auprc.append(auprc_val)

    # ---- Confusion matrix ----
    conf_mat = confusion_matrix(y, y_xgb_pred)
    all_conf_mats.append(conf_mat)

    print(f"AUC (Run {run+1}): {auc_val:.4f}")
    print(f"AUPRC (Run {run+1}): {auprc_val:.4f}")

# ---- Compute means and std ----
mean_conf = np.mean(all_conf_mats, axis=0)
print("\n========================================")
print(f"Mean AUC across {num_runs} runs: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
print(f"Mean AUPRC across {num_runs} runs: {np.mean(all_auprc):.4f} ± {np.std(all_auprc):.4f}")
print("Mean Confusion Matrix:")
print(mean_conf.round(2))
print("========================================")

mean_conf = np.mean(all_conf_mats, axis=0)
mean_auprc = np.mean(all_auprc)
std_auprc = np.std(all_auprc)

print("\n======================================")
print(f"Mean AUC across {num_runs} runs: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
print(f"Mean AUPRC across {num_runs} runs: {mean_auprc:.4f}")
print(f"Std AUPRC across {num_runs} runs: {std_auprc:.4f}")
print("Mean Confusion Matrix:")
print(mean_conf.round(2))
print("======================================")


# Plot AUC distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=np.arange(1, num_runs + 1), y=all_auc, palette="viridis")
plt.title(f"AUC per Run ({num_runs} runs of {num_cv}-fold CV, Raw Expression Data)")
plt.xlabel("Run")
plt.ylabel("AUC")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=np.arange(1, num_runs + 1), y=all_auprc, palette='magma')
plt.title(f"AUPRC per Run ({num_runs} runs of {num_cv}-fold CV)")
plt.xlabel("Run")
plt.ylabel("AUPRC")
plt.ylim(0, 1)
plt.show()


# Plot mean confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(mean_conf, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Mean Confusion Matrix – Raw Expression Data")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# Train final model for feature importance extraction
final_xgb = XGBClassifier(
    **gsearch_loop1.best_params_,
    scale_pos_weight=scale_pos_weight,
    max_delta_step=1,
    eval_metric="logloss"
)
final_xgb.fit(X, y)

booster = final_xgb.get_booster()
importance_dict = booster.get_score(importance_type='gain')

importances_df = pd.DataFrame(
    list(importance_dict.items()), columns=['Feature', 'Importance']
).sort_values(by='Importance', ascending=False)

print("\nTop 20 most important raw features (genes):")
print(importances_df.head(20))

# -----------------------------------------
# Plot top features by importance (bar plot)
# -----------------------------------------
TOP_N = 20  

plt.figure(figsize=(8, 6))
sns.barplot(
    data=importances_df.head(TOP_N),
    x="Importance", y="Feature",
    palette="viridis"
)
plt.title(f"Top {TOP_N} Most Important Features")
plt.xlabel("Importance (gain)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


importances_df.to_csv("Ranking_RawData_feature_importance.txt", sep="\t", index=False)
print("Feature importance saved to 'Ranking_RawData_feature_importance.txt'")

