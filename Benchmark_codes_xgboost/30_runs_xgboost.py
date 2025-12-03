import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve


# --------------------------
# Random seed
r_seed = 42
np.random.seed(r_seed)
torch.manual_seed(r_seed)

# Number of CV folds
num_cv = 5

print("Current random seed:", r_seed)
print("PyTorch version:", torch.__version__)
print("Python version:", sys.version)

# --------------------------
# Reading the VAE-compressed dataset
compress_path = '../VAE_models/counts_data/vae_compressed_wLabels/encoded_BRCA_VAE_z50_withLabels_pytorch_exp2.txt'
og_data = pd.read_csv(compress_path, sep="\t", index_col=0)
og_data = og_data.dropna(axis='columns')
print("Dimension of input data:", og_data.shape)

# --------------------------
# Raw dataframe
df_raw = og_data.set_index('Ensembl_ID')

# Features and labels
X = df_raw.iloc[:, :-1].values
y = df_raw['response_group'].values
class_names = np.unique(y)
print("Unique labels from y:", class_names)

# --------------------------
# Confusion matrix plotting function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# --------------------------
# Multi-class ROC-AUC plotting
def plot_multiclass_roc_auc(actual, probs, n_classes, title='multi-class roc'):
    lb = LabelBinarizer()
    lb.fit(actual)
    actual = lb.transform(actual)
    y_prob = probs
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    colors = cycle(['blue', 'red', 'green', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.10f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using ' + title)
    plt.legend(loc="lower right")

# --------------------------
# Multi-class and binary-class ROC-AUC scoring
def multiclass_roc_auc_score(y_test, y_prob, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_prob, average=average)

def binary_class_roc_auc_score(y_true, y_score, average="weighted"):
    return roc_auc_score(y_true, y_score, average=average)

multiclass_score = make_scorer(multiclass_roc_auc_score, needs_proba=True)
binaryclass_score = make_scorer(binary_class_roc_auc_score, needs_threshold=True)

# --------------------------
# Hyperparameter tuning using RandomizedSearchCV
def hypertuning_rscv(est, p_distr, nbr_iter, X, y):
    cv = StratifiedKFold(n_splits=3, random_state=r_seed, shuffle=True)
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, scoring=multiclass_score,
                                   n_jobs=-1, n_iter=nbr_iter, cv=cv, return_train_score=True, verbose=10)
    start = time()
    rdmsearch.fit(X, y)
    print('hyper-tuning time : %d seconds' % (time() - start))
    ht_cv_results = rdmsearch.cv_results_
    ht_estimator = rdmsearch.best_estimator_
    ht_params = rdmsearch.best_params_
    return ht_estimator, ht_params, ht_cv_results

from collections import Counter

# --------------------------
# Handle class imbalance
counter = Counter(y)
print("Class distribution:", counter)
if len(counter) == 2:
    scale_pos_weight = counter[0] / counter[1]
    print("Applying scale_pos_weight:", scale_pos_weight)
else:
    scale_pos_weight = 1
    print("Non-binary labels detected, no scale_pos_weight used.")


# --------------------------
# Example: GridSearchCV for XGBoost with scale_pos_weight
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
print("Best params:", gsearch_loop1.best_params_)
print("Best score:", gsearch_loop1.best_score_)

# --------------------------
# 30 runs of 5-fold CV
num_runs = 30
all_auc = []
all_conf_mats = []

for run in range(num_runs):
    print(f"\n===== Run {run+1}/{num_runs} =====")
    cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed + run, shuffle=True)

    xgb = XGBClassifier(
        **gsearch_loop1.best_params_,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    y_xgb_prob = cross_val_predict(xgb, X, y, cv=cv, method='predict_proba', n_jobs=-1)
    y_xgb_pred = np.argmax(y_xgb_prob, axis=1) if y_xgb_prob.shape[1] > 1 else (y_xgb_prob[:,1] > 0.5).astype(int)

    auc_val = roc_auc_score(y, y_xgb_prob[:,1])


    conf_mat = confusion_matrix(y, y_xgb_pred)
    
    all_auc.append(auc_val)
    all_conf_mats.append(conf_mat)

    print(f"AUC (Run {run+1}): {auc_val:.4f}")

# --------------------------
# Compute mean confusion matrix (average of all runs)
mean_conf = np.mean(all_conf_mats, axis=0)

# Summary results
print("\n========================================")
print(f"Mean AUC across {num_runs} runs: {np.mean(all_auc):.4f}")
print(f"Std AUC across {num_runs} runs: {np.std(all_auc):.4f}")
print("Mean Confusion Matrix:")
print(mean_conf.round(2))
print("========================================")

# --------------------------
# Plot bar plot of AUC for each run
plt.figure(figsize=(10, 5))
sns.barplot(x=np.arange(1, num_runs + 1), y=all_auc, palette='viridis')
plt.title(f"AUC per Run ({num_runs} runs of {num_cv}-fold CV)")
plt.xlabel("Run")
plt.ylabel("AUC")
plt.ylim(0, 1)
plt.show()

# --------------------------
# Plot heatmap of mean confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(mean_conf, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Mean Confusion Matrix over {num_runs} runs")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# --------------------------
# Feature importance extraction from the final trained model
# --------------------------

# Fit one final XGBoost model on the full dataset (with best params)
final_xgb = XGBClassifier(
    **gsearch_loop1.best_params_,
    scale_pos_weight=scale_pos_weight,
    max_delta_step=1,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_xgb.fit(X, y)

# Extract feature importances (gain = how much a feature improves splits)
booster = final_xgb.get_booster()
importance_dict = booster.get_score(importance_type='gain')

# Convert to DataFrame
importances_df = pd.DataFrame(
    list(importance_dict.items()), columns=['Feature', 'Importance']
).sort_values(by='Importance', ascending=False)

# Print top 20
print("\nTop 20 most important latent features (from XGBoost):")
print(importances_df.head(20))

# Save to file for reference or plotting
importances_df.to_csv(
    "Ranking_VAE_feature_importance.txt",
    sep="\t", index=False
)
print("\nFeature importance table saved as 'Ranking_VAE_feature_importance.txt'")

