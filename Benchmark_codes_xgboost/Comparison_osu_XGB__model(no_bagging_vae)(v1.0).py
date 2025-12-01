# coding: utf-8

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
compress_path = '../VAE_models/counts_data/vae_compressed_wLabels/encoded_BRCA_VAE_z50_withLabels_pytorch.txt'
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

# --------------------------
# Example: GridSearchCV for XGBoost
param_test_loop1 = {
    'learning_rate': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    'n_estimators': [i for i in range(1, 40)],
    'booster': ['gbtree'],
    'verbosity': [0],
    'random_state': [r_seed]
}
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)
gsearch_loop1 = GridSearchCV(estimator=XGBClassifier(booster='gbtree'),
                             param_grid=param_test_loop1,
                             scoring="roc_auc",
                             n_jobs=-1,
                             cv=cv,
                             verbose=10)
gsearch_loop1.fit(X, y)
print("Best params:", gsearch_loop1.best_params_)
print("Best score:", gsearch_loop1.best_score_)

# --------------------------
# Using best estimator
xgb = gsearch_loop1.best_estimator_
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)

# Predict probabilities and compute AUROC
y_xgb_prob = cross_val_predict(xgb, X, y, cv=cv, method='predict_proba')
xgb_multiclass_auroc = binary_class_roc_auc_score(y, y_xgb_prob[:,1], average="weighted")
y_xgb_pred = cross_val_predict(xgb, X, y, cv=cv)
xgb_conf_mat = confusion_matrix(y, y_xgb_pred)

print(xgb)
print("AUROC across all folds: %0.5f" % xgb_multiclass_auroc)
print("Random seed:", r_seed)
print("Confusion matrix:\n", xgb_conf_mat)

# --------------------------
# Plot ROC for binary classification
fpr, tpr, threshold = roc_curve(y, y_xgb_prob[:,1])
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
