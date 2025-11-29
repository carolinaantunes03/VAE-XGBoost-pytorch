# coding: utf-8

# # Load packages and use PyTorch as backend context

# In[4]:

######################################################
# Install a pip package in the current Jupyter kernel
# import system level packages
#!{sys.executable} -m pip install numpy
#!{sys.executable} -m pip install requests
#!{sys.executable} -m pip install torch torchvision
######################################################
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools # Added for the confusion matrix function

import platform
# REPLACEMENT: PyTorch imports instead of TensorFlow
import torch

########################################################
# importing necessary libraries for scikit-learn

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from scipy import stats
from sklearn import preprocessing

# In[5]:

##################################################
# Test PyTorch (Replacement for TF Session check)
# using kernel that supports GPU computing
# simple test to confirm torch is actually working


# manually set the random seed to define a replication
r_seed = 42

# manually set the number for cross validation
num_cv = 5

print("current random seed is: ", r_seed)


# # check the system information

# In[6]:

#######################################################################################################
# check the system information, check if cuda and gpu computing for PyTorch is installed properly
#######################################################################################################
print(f"Hello, PyTorch! Version: {torch.__version__}")
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")

# Check for Apple Silicon GPU (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nSUCCESS: Apple Metal (MPS) GPU is available!")
    print("PyTorch will use the M1 GPU for tensor operations.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("NVIDIA CUDA GPU is available.") 
else:
    device = torch.device("cpu")
    print("No GPU detected. Using CPU.")

# Simple Tensor Test on GPU
x = torch.ones(5, device=device)
print(f"\nTest Tensor Device: {x.device}")


# # Reading files

# In[7]:

##########################################################
# Reading files/documents                                #
# !!! Need to change to file location on your local drive#
##########################################################

# vae file
compress_path = '../VAE_models/counts_data/vae_compressed_wLabels/encoded_BRCA_VAE_z50_withLabels.txt'

# vae with grade file
# compress_path = 'counts_data/vae_compressed_with_grade/TCGA_4cancers_(BLCA_perSP_minmax_3labels_6LF6k_z200)_with_grade.txt'
# vae with stage file
# compress_path = 'counts_data/vae_compressed_with_stage/TCGA_4cancers_(PAAD_perSP_minmax_3labels_6LF6k_z50)_with_stage.txt'

og_data = pd.read_csv(compress_path, sep="\t", index_col=0)
og_data = og_data.dropna(axis='columns')
# ExprAlldata.columns = ["Gene", "Counts"]
print("dimension of the input data: ", og_data.shape)
og_data.head(5)


# ## Number of cases in each category

# In[8]:

df_count = og_data.groupby('response_group')['Ensembl_ID'].nunique()
print(df_count)
# df_count.nlargest(10)


# In[9]:

###################################################
# store the raw data, and use ensembl id as index
##################################################
df_raw = og_data.iloc[:, 0:]
df_raw = df_raw.set_index('Ensembl_ID')

# notice the last column is the response_group
# df_raw.shape
df_raw.head(3)


# In[10]:

#####################################!#################################
# here begins full data
################################
# full data, 4 labels analysis
# Complete Response    21
# Clinical Progressive Disease    10
# Radiographic Progressive Disease     7
# Stable Disease     7

# features
df_raw_coln = len(df_raw.columns)
X = df_raw.iloc[:, 0:(df_raw_coln - 1)]
X = X.values

# label/target
y = df_raw.loc[:, 'response_group']
y = y.values

# !!!!!!!
# check to confirm the last column is not response group, only y contains response group information
col = X.shape[1]
# print(X[:,(col-1)])

class_names = np.unique(y)
print("unique labels from y: ", class_names)


# # Load necessary methods

# In[11]:

#########################################################################################
# plot confusion matrix
# inputs: cm, confusion matrix from cross_val_predict
#        normalize, whether to use normalize for each sections 
#        title, input the title name for the figure
#        cmap, color map using blue as default
# output: a confusion matrix plot with true label as y axis, and predicted label as x axis
#########################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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


# In[12]:

##############################################################
# plot area under curve graph
# input: actual, true labels/target without one hot encoding
#       probs, predicted probabilities
#       n_classes, number of unique classes in target
#       title, input the title name for the figure
# output: a roc curve plot for multi class task
###############################################################
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle

def plot_multiclass_roc_auc(actual, probs, n_classes, title='multi-class roc'):
    lb = LabelBinarizer()
    lb.fit(actual)
    actual = lb.transform(actual)
    y_prob = probs
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], y_prob[:, i])
        # please notice the difference between auc() and roc_auc_score()
        # also auc() only works on monotonic increasing or monotonic
        # decreasing input x
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    colors = cycle(['blue', 'red', 'green', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.10f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using ' + title)
    plt.legend(loc="lower right")
    # commented thus being able to use fig save function
    # plt.show()


# In[13]:

#######################################################
# Random search CV method
# and
# Multi class roc_auc score method
########################################################
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer, roc_auc_score

###########################################################################################
# Multi class roc_auc score method
# input: y_test, true labels from test fold
#       y_prob, predicted probability on test fold
#       average, string, [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]
##############################################################################################
def multiclass_roc_auc_score(y_test, y_prob, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_prob, average=average)

# !!!
# Notice that by default,needs_proba : boolean, default=False
# thus the multiclass_score will try to use the predicted label instead of predicted probability
multiclass_score = make_scorer(multiclass_roc_auc_score, needs_proba=True)

###############################################################################################
# Binary class roc auc score method
# input: y_true, true labels from test fold
#       y_score, predicted probability on test fold
#############################################################################################
def binary_class_roc_auc_score(y_true, y_score, average="weighted"):
    return roc_auc_score(y_true, y_score, average=average)

binaryclass_score = make_scorer(binary_class_roc_auc_score, needs_threshold=True)

###################################################################################
# Random search CV method
# input: est, input estimator/classifier
#       p_distr, the grid of parameters to search on
#       nbr_iter, numbers of iteration on random search
#       X, feature, y, true labels
# output: ht_estimator, best estimator based on mean value of all folds
#        ht_params, best parameters
################################################################################################
def hypertuning_rscv(est, p_distr, nbr_iter, X, y):
    cv = StratifiedKFold(n_splits=3, random_state=r_seed, shuffle=True)
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, scoring=multiclass_score,
                                   n_jobs=-1, n_iter=nbr_iter, cv=cv, return_train_score=True, verbose=10)
    start = time()
    rdmsearch.fit(X, y)
    print('hyper-tuning time : %d seconds' % (time() - start))
    start = 0
    
    ht_cv_results = rdmsearch.cv_results_
    ht_estimator = rdmsearch.best_estimator_
    ht_params = rdmsearch.best_params_
    
    return ht_estimator, ht_params, ht_cv_results


# # Grid search

# In[14]:

###########################################################
# Grid search Tune learning rate, n_estimators, and booster
##########################################################
param_test_this_loop = {
    'learning_rate': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    'n_estimators': [i for i in range(1, 40)],
    'booster': ['gbtree'],
    # 'booster':['gbtree','gblinear','dart'],
    'verbosity': [0],
    'random_state': [r_seed]
}
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)


gsearch_loop1 = GridSearchCV(estimator=XGBClassifier(booster='gbtree'),
                             param_grid=param_test_this_loop, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=10)
gsearch_loop1.fit(X, y)

print("Best params:", gsearch_loop1.best_params_)
print("Best score:", gsearch_loop1.best_score_)


# In[16]:

#################################################
# Grid search Tune max_depth and min_child_weight
# default
#################################################
param_test_this_loop = {
    'max_depth': [i for i in range(1, 10)],
    'min_child_weight': [i for i in range(0, 10)],
    'verbosity': [0],
    'random_state': [r_seed]
}
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)


gsearch_loop2 = GridSearchCV(estimator=XGBClassifier(learning_rate=gsearch_loop1.best_params_["learning_rate"],
                                                     n_estimators=gsearch_loop1.best_params_["n_estimators"],
                                                     booster=gsearch_loop1.best_params_["booster"]),
                             param_grid=param_test_this_loop, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=10)
gsearch_loop2.fit(X, y)
print("Best params:", gsearch_loop2.best_params_)
print("Best score:", gsearch_loop2.best_score_)


# In[17]:

##########################################
# Grid search Tune subsample and colsample
##########################################
param_test_this_loop = {
    'subsample': [i / 100.0 for i in range(10, 110, 10)],
    'colsample_bytree': [i / 100.0 for i in range(10, 110, 10)],

    'verbosity': [0],
    'random_state': [r_seed]
}
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)


gsearch_loop3 = GridSearchCV(estimator=XGBClassifier(learning_rate=gsearch_loop1.best_params_["learning_rate"],
                                                     n_estimators=gsearch_loop1.best_params_["n_estimators"],
                                                     booster=gsearch_loop1.best_params_["booster"],
                                                     max_depth=gsearch_loop2.best_params_["max_depth"],
                                                     min_child_weight=gsearch_loop2.best_params_["min_child_weight"]),
                             param_grid=param_test_this_loop, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=10)
gsearch_loop3.fit(X, y)
print("Best params:", gsearch_loop3.best_params_)
print("Best score:", gsearch_loop3.best_score_)


# In[18]:

##########################################
# Grid search Tune reg_alpha and reg_lambda
##########################################
param_test_this_loop = {
    'reg_alpha': [i for i in range(0, 3)],
    'reg_lambda': [i for i in range(1, 100)],
    'verbosity': [0],
    'random_state': [r_seed]
}
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)


gsearch_loop4 = GridSearchCV(estimator=XGBClassifier(learning_rate=gsearch_loop1.best_params_["learning_rate"],
                                                     n_estimators=gsearch_loop1.best_params_["n_estimators"],
                                                     booster=gsearch_loop1.best_params_["booster"],
                                                     max_depth=gsearch_loop2.best_params_["max_depth"],
                                                     min_child_weight=gsearch_loop2.best_params_["min_child_weight"],
                                                     subsample=gsearch_loop3.best_params_["subsample"],
                                                     colsample_bytree=gsearch_loop3.best_params_["colsample_bytree"]),
                             param_grid=param_test_this_loop, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=10)
gsearch_loop4.fit(X, y)
print("Best params:", gsearch_loop4.best_params_)
print("Best score:", gsearch_loop4.best_score_)


# # Training the XGBoost model with the best parameters

# In[21]:

###########################
# training a XGBoost model
##########################

# if using GridSearch method
xgb = gsearch_loop4.best_estimator_
cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)

##!!!!
# notice that mean of auroc of each fold is different from the auroc calculated by all the predicted probability
# svm_scores = cross_val_score(svm_model_linear, X, y, cv = cv, scoring=multiclass_score)
y_xgb_prob = cross_val_predict(xgb, X, y, cv=cv, method='predict_proba')

# calculate the auroc by directly using the multiclass_roc_auc_score scorer
# xgb_multiclass_auroc = multiclass_roc_auc_score(y, y_xgb_prob, average="weighted")

# calculate the auroc by directly using the binaryiclass_roc_auc_score scorer
xgb_multiclass_auroc = binary_class_roc_auc_score(y, y_xgb_prob[:, 1], average="weighted")


# In[23]:

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

cv = StratifiedKFold(n_splits=num_cv, random_state=r_seed, shuffle=True)

y_xgb_pred = cross_val_predict(xgb, X, y, cv=cv)
xgb_conf_mat = confusion_matrix(y, y_xgb_pred)


# ## Save and plot feature importance

# In[34]:

# count the importance of features, and see actually how many are useful
print("Number of features have importance greater than zero: ", np.count_nonzero(xgb.feature_importances_))


# In[29]:

#######################################
# plot feature importance
###########################################
import xgboost
# xgboost.plot_importance(xgb)
# plt.rcParams['figure.figsize'] = [10, 30]
# plt.savefig('counts_data/(0806)Feature_Importance(deep10+3L_0.1t_0.2var)(BLCA,seed9).png')
# plt.show()


# ## Print out roc auc figures

# In[30]:

########################################
# print out binary class roc auc figure
############################################
fpr, tpr, threshold = metrics.roc_curve(y, y_xgb_prob[:, 1])
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[37]:

print(xgb)
print("Auroc across all folds: %0.5f" % (xgb_multiclass_auroc))
print("Random seed is: ", r_seed)
print("The confusion martix is:\n", xgb_conf_mat)