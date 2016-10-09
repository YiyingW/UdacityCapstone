'''
Build a random forest model
use cross validation to do hyperparameter tuning and find the best model 
evaluate the best model
'''

import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.cross_validation import ShuffleSplit 
from time import time 
import pickle
import matplotlib.pyplot as plt

'''
class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
random_state=None, verbose=0, warm_start=False, class_weight=None)[source]
'''


# # Load in dataset, split into training, validation and test sets #
pickle_in = open('preprocessed_dataset.pickle', 'rb')
dataset = pickle.load(pickle_in)

train_validation = dataset[dataset['tst']==0]
test = dataset[dataset['tst']==1]

train_validation = train_validation.drop(['tst'], 1)
test = test.drop(['tst'], 1)

X_all = train_validation.drop(['IsBadBuy'], 1)
y_all = train_validation['IsBadBuy']


X_all = np.array(X_all)


X_train, X_validation, y_train, y_validation = cross_validation.train_test_split(X_all, y_all, test_size=0.2)

# # End of loading and spliting data #


CV_sets = ShuffleSplit(X_train.shape[0], n_iter=5, test_size=0.2, random_state=0)

# parameter lists to tune
n_estimators = [10, 20, 30, 50]
max_features = ['auto', 'log2']
min_samples_split = [2, 4]

parameters = {'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split}

# initialize the classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')

# Make an AUC scoring function using 'make_scorer'
def AUC_metric(y_true, y_predict):
	score = roc_auc_score(y_true, y_predict)
	return score

def predict_labels(clf, features, target):
	y_pred = clf.predict(features)
	return roc_auc_score(target.values, y_pred)

auc_scorer = make_scorer(AUC_metric)

start = time()
print 'Start to Train the Model'
# Perform grid search on the classifier using the AUC score as the scoring method
grid_obj = GridSearchCV(rf, param_grid=parameters, scoring=auc_scorer, cv=CV_sets)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
rf = grid_obj.best_estimator_
print rf.get_params()
end = time()

print "The time it takes to find the best model: "
print end - start 

# # Report the final F1 score for training and testing after parameter tuning
# print "Tuned model has a training AUC score of {:.4f}.".format(predict_labels(rf, X_train, y_train))
# print "Tuned model has a testing AUC score of {:.4f}.".format(predict_labels(rf, X_validation, y_validation))


# result = pd.read_csv('data/example_entry.csv')
# result.IsBadBuy = final_test_y
# result.to_csv('rf.csv', index=False)


y_vali_predictions = rf.predict_proba(X_validation)

actual = y_validation
predictions = y_vali_predictions

# Compute ROC curve and ROC area for each class
n_classes = 2
y_score = predictions

y_test = np.array(pd.get_dummies(actual, prefix='y'))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
print fpr 
print tpr

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print "class 0: ", roc_auc[0]
print "class 1: ", roc_auc[1]
##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.3f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()










