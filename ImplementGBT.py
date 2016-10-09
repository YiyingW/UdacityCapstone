from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.cross_validation import ShuffleSplit 
from time import time 
import pickle
import matplotlib.pyplot as plt 



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


# # initialize the classifier

loss = 'deviance'
n_estimators = 100
learning_rate = 0.1
min_samples_split = 4
max_depth = 5

print "Start to Train"
classifier = GradientBoostingClassifier(loss=loss, n_estimators=n_estimators,learning_rate=learning_rate, min_samples_split=min_samples_split,max_depth=max_depth)  # class_weight set to balanced greatly improved performance

classifier.fit(X_train, y_train)

y_vali_predictions = classifier.predict_proba(X_validation)

actual = y_validation
predictions = y_vali_predictions



# Compute ROC curve and ROC area for each class
n_classes = 2
y_score = predictions
print y_score[:10]
y_test = np.array(pd.get_dummies(actual, prefix='y'))
print y_test[:10]
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
