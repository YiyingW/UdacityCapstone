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
pickle_in.close()

train_validation = dataset[dataset['tst']==0]
test = dataset[dataset['tst']==1]

train_validation = train_validation.drop(['tst'], 1)
test = test.drop(['tst', 'IsBadBuy'], 1)
test = np.array(test)

X_all = train_validation.drop(['IsBadBuy'], 1)
y_all = train_validation['IsBadBuy']


X_all = np.array(X_all)


X_train, X_validation, y_train, y_validation = cross_validation.train_test_split(X_all, y_all, test_size=0.2)

# End of loading and spliting data #


# # initialize the classifier

loss = 'deviance'
n_estimators = 100
learning_rate = 0.1
min_samples_split = 4
max_depth = 5

print "Start to Train"
classifier = GradientBoostingClassifier(loss=loss, n_estimators=n_estimators,learning_rate=learning_rate, min_samples_split=min_samples_split,max_depth=max_depth)  # class_weight set to balanced greatly improved performance

classifier.fit(X_all, y_all)


# Make predictions for the test dataset
y_predict_test = classifier.predict_proba(test)
y_predict_test = y_predict_test[:, 1]

# output the prediction result to a csv file
result = pd.read_csv('data/example_entry.csv')
result.IsBadBuy = y_predict_test
result.to_csv('tosubmit.csv', index=False)

# Make a histgram for test predictions 
plt.hist(y_predict_test[:, 1], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
									 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
plt.xlabel('Probability of being a kicked car')
plt.ylabel('Counts')
plt.title('Predictions for test dataset ')
plt.show()

