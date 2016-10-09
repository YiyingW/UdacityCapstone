import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import cross_validation
from time import time 
import pickle
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import tensorflow as tf 

# Load in dataset, split into training, validation and test sets #
pickle_in = open('preprocessed_dataset.pickle', 'rb')
dataset = pickle.load(pickle_in)

train_validation = dataset[dataset['tst']==0]
test = dataset[dataset['tst']==1]

train_validation = train_validation.drop(['tst'], 1)
test = test.drop(['tst'], 1)

X_all = train_validation.drop(['IsBadBuy'], 1)
y_all = train_validation['IsBadBuy']
y_all_2 = pd.get_dummies(y_all, prefix='y')



X_all = np.array(X_all)
y_all_2 = np.array(y_all_2)

X_train, X_validation, y_train, y_validation = cross_validation.train_test_split(X_all, y_all_2, test_size=0.2)

# End of loading and spliting data #


# these can be changed 
n_nodes_hl1 = 14
n_nodes_hl2 = 13
n_nodes_hl3 = 10

n_classes = 2
batch_size = 1000

x = tf.placeholder('float', [None, 205])
y = tf.placeholder('float')

### Setting up the computation graph ###
def neural_network_model(data):

	# input_data * weights + biases
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([205, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.sigmoid(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output



def train_neural_network(x):
	auc_results = {'Train_AUC':[], 'Validation_AUC':[]}
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	

	# learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles of feed forward and backprop
	hm_epochs = 60

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			hm_batch = int(len(X_train)/batch_size)
			for _ in range(hm_batch):
				# 0:batch_size, batch_size:2*batch_size, ..., _*batch_size, (_+1)*batch_size, 
				if _ != (hm_batch - 1):
					epoch_x, epoch_y = X_train[_*batch_size:(_+1)*batch_size], y_train[_*batch_size:(_+1)*batch_size].reshape([batch_size, 2])  # Build a function to do this, epoch_x and epoch_y are numpy.ndarray
				else:
					epoch_x, epoch_y = X_train[_*batch_size:], y_train[_*batch_size:].reshape([(len(y_train) - _*batch_size), 2])
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss
			predicts = sess.run(prediction, feed_dict={x: X_train})
			print 'train auc:', roc_auc_score(y_train, predicts)
			auc_results['Train_AUC'].append(roc_auc_score(y_train, predicts))
			predicts2 = sess.run(prediction, feed_dict={x: X_validation})
			print 'test auc:', roc_auc_score(y_validation, predicts2)
			auc_results['Validation_AUC'].append(roc_auc_score(y_validation, predicts2))

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print 'Training Accuracy:', accuracy.eval({x:X_train, y:y_train.reshape([len(y_train), 2])}) 
		print 'Validation Accuracy:', accuracy.eval({x:X_validation, y:y_validation.reshape([len(y_validation), 2])}) 
		pred = sess.run(prediction, feed_dict={x:X_validation})
		return (pred, auc_results)



y_vali_predictions, auc_results = train_neural_network(x)
auc_results = pd.DataFrame(auc_results)

train_auc = plt.plot(np.array(auc_results['Train_AUC']), label = 'Train data')
validation_auc = plt.plot(np.array(auc_results['Validation_AUC']), label='Validation data')
plt.legend(loc='lower right')
plt.ylabel('AUC')
plt.xlabel('Iterations')
plt.show()

# sess = tf.Session()
# actual = sess.run(tf.argmax(y_validation, 1))
# predictions = sess.run(tf.argmax(y_vali_predictions, 1))
# sess.close()


# Compute ROC curve and ROC area for each class

y_score = y_vali_predictions
print y_score[:10]
y_test = y_validation
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



