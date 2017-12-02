
from time import time

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

t1 =time()
#1.Read Train Data from CSV
read_data = pd.read_csv('/media/top/TOP G/PROJECT_2/IrisProject/file/Train_data.csv')
data = read_data.values

Y_train = data[:,1]
X_train_pca = np.delete(data,[0,1],1)

#Read Test Data from CSV
read_data2 = pd.read_csv('/media/top/TOP G/PROJECT_2/IrisProject/file/Test_data.csv')
data2 = read_data2.values

Y_test = data[:,1]
X_test_pca = np.delete(data2,[0,1],1)

print("time chuan bi data: %0.3f"%(time()-t1))
def predict(model, X):

	return model.predict(X)


#3. Build Model
num_neuron = np.array([20,25])

t2 = time()
for i in (num_neuron):
	mlp = MLPClassifier(hidden_layer_sizes=(i,),max_iter=400,activation='relu',solver='sgd',
	                        learning_rate_init=0.001,tol=1e-4,random_state=1,verbose=False)
	mlp.fit(X_train_pca,Y_train)

	Y_predict = predict(mlp,X_test_pca)
	print("Loss of %d Neuron of Hidden Layer: %0.6f"%(i,mlp.loss_))
	print("Score of %d Neuron of Hidden Layer: %d"%(i,accuracy_score(Y_test, Y_predict, normalize=False)))
	#from sklearn.metrics import classification_report
	from sklearn.metrics import precision_recall_fscore_support
	print("Neuron %d"%i)
	print(precision_recall_fscore_support(Y_test, Y_predict, average = 'weighted'))


print("Time train NN = %0.3fs"%(time()-t2))








            
