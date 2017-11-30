
from __future__ import print_function
from time import time
import numpy as np

#import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV


from sklearn.svm import SVC

#import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def build_list(path,vx,vy):
	list_fn = []
	for i in range(1,vx+1):
		for j in (vy):
			fn = path+'subject'+str(i).zfill(2) + j
			list_fn.append(fn)
	return list_fn

def build_data(list_name, x, y, r,h):
	X_full = np.zeros((x*y,r*h))
	for i in range(len(list_name)):
		gray=np.array( Image.open(list_name[i]))
		im_vec = gray.reshape(1,320*243)
		X_full[i, :] = im_vec
	return X_full
def lebel(nx,ny):
	Y = np.zeros((nx*ny,))
	for i in range(nx):
		for j in range(ny):
			Y[i*ny+j]=i+1
	return Y.astype(int)


def predict(model, X):

	return model.predict(X)


path_train='/media/top/TOP G/database1/yalefaces/yalefaces/'
# list_tr = [".centerlight",".glasses",".happy",".leftlight",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"]
list_tr = [".glasses",".happy",".leftlight",".rightlight",".sad",".sleepy",".surprised",".wink"]

n_person = 15
n_pic = 8
x = 243
y = 320
list_train = build_list(path_train,n_person,list_tr)
X_train = build_data(list_train,n_person, n_pic,x,y)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

t1 = time()
pca = PCA(n_components =150,svd_solver='full' ).fit(X_train_norm)
X_train_pca = pca.transform(X_train_norm)

print("Time PCA = %0.3fs"%(time()-t1))

Y_train = lebel(n_person,n_pic)


#Test data

path_test='/media/top/TOP G/database1/yalefaces/test/'
list_te = ['.noglasses','.normal',".centerlight"]
n_pic_test = 3
list_test = build_list(path_test,n_person,list_te)
X_test = build_data(list_test,n_person, n_pic_test,x,y)

X_test_norm = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_norm)

Y_test = lebel(n_person,n_pic_test)

# #ANN model-------------

# num_neuron = np.array([70,75,80,85,90,95,100,105,110,120])
# t2 = time()
# for i in (num_neuron):
# 	mlp = MLPClassifier(hidden_layer_sizes=(i,),max_iter=500,activation='relu',solver='sgd',
# 	                    learning_rate_init=0.001,tol=1e-6,random_state=1,verbose=False)
# 	mlp.fit(X_train_pca,Y_train)
# 	Y_predict = predict(mlp,X_test_pca)
# 	print("Loss of %d Neuron of Hidden Layer: %0.6f"%(i,mlp.loss_))
# 	print("Score of %d Neuron of Hidden Layer: %d"%(i,accuracy_score(Y_test, Y_predict, normalize=False)))

# # 65 la ra loss tot nhat
# print("Time train NN = %0.3fs"%(time()-t2))

#SVM model---------------

t3 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid,return_train_score=True)
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t3))


Y_pred = clf.predict(X_test_pca)

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
print(Y_pred)
# print("Score SVM: %f"%(accuracy_score(Y_test, Y_pred)))


#print(classification_report(Y_test, Y_pred))
#
#print(confusion_matrix(Y_test, Y_pred, labels=range(n_person)))










            
