
from time import time
import numpy as np
#import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def build_list(path,vx,vy):
	list_fn = []
	for i in range(vx):
		for j in (vy):
			fn = path+str(i).zfill(3) + '/' +'S6'+str(i).zfill(3) +'S'+ str(j).zfill(2) + '.jpg'
			list_fn.append(fn)
	return list_fn

def build_data(list_name, x, y, r,h):
	X_full = np.zeros((x*y,r*h))
	for i in range(len(list_name)):
		gray=np.array( Image.open(list_name[i]))
		im_vec = gray.reshape(1,480*640) #can than kich thuoc reshape
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


path_train='/media/top/TOP G/database1/IRIS/CASIA-Iris-Syn/'
# list_tr = [".centerlight",".glasses",".happy",".leftlight",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"]
list_tr = np.arange(10)
n_person = 10
n_pic = 10
x = 480
y = 640
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


# #Test data

# path_test='/media/top/TOP G/database1/yalefaces/test/'
# list_te = ['.noglasses','.normal',".centerlight"]
# n_pic_test = 3
# list_test = build_list(path_test,n_person,list_te)
# X_test = build_data(list_test,n_person, n_pic_test,x,y)

# X_test_norm = scaler.transform(X_test)
# X_test_pca = pca.transform(X_test_norm)

# Y_test = lebel(n_person,n_pic_test)



# num_neuron = np.array([70,75,80,85,90,95,100,105,110,120])
# t2 = time()
# for i in (num_neuron):
# 	mlp = MLPClassifier(hidden_layer_sizes=(i,),max_iter=500,activation='relu',solver='sgd',
# 	                        learning_rate_init=0.001,tol=1e-6,random_state=1,verbose=False)
# 	mlp.fit(X_train_pca,Y_train)

# 	Y_predict = predict(mlp,X_test_pca)
# 	print("Loss of %d Neuron of Hidden Layer: %0.6f"%(i,mlp.loss_))
# 	print("Score of %d Neuron of Hidden Layer: %d"%(i,accuracy_score(Y_test, Y_predict, normalize=False)))
# 	from sklearn.metrics import classification_report
# 	print("Neuron %d"%i)
# 	print(classification_report(Y_test, Y_predict))

# # 65 la ra loss tot nhat
# print("Time train NN = %0.3fs"%(time()-t2))








            
