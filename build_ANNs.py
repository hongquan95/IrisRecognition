from time import time
import numpy as np
import cv2

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
#import matplotlib.pyplot as plt

view_ids = np.arange(16)

path_train='/TOP G/database1/yalefaces/yalefaces/'
path = '/G/database1/yalefaces/yalefaces/'
list_tr = [".centerlight",".glasses",".happy",".leftlight",'.noglasses',".normal",".rightlight",".sad",".sleepy",".surprised",".wink"]
list_fn = []


for v_id1 in range(2,16):
	for v_id2 in (list_tr):
		fn = path+'subject'+str(v_id1).zfill(2)+v_id2
		list_fn.append(fn)
ax=243
ay=320
n_sample = 15
X_full = np.zeros((14*11,ax*ay))
for i in range(len(list_fn)):
	gray=np.array( Image.open(list_fn[i]))
	im_vec = gray.reshape(1, ax*ay)
	X_full[i, :] = im_vec
#X_mean = np.mean(X_full,axis = 0)
#std = np.std(X_full,axis = 0)
#X_train = (X_full - X_mean) / std

t1 = time()
scaler = StandardScaler()
scaler.fit(X_full)
X_train_lib = scaler.transform(X_full)
pca = PCA(n_components =150,svd_solver='full' )
pca.fit(X_train_lib)

X_train_pca = pca.transform(X_train_lib)
Y_train = np.zeros((14*11,))
for i in range(14):
    for j in range(11):
        Y_train[i*11+j]=i+1
        Y_train = Y_train.astype(int)
    

print("Time PCA = %0.3fs"%(time()-t1))

num_neuron = np.array([80])
t2 = time()


mlp = MLPClassifier(hidden_layer_sizes=(80,),max_iter=1000,activation='relu',solver='sgd',
                    learning_rate_init=0.001,tol=1e-6,random_state=1,verbose=True)
mlp.fit(X_train_pca,Y_train)
print("Lost of %d=%0.10f"%(i,mlp.loss_))
epsilon = mlp.loss_
print("Iter stop of %d = %d"%(i,mlp.n_iter_))

#for i in (num_neuron):
#    mlp = MLPClassifier(hidden_layer_sizes=(i,),max_iter=200,activation='relu',solver='sgd',
#                       learning_rate_init=0.01,tol=1e-6,random_state=1,verbose=True)
#    mlp.fit(X_train_pca,Y_train)
#    print("Iter stop of %d = %d"%(i,mlp.n_iter_))

# 65 la ra loss tot nhat
print("Time train NN = %0.3fs"%(time()-t2))





            
