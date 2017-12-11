
from time import time

import pandas as pd
import numpy as np
#import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




np.seterr(divide='ignore', invalid='ignore')


def build_list(path,vx,vy):
	list_fn = []
	for i in range(1,vx+1):
		for j in (vy):
			fn = path+str(i).zfill(1) + '_' +str(j).zfill(1)+'.jpg'
			list_fn.append(fn)
	return list_fn

def build_data(list_name, x, y, r,h):
	raito = 1  #CHINH RESIZE
	a = int(r*raito)
	b = int(h*raito)
	X_full = np.zeros((x*y,a*b))
	

	for i in range(len(list_name)):
		img = Image.open(list_name[i])
		gray = img.resize((b,a), Image.ANTIALIAS)
		gray=np.array(gray)
		im_vec = gray.reshape(1,b*a) #can than kich thuoc reshape
		X_full[i, :] = im_vec
	return X_full
	
def lebel(nx,ny):
	Y = np.zeros((nx*ny,))
	for i in range(nx):
		for j in range(ny):
			Y[i*ny+j]=i+1
	Y = Y.astype(int)
	Y = Y.reshape(nx*ny,1)
	return Y


#1.Train data

path_train='/media/top/TOP G/database1/IRIS/PHONG/'
# list_tr = [".centerlight",".glasses",".happy",".leftlight",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"]

list_all = np.arange(1,11)
np.random.shuffle(list_all)
list_tr = list_all[0:2]

print(list_tr[0:2])

n_person = 49
n_pic = 2
x = 60
y = 440
list_train = build_list(path_train,n_person,list_tr)
X_train = build_data(list_train,n_person, n_pic,x,y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

t1 = time()
pca = PCA(n_components =170,svd_solver='full' ).fit(X_train_norm)
X_train_pca = pca.transform(X_train_norm)


print("Time PCA = %0.3fs"%(time()-t1))

Y_train = lebel(n_person,n_pic)

#2.Save Train Data to CSV
data =  np.concatenate((Y_train,X_train_pca),axis=1)
df = pd.DataFrame(data)
df.to_csv('/media/top/TOP G/PROJECT_2/IrisProject/file/Train_data.csv')



#3.Test data

path_test='/media/top/TOP G/database1/IRIS/PHONG/'    
list_te = list_all[2:10]
n_pic_test = 8
list_test = build_list(path_test,n_person,list_te)
X_test = build_data(list_test,n_person, n_pic_test,x,y)

X_test_norm = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_norm)
Y_test = lebel(n_person,n_pic_test)

#4.Save Test Data to CSV
data2 =  np.concatenate((Y_test,X_test_pca),axis=1)
df2 = pd.DataFrame(data2)
df2.to_csv('/media/top/TOP G/PROJECT_2/IrisProject/file/Test_data.csv')