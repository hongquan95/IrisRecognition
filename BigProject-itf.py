
from time import time

import pandas as pd
import numpy as np
#import cv2
import cv2
from sklearn.preprocessing import StandardScaler



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
	#can than kich thuoc reshape
		X_full[i, :] = vectorize_img(list_name[i])
	X = np.dot(X_full, ProjectionMatrix)	
	return X, a, b

def vectorize_img(filename):    
    # load image 
	rgb = cv2.imread(filename) 
    
	
	gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY) 
	
	im_vec = gray.reshape(1, (x*y))
	return im_vec 
	
def lebel(nx,ny):
	Y = np.zeros((nx*ny,))
	for i in range(nx):
		for j in range(ny):
			Y[i*ny+j]=i+1
	Y = Y.astype(int)
	Y = Y.reshape(nx*ny,1)
	return Y




#1.Train data function

def Train_data(path_train):
	print('aa')
	
	
	
	list_trainain = build_list(path_train,NumPerson,list_train)


	X_train,a,b = build_data(list_trainain,NumPerson, NumPicTrain,x,y)

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_norm = scaler.transform(X_train)

	


	Y_train = lebel(NumPerson,NumPicTrain)

	return X_train_norm, Y_train, scaler




#2.Save Train Data to CSV
def Save_data_train(X_train_norm, Y_train):
	data =  np.concatenate((Y_train,X_train_norm),axis=1)
	df = pd.DataFrame(data)
	df.to_csv('file/Train_data.csv')


# ##3.Phan validation
# def Test(path_test,scaler):
	
	
# 	list_test = build_list(path_test,NumPerson,list_te)
# 	X_test,_,_ = build_data(list_test,NumPerson, NumPicTest,x,y)

# 	X_test_norm = scaler.transform(X_test)
	
# 	Y_test = lebel(NumPerson,NumPicTest)
# 	return X_test_norm, Y_test

# def Save_data_test(X_test_norm, Y_test):

# #2.Save Test Data to CSV
# 	data2 =  np.concatenate((Y_test,X_test_norm),axis=1)
# 	df2 = pd.DataFrame(data2)
# 	df2.to_csv('file/Test_data.csv')



#4Phan lay anh de tes cuoi cung

def test_final(path, scaler):

	

	im_vec = vectorize_img(path) #can than kich thuoc reshape

	im1 = np.dot(im_vec, ProjectionMatrix)
	f_test_norm = scaler.transform(im1)

	return f_test_norm



from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



#Good parameters
# mlp = MLPClassifier(hidden_layer_sizes=(i,),max_iter=500,activation='relu',solver='sgd',
# 		                        learning_rate_init=0.001,tol=1e-4,random_state=1,verbose=False)


def predict(model, X):
	predicta = model.predict(X)

	return predicta

def Train_Model(num_neuron, max_iter, activation, learning_rate, epsilon):

	

	#1.Read Train Data from CSV
	read_data = pd.read_csv('file/Train_data.csv')
	data = read_data.values
	# np.random.shuffle(data) ########################3

	Y_train = data[:,1]
	X_train_norm = data[:,1:-1] #Fucking sai chi so

	#Read Test Data from CSV
	# read_data2 = pd.read_csv('file/Test_data.csv')
	# data2 = read_data2.values
	# # np.random.shuffle(data2)#######################

	# Y_test = data2[:,1]
	# X_test_norm = data2[:,2:-1]

	


	#3. Build Model



	
	mlp = MLPClassifier(hidden_layer_sizes=(num_neuron,),max_iter=max_iter,activation=activation,solver='sgd',
	                        learning_rate_init=learning_rate,tol=epsilon,random_state=1,verbose=False)
	mlp.fit(X_train_norm,Y_train)
	print("Loss of %d Neuron of Hidden Layer: %0.6f"%(num_neuron,mlp.loss_))

	# Y_predict = predict(mlp,X_test_norm)
	
	# print("Score of %d Neuron of Hidden Layer: %d/%d"%(num_neuron,accuracy_score(Y_test, Y_predict, normalize=False),len(Y_test)))
	# #from sklearn.metrics import classification_report
	# from sklearn.metrics import precision_recall_fscore_support
	# print("Neuron %d"%num_neuron)
	# print(precision_recall_fscore_support(Y_test, Y_predict, average = 'weighted'))
	# print(accuracy_score(Y_test, Y_predict))

	#accuracy_score(Y_test, Y_predict, normalize=False)
	return mlp.loss_, mlp






###################33Main##########################
D = 440*60 # original dimension #kich thuoc
d = 500 # new dimension 
ProjectionMatrix = np.random.randn(D, d) 

NumPerson = 49
NumPicTrain = 4
NumPicTest = 0
list_train = np.arange(1,5)
list_te = np.arange(0)


x = 60
y = 440
raito = 1
x1 = int(x*raito)
y1 = int(y*raito)

def full(value,a,b,c,d,e):
	
	if value == 1:
		path_test = a
		X_train_norm, Y_train, scaler  = Train_data(a)   ################  Choice Data
		# X_test_norm, Y_test = Test(path_test, scaler)		
		# Save_data_test(X_test_norm, Y_test)
		print("Full 1 Done!")
		return  X_train_norm,Y_train, scaler  #, X_test_norm,Y_test		

	if value == 2:
		print("Full 2 Done!")
		return Save_data_train(X_train_norm = a, Y_train = b)										###############  Save Data

	
	#X_test_final_pca la anh dua vo nut Identify


	if value == 3:
		X_test_final_norm = test_final(path=a,scaler=b)  #########Choice image test
		print("Full 3 Done!")
		return X_test_final_norm


	#Indentify

	#Train_Model(num_neuron, max_iter, activation, learning_rate, epsilon):
	if value == 4:
		Cost_function, Mlp=Train_Model(num_neuron =a, max_iter=b, activation =c,learning_rate= d,epsilon= e) #####NutTrainData
		print("Full 4 Done!")

		return Cost_function, Mlp

	if value == 5:
		Y_predict_final = predict(a, b)		####Nut indentify	
		

		#Nhap threshold
		

		if kt(y=Y_predict_final,X_train_norm=d,X_test_final_norm =b,threshold= c) == True:
			result = True
		else:
			result = False

		print("Full 5 Done!")

		return Y_predict_final,result 		####Nut indentify



def kt(y ,X_train_norm, X_test_final_norm, threshold):

	distance = np.zeros(4)
	x1 = X_test_final_norm

	for i in range(4):
		x2 = X_train_norm[ int(4 * (y - 1) + i) ,:]
		distance[i] = np.sqrt( ((x2 - x1)**2).sum() )




	for j in range(4):
		print(distance[j])
		if distance[i] > threshold:
			return False
	return True
	

####################MAIN#######################
if __name__ == '__main__':
		
	#Chon du lieu	
	path_train='/media/top/TOP G/database1/IRIS/PHONG/'

	X_train_norm, Y_train, scaler = full(1,path_train,None, None, None, None)

	full(2,X_train_norm,Y_train,None,None, None)

	path_test_final = '/media/top/TOP G/database1/IRIS/PHONG/1_7.jpg'


	X_test_final_norm = full(3,path_test_final,scaler,None, None, None)

	Cost_function, Mlp = full(4,160,400,'relu',0.01,1e-4)
	# #Truyen threshold
	threshold = 50;

	Y_predict_final, result = full(5,Mlp, X_test_final_norm, threshold, X_train_norm, None)

	print(Y_predict_final, result)