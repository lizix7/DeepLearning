from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras.optimizers import RMSprop
import numpy as np

# Author: Zhixing Li
# Date: 1/1/2017
# Title: Multi-lable classification using keras+ theano 

# fix random seed for reproducibility
# use for random shuffling 
seed = 7
np.random.seed(seed)

# load gas dataset
metadata = np.loadtxt('HT_Sensor_metadata.dat', skiprows=1, dtype=str)

dataset = np.loadtxt('HT_Sensor_dataset.dat', skiprows=1)

banana_id = np.array(metadata[metadata[:,2]=="banana",0],dtype=float)

wine_id = np.array(metadata[metadata[:,2]=="wine",0],dtype=float)

background_id = np.array(metadata[metadata[:,2]=="background",0],dtype=float)

for index in range(len(dataset)):
	if dataset[index,0] in banana_id:
		dataset[index,0] = 0
		
	elif dataset[index,0] in wine_id:
		dataset[index,0] = 1
		
	else:
		dataset[index,0] = 2
		
# print dataset

# split into input (X) and output (Y) variables
X = dataset[:,2:12]
Y = np.array(dataset[:,0],dtype=int)

# data processing

scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler.fit_transform(X)

#print len(X)
#print len(Y)
#print X
#print Y

# split into 80% for training and 20% for test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=seed)

# one hot coding for multi-label classification
y_train = np_utils.to_categorical(Y_train, 3)
y_test = np_utils.to_categorical(Y_test, 3)

# create model
model = Sequential()
model.add(Dense(20, input_dim=10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, nb_epoch=50, batch_size=50)
# evaluate the model
scores = model.evaluate(X_test,y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

