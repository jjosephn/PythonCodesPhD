import time
start = time.time()


import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io as sio

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers.core import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import sys
# keras.layers.normalization.BatchNormalization


print('-' * 100);
print("""----------------------------   SIMPLE RNN 16 Time Steps  ----------------------------""");      
print('-' * 100);

# Requirements

# Python 3
# numpy
# matplotlib
# pywt (for wavelet transformation)
# scipy

# Tensorflow
# keras

# Simple RNN model to convert 3 leads Lead I, Lead II and V3 to five precordial leads V1, V2, V4, V5 and V6.
# time steps used is 16. Total sample length is 30000 (training and validation 20000 and testing on last five seconds)
# change the paths accordingly

# The input is .mat file data which is preprocessed and is saved in the ../DNN/dataIn/ folder. The output
# data will be stored in ../DNN/dataOutSimpleRNN16V3/ folder. 

# change totalData = 549 for all records

totalData = 1
epoch = 1000

filenameIn = []
filenameOut = []
for var1 in range(totalData):
	addIn = '../DNN/dataIn/pythonData%i.mat' %var1
	addOut = '../DNN/dataOutSimpleRNN16V3/dataOut%i.mat' %var1
	filenameIn.append(addIn)
	filenameOut.append(addOut)

# print('-' * 50);   

for var1 in range(totalData):
	np.random.seed(1)
	print('-' * 100);      
	print(filenameIn[var1])
	print('-' * 100);      
	dat = sio.loadmat(filenameIn[var1])
	pythonData = dat['pythonData']

	x1 = pythonData[0,:]
	x1.shape = (1,30000)
	x2 = pythonData[1,:]
	x2.shape = (1,30000)
	x3 = pythonData[2,:]
	x3.shape = (1,30000)
	x4 = pythonData[3,:]
	x4.shape = (1,30000)
	x5 = pythonData[4,:]
	x5.shape = (1,30000)
	x6 = pythonData[5,:]
	x6.shape = (1,30000)
	x7 = pythonData[6,:]
	x7.shape = (1,30000)
	x8 = pythonData[7,:]
	x8.shape = (1,30000)
	x9 = pythonData[8,:]
	x9.shape = (1,30000)
	x10 = pythonData[9,:]
	x10.shape = (1,30000)
	x11 = pythonData[10,:]
	x11.shape = (1,30000)
	x12 = pythonData[11,:]
	x12.shape = (1,30000)
	length = 20000
	samples = 30000


	inDataConcatTrain = np.stack((x1[0,0:length], x2[0,0:length], x9[0,0:length]))
	outDataConcatTrain = np.stack((x7[0,0:length], x8[0,0:length], x10[0,0:length], x11[0,0:length], x12[0,0:length]))
	inDataConcatTrain = np.transpose(inDataConcatTrain)

	batchStartTrain = 0
	timeSteps = 16
	batchNumberTrain = 20000 - timeSteps

	inDataTrain = np.zeros((batchNumberTrain,timeSteps,3), dtype=np.float32)
	for i in range(batchNumberTrain):
	    inDataTrain[i] = inDataConcatTrain[batchStartTrain:batchStartTrain+timeSteps,:]
	    # inDataTest[i] = inDataConcatTest[batchStartTrain:batchStartTrain+timeSteps,:]
	    batchStartTrain += 1

	outDataTrain = np.transpose(outDataConcatTrain[:,timeSteps:batchNumberTrain + timeSteps])

    


	inDataConcatTest = np.stack((x1[0,length:samples], x2[0,length:samples], x9[0,length:samples]))	
	outDataConcatTest = np.stack((x7[0,length:samples], x8[0,length:samples], x10[0,length:samples], x11[0,length:samples], x12[0,length:samples]))
	inDataConcatTest = np.transpose(inDataConcatTest)
	
	batchStartTest = 0
	timeSteps = 16	
	batchNumberTest = 10000 - timeSteps

	
	inDataTest = np.zeros((batchNumberTest,timeSteps,3), dtype=np.float32)
	for i in range(batchNumberTest):
		inDataTest[i] = inDataConcatTest[batchStartTest:batchStartTest+timeSteps,:]
		batchStartTest += 1

	outDataTest = np.transpose(outDataConcatTest[:,timeSteps:batchNumberTest + timeSteps])



	model = Sequential()
	# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, input_shape=(timeSteps,3),
	# 	scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
	# 	moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
	# model.add(LSTM(15, input_shape=(timeSteps,3),return_sequences=True))
	# model.add(GRU(15, input_shape=(timeSteps,3),return_sequences=True))
	model.add(SimpleRNN(15, input_shape=(timeSteps,3),return_sequences=True))
	# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
	# 	scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
	# 	moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

	# model.add(Dropout(.2, noise_shape=None, seed=None))
	# model.add(LSTM(15, input_shape=(timeSteps,15),return_sequences=True))
	# model.add(GRU(15, input_shape=(timeSteps,15),return_sequences=True))
	# model.add(SimpleRNN(15, input_shape=(timeSteps,15),return_sequences=True))
	# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
	# 	scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
	# 	moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

	# model.add(Dropout(.2, noise_shape=None, seed=None))
	# model.add(LSTM(5, input_shape=(timeSteps,15),return_sequences=False))
	# model.add(GRU(5, input_shape=(timeSteps,15),return_sequences=False))
	model.add(SimpleRNN(5, input_shape=(timeSteps,15),return_sequences=False))
	# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
	# 	scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
	# 	moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

	# model.add(Dropout(.2, noise_shape=None, seed=None))
	model.add(Dense(5))
	# model.add(Activation('sigmoid'))


	filepath = '../DNN/dataOutSimpleRNN16V3/bestweights.hdf5'                                            # For loading the weights
	# filepath_s="bestweights_1.hdf5"                                              # For saving the best weights(only if the error improves)
	# model.load_weights(filepath_l)                                               # load saved weights if required
	model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])

	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')  # Save the best weights 
	callbacks_list = [checkpoint]

	# model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
	# model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
	# model.fit(inDataTrain, outDataTrain, nb_epoch=500, batch_size=batchNumberTrain, verbose=2, validation_data=(inDataTest, outDataTest))
	model.fit(inDataTrain, outDataTrain, validation_split=0.2, epochs=epoch, batch_size=batchNumberTrain, verbose=2, callbacks=callbacks_list)


	model.load_weights(filepath)                                          #load weights
	model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])



	predict = model.predict(inDataTest)
	predict = np.transpose(predict)
	outDataTest = np.transpose(outDataTest)

	# sio.savemat(filenameOut[var1], mdict={'prediction': predict, 'original': outDataTest})

	K.clear_session()

	print('-' * 100);
	print('----------------------------SIMPLE RNN Completed for inDataTrain', var1);
	print('-' * 100);

print('-' * 100);
print("""----------------------------   SIMPLE RNN 16 Time Steps Completed ----------------------------""");
print('-' * 100);

end = time.time()
print()
print(round((end-start),2), "seconds")

	# pltStart = 0
	# pltStop = 4000

	# plt.figure(1)
	# plt.subplot(511)
	# plt.plot(outDataTest[0, pltStart:pltStop], 'r', predict[0, pltStart:pltStop], 'b')
	# plt.subplot(512)
	# plt.plot(outDataTest[1, pltStart:pltStop], 'r', predict[1, pltStart:pltStop], 'b')
	# plt.subplot(513)
	# plt.plot(outDataTest[2, pltStart:pltStop], 'r', predict[2, pltStart:pltStop], 'b')
	# plt.subplot(514)
	# plt.plot(outDataTest[3, pltStart:pltStop], 'r', predict[3, pltStart:pltStop], 'b')
	# plt.subplot(515)
	# plt.plot(outDataTest[4, pltStart:pltStop], 'r', predict[4, pltStart:pltStop], 'b')
	# plt.show()