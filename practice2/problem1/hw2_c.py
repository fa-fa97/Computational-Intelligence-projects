# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:14:09 2019

@author: Fatemeh
"""

import sys
sys.path.append('./HodaDatasetReader')


from HodaDatasetReader import read_hoda_cdb
from HodaDatasetReader import read_hoda_dataset
import keras
from keras.models import Sequential
#add dropout 
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

#### Read dataset

print('Reading train dataset (Train 60000.cdb)...')
X_train, Y_train = read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/Train 60000.cdb',
                                images_height=32,
                                images_width=32,
                                one_hot=True,
                                reshape=True)

print('Reading test dataset (Test 20000.cdb)...')
X_test, Y_test = read_hoda_dataset(dataset_path='./HodaDatasetReader/DigitDB/Test 20000.cdb',
                              images_height=32,
                              images_width=32,
                              one_hot=True,
                              reshape=True)

#Calculate F1 score ,Reacal , Percision
class LossHistory(keras.callbacks.Callback): 
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.t_losses = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.t_losses.append(self.model.evaluate(self.x_test, self.labels_test)[0])
   
    def on_train_end(self, logs={}):
        val_predict = (np.asarray(self.model.predict(self.x_test))).round()
        val_targ = self.labels_test
        val_f1 = f1_score(val_targ, val_predict, average=None)
        val_f1_all = f1_score(val_targ, val_predict, average="micro")
        val_recall = recall_score(val_targ, val_predict, average=None)
        val_recall_all = recall_score(val_targ, val_predict, average="micro")
        val_precision = precision_score(val_targ, val_predict, average=None)
        val_precision_all = precision_score(val_targ, val_predict, average="micro")
        for i in range(10):
            print( " " + str(i))
            print( " f1 : " + str(val_f1[i]) + " recall: " + str(val_recall[i]) + " precision: " + str(val_precision[i]))
        print("Final Result is" ," f1 : " + str(val_f1_all) + " recall: " + str(val_recall_all) + " precision: " + str(val_precision_all))




model = Sequential()
model.add(Dense(32*32, activation='relu', input_dim=32*32))
model.add(Dropout(0.5))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])




h = LossHistory()
h.x_test = X_test
h.labels_test = Y_test
model.fit(X_train, Y_train, epochs=10, callbacks=[h])

# plot errors
r = plt.figure(1)
plt.plot(h.losses, 'g--', label="Train cost")
plt.plot(h.t_losses, label="Test cost")
plt.legend()
r.show()






