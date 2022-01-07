# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:23:51 2021

Python Version: 3.8.8, using Conda

@author: Daniel Girão
         Gustavo Pelayo 
         José Dias
"""

import numpy as np
from Classifiers.LDA import LDA
from DataProcessing.readData import readData
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score
    
#-------------Data Processing-----------------#
r = readData('DataProcessing/voice.csv')
Dataset = r.getDataset()
print(Dataset.shape)
#50% of the dataset for training -> -1584
X_train = Dataset[:1584,:20]
Y_train = Dataset[:1584,20]

f = open("training.txt","w")
np.savetxt("training.txt",Dataset)
f.close()
#25% of the dataset for validation -> 1584-2376
X_val = Dataset[1584:2376,:20]
Y_val = Dataset[1584:2376,20]
            
#25% of the dataset for testing -> 2376-
X_test = Dataset[2376:,:20]
Y_test = Dataset[2376:,20]
#LDA SAMPLED CODE
lda = LDA()
lda.fit(X_train, Y_train)
X_projected = lda.transform(X_train, Y_train)

Y_pred = lda.predict(X_test)
print("Test data perfomance: ", accuracy_score(Y_test, Y_pred))


    