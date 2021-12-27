# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:23:51 2021

@author: Utilizador
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from math import exp




def data_process(filename):
#-------------Data Pre-Processing-----------------#

    classes = np.array([]) #label
    
    #attributes --> attention: nor duration nor peakf are used!
    att1 = np.array([])  #meanfreq
    att2 = np.array([])  #sd
    att3 = np.array([])  #median
    att4 = np.array([])  #Q25
    att5 = np.array([])  #Q75
    att6 = np.array([])  #IQR
    att7 = np.array([])  #skew
    att8 = np.array([])  #kurt
    att9 = np.array([])  #sp.ent
    att10 = np.array([]) #sfm
    att11 = np.array([]) #mode
    att12 = np.array([]) #centroid
    att13 = np.array([]) #meanfun
    att14 = np.array([]) #minfun
    att15 = np.array([]) #maxfun
    att16 = np.array([]) #meandom
    att17 = np.array([]) #mindom
    att18 = np.array([]) #maxdom
    att19 = np.array([]) #dfrange
    att20 = np.array([]) #modindx
    
    fields = np.array([])
    n_humans = 0
    
    #reading csv file
    with open(filename, 'r') as file:
        reader = csv.reader(file) #reader object created
        
        fields = next(reader)
        
        for line in reader:
            att1 = np.append(att1, float(line[0]))
            att2 = np.append(att2, float(line[1]))
            att3 = np.append(att3, float(line[2]))
            att4 = np.append(att4, float(line[3]))
            att5 = np.append(att5, float(line[4]))
            att6 = np.append(att6, float(line[5]))
            att7 = np.append(att7, float(line[6]))
            att8 = np.append(att8, float(line[7]))
            att9 = np.append(att9, float(line[8]))
            att10 = np.append(att10, float(line[9]))
            att11 = np.append(att11, float(line[10]))
            att12 = np.append(att12, float(line[11]))
            att13 = np.append(att13, float(line[12]))
            att14 = np.append(att14, float(line[13]))
            att15 = np.append(att15, float(line[14]))
            att16 = np.append(att16, float(line[15]))
            att17 = np.append(att17, float(line[16]))
            att18 = np.append(att18, float(line[17]))
            att19 = np.append(att19, float(line[18]))
            att20 = np.append(att20, float(line[19]))
            if(line[20]=='male'):
                classes = np.append(classes, 1)
            else:
                classes = np.append(classes, 0)
                    
            n_humans = n_humans + 1
            
    file.close()
    #print(classes)
    #print(n_humans)
    #-----------end----------#
    
    #-------Divide the data set in Train, Validation, and Test data------------#
    
    #100% -> 3168 randomize every line so we don´t have males and then females
    Dataset = np.transpose(np.array([att1, att2, att3, att4, att5, att6, att7, att8, att9, att10, att11, att12, att13, att14, att15, att16, att17, att18, att19, att20, classes]))
    np.random.shuffle(Dataset)

    
    #print(Dataset.shape)
    
    classes = Dataset[:,-1]
    #50% of the dataset for training -> -1584
    X_train = np.transpose(np.array([Dataset[:1584,:20]]))
    Y_train = np.transpose(np.array([Dataset[:1584,20]]))
    #print(X_train)
    #print(Y_train)
    
    #25% of the dataset for validation -> 1584-2376
    X_val = np.transpose(np.array([Dataset[1584:2376,:20]]))
    Y_val = np.transpose(np.array([Dataset[1584:2376,20]]))
    #print(X_val)
    #print(Y_val)
                         
    #25% of the dataset for testing -> 2376-
    X_test = np.transpose(np.array([Dataset[2376:,:20]]))
    Y_test = np.transpose(np.array([Dataset[2376:,20]]))
    #print(X_test)
    #print(Y_test)
    
    #print((X_train.size + X_val.size + X_test.size)/20)
    #print(Y_train.size + Y_val.size + Y_test.size)
    
    #---------end----------#
    return X_train, classes
xtrain =  np.array([]) 
xtrain, classes = data_process("voice.csv")
coef = np.array([[-0.406605464, 0.852573316, -1.104746259, -1.24353, 0.444, -0.7, 1.2, -1.3, 0.8, 1.2, 0.111, -0.111, 1.2, -1.2 , -0.333, 1.7,-2.1, -0.543, -1.4, 0.112, -0.223]])  
print(classes)

# Make a prediction with coefficients
def predict(column, coefficients):
	yhat = coefficients[0][0]
	for i in range(column.size):
		 yhat += coefficients[0][i+1] * column[0][i]
	return 1/(1+np.exp(-(yhat)))

for i in range(0,1584,1):
    coul = np.array((xtrain[:,i]))
    yhat = predict(coul.T, coef)
    print("Expected=%.3f, Predicted=%.3f [%d]" % (classes[i], yhat, round(yhat)))
