# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:23:51 2021

@author: Daniel Girão
         Gustavo Pelayo 
         José Dias
"""

import numpy as np
import csv


#-------------Data Processing-----------------#

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
with open('voice.csv', 'r') as file:
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
        classes = np.append(classes, line[20])
        n_humans = n_humans + 1
        
file.close()
#print(n_humans)
#-----------end----------#

#-------Divide the data set in Train, Validation, and Test data------------#



    