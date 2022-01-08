# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:47:49 2022

@author: pelay
"""

import csv
from random import shuffle
import numpy as np
import math 
import matplotlib.pyplot as plt


def column(matrix, i):
    return [row[i] for row in matrix]

def read_file(filename):
    dataset = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for i in reader:
            dataset.append(i)
    return dataset

def string_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])
    return dataset


def min_max(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_val = [j[i] for j in dataset]
        min_ = min(col_val)
        max_ = max(col_val)
        minmax.append([min_,max_])
    return minmax


def normalization(dataset,minmax):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            n = dataset[i][j] - minmax[j][0]
            d = minmax[j][1] - minmax[j][0]
            dataset[i][j] = n/d
    return dataset

def train_test(dataset):
    shuffle(dataset)
    n = int(0.8*len(dataset))
    
    train_data = dataset[:n]
    test_data =  dataset[n:]
    
    return train_data,test_data


def accuracy_check(pred,actual):
    c = 0
    for i in range(len(actual)):
        if(pred[i]==actual[i]):
            c+=1
    acc = (c/len(actual))*100
    return acc


def prediction(row,parameters):
    hypothesis = parameters[0]
    for i in range(len(row)-1):
        hypothesis+=row[i]*parameters[i+1]
    return 1 / (1 + math.exp(-hypothesis))


def cost_function(x,parameters):
    cost = 0
    for row in x:
        pred = prediction(row,parameters)
        y = row[-1]
        cost+= -(y*np.log(pred))+(-(1-y)*np.log(1-pred))
    avg_cost = cost/len(x)
    return avg_cost


dataset = read_file('voice.csv')


def gradient_descent(x,epochs,alpha):
    
    parameters = [0]*len(x[0])
    cost_history = []
    n = len(x)
    
    for i in range(epochs):
        for row in x:
            pred = prediction(row,parameters)
            #for theta 0 partial derivative is different
            parameters[0] = parameters[0]-alpha*(pred-row[-1])
            for j in range(len(row)-1):
                parameters[j+1] = parameters[j+1]-alpha*(pred-row[-1])*row[j]
        cost_history.append(cost_function(x,parameters))
    return cost_history,parameters

def algorithm(train_data,test_data):
    
    epochs = 100
    
    alpha = 0.001
    cost_history,parameters = gradient_descent(train_data,epochs,alpha)
    predictions = []
    
    for i in test_data:
        pred = prediction(i,parameters)
        predictions.append(round(pred))
    y_actual = [i[-1] for i in test_data]    
    accuracy = accuracy_check(predictions,y_actual)
    iterations = [i for i in range(1,epochs+1)]
    plt.plot(iterations,cost_history)
    plt.show()
    return accuracy

dataset = dataset = read_file('voice.csv')
dataset = string_to_float(dataset)
minmax = min_max(dataset)
dataset = normalization(dataset,minmax)
train_data,test_data = train_test(dataset)
accuracy = algorithm(train_data,test_data)
print(accuracy)