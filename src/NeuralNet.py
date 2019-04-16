#!/usr/bin/python

import numpy as np
import random
from DataLib import DataLib

class NeuralNet:
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    b1 = None
    b2 = None

    def __init__(self,data,batchsize,K):
        self.nbInstances = data.shape[0]
        self.nbFeatures = data.shape[1]
        self.batchSize = batchsize
        self.nbClasses = K
        self.trainingData = data
        self.trainingSize = (int)(0.75*self.nbInstances)
        self.testingSize = self.nbInstances - self.trainingSize
        self.trainingData = data[0:self.trainingSize][:]
        self.testingData = data[self.trainingSize:self.nbInstances][:]
        self.W1 = np.random.rand(self.nbFeatures-1,5)
        self.W2 = np.random.rand(5,self.nbClasses)
        self.b1 = np.full((1,5),0.0)
        self.b2 = np.full((1,2),0.0)

# IMPORTANT NOTE: lines & columns may be inverted in all the following
# functions. For now, I have just translated the java code of TP4 in 
# python.

    
    #done, not tested
    def createOneHot(self,K,indexInBatch,Y):
        if(k>Y.shape[0]):
            print("There is only " + Y.shape[0] + "possible classes!")
        for i in range(Y.shape[0]):
            if i==k:
                Y[i][indexInBatch] = 1
            else:
                Y[i][indexInBatch] = 0
    
    #done
    def shuffleTainingData(self):
        np.random.shuffle(self.trainingData)

    # done, not tested
    def loadAttributesAndLabels(self,dataSet,X,Y,dataIndex,batchSize):
        i = 0
        indexInBatch = dataIndex%batchSize
        for i in range(dataSet.shape[1]-1):
            X[indexInBatch][i] = dataSet[indexInBatch][i]
        
        createOneHot((int)(dataSet[dataIndex][i]),indexInBatch,Y)
    
    # not finished
    def testPredicitions(self):
        currentDataIndex = 0
        cost = 0.0
        countPredicitions = 0

    # not done
    def trainingEpoch(self):
        return "training"

    # done, not tested
    def train(self,nbEpoch):
        trainingProgress = ""
        for e in range(nbEpoch):
            print(" [ Epoch "+str(e)+" ]")
            trainingProgress += str(e) + "," + self.trainingEpoch() + "\n"
        DataLib.exportCSV()



csv = DataLib()
nn = NeuralNet(csv.csvToArray("../heart_disease_dataset.csv"),2,4)
#print(nn.trainingData)
#print(nn.testingData)
#nn.train(10)