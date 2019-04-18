#!/usr/bin/python

import numpy as np
import random
from DataLib import DataLib
from NNLib import NNLib

class NeuralNet:
    eta = 0.01

    def __init__(self,data,batchsize,K, hiddenlayers):
        self.nbInstances = data.shape[0]
        self.nbFeatures = data.shape[1]-1
        self.batchSize = batchsize
        self.nbClasses = K
        self.trainingData = data
        self.hiddenLayers = hiddenlayers
        self.trainingSize = (int)(0.75*self.nbInstances)
        self.testingSize = self.nbInstances - self.trainingSize
        self.trainingData = data[0:self.trainingSize][:]
        #self.testingData = data[self.trainingSize:self.nbInstances][:]
        self.testingData = data
        
        self.W = [np.random.rand(self.nbFeatures, 5)]
        self.W = self.W + [np.random.rand(5,5)] * (self.hiddenLayers - 1)
        self.W = self.W + [np.random.rand(5,self.nbClasses)]

        self.b = [np.full((1,5),0.0)] * self.hiddenLayers
        self.b = self.b + [np.full((1,2),0.0)]

        self.X_train = np.empty([self.batchSize,self.nbFeatures])
        self.Y_train = np.empty([self.batchSize,self.nbClasses])
        self.X_test = np.empty([self.batchSize,self.nbFeatures])
        self.Y_test = np.empty([self.batchSize,self.nbClasses])

        self.result = 0;

# IMPORTANT NOTE: lines & columns may be inverted in all the following
# functions. For now, I have just translated the java code of TP4 in 
# python.

    
    #done, not tested
    # if label = 0 then one-hot = (1,0)
    # if label = 1 then one-hot = (0,1)
    def createOneHot(self,k,indexInBatch,Y):
        if(k>Y.shape[1]-1): # if k is higher than the number of columns
            print("There is only " + str(Y.shape[1]) + " possible classes!")
        for i in range(Y.shape[1]):
            if i==k:
                Y[indexInBatch][i] = 1
            else:
                Y[indexInBatch][i] = 0
    
    #done
    def shuffleTainingData(self):
        np.random.shuffle(self.trainingData)

    # done, not tested
    def loadAttributesAndLabels(self,dataSet,X,Y,dataIndex,batchSize):
        i = 0
        indexInBatch = dataIndex%batchSize
        for i in range(dataSet.shape[1]-1): # -1 because we do not want to take the label with the features
            X[indexInBatch][i] = dataSet[dataIndex][i]
        i += 1
        self.createOneHot((int)(dataSet[dataIndex][i]),indexInBatch,Y)
    
    # not finished
    def testPrediction(self):
        currentDataIndex = 0
        cost = 0.0
        countPredicitions = 0

        seenTestingData = 0
        offset = 0

        while seenTestingData<self.testingData.shape[0]:
            for i in range(self.batchSize):
                self.loadAttributesAndLabels(self.testingData,self.X_test,self.Y_test,offset+i,self.batchSize)
            offset += self.batchSize

            Z = []
            A = []

            #Forward propagation
            for i in range(self.hiddenLayers+1):
                if i == 0:
                    Z.append(np.add(np.dot(self.X_test,self.W[0]),self.b[0]))
                else:
                    Z.append(np.add(np.dot(A[i-1],self.W[i]),self.b[i]))
                
                if i == self.hiddenLayers:
                    A.append(NNLib.softMax(Z[i]))
                else:
                    A.append(NNLib.tanh(Z[i]))


            # testing if the result matches
            if abs(self.Y_test[0][0] - A[self.hiddenLayers][0][0]) <= 0.5:
                self.result += 1
            # print otherwise for information purposes
            else:
                print(self.Y_test[0])
                print("Label trouvé: "+str(A[self.hiddenLayers]))
                print("")
            
            seenTestingData += self.batchSize

        print(self.result)
        print(self.testingData.shape[0])
        print(str(self.result/self.testingData.shape[0]) + "%")

    # done
    def trainingEpoch(self):
        seenTrainingData = 0
        trainingError = 0.0
        testingError = 0.0
        offset = 0
        
        self.shuffleTainingData()

        while seenTrainingData<self.trainingData.shape[0]:
            for i in range(self.batchSize):
                self.loadAttributesAndLabels(self.trainingData,self.X_train,self.Y_train,offset+i,self.batchSize)
            offset += self.batchSize

            Z = []
            A = []
            delta = []
            dW = []
            db = []

            #Forward propagation
            for i in range(self.hiddenLayers+1):
                if i == 0:
                    Z.append((self.X_train @ self.W[i]) + self.b[i])
                else:
                    Z.append((A[i-1] @ self.W[i]) + self.b[i])
                
                if i != self.hiddenLayers:
                    A.append(NNLib.tanh(Z[i]))
                else:
                    A.append(NNLib.softMax(Z[i]))
                    
                    #Error computing
                    trainingError = NNLib.crossEntropy(A[i],self.Y_train)
            
            #Retropropagation of error
            for i in range(self.hiddenLayers+1):
                if i == 0:
                    delta.append(A[self.hiddenLayers] - self.Y_train)
                else:
                    delta.append(delta[i-1] @ np.transpose(self.W[self.hiddenLayers - i + 1]) * NNLib.tanhDeriv(Z[self.hiddenLayers - i]))
                
                db.append(delta[i])
                
                if i != self.hiddenLayers:
                    dW.append(np.transpose(A[self.hiddenLayers - i - 1]) @ delta[i])
                else:
                    dW.append(np.transpose(self.X_train) @ delta[i])
            


            
            #Parameters update
            for i in range(self.hiddenLayers+1):
                self.W[i] = self.W[i] - self.eta * dW[self.hiddenLayers - i]
                self.b[i] = self.b[i] - self.eta * db[self.hiddenLayers - i]

            seenTrainingData += self.batchSize

        # return self.testPrediction()

    # done, not tested
    def train(self,nbEpoch):
        trainingProgress = ""
        for e in range(nbEpoch):
            if e % 100 == 0:
                print(" Epoch "+str(e))
            self.trainingEpoch()
            #trainingProgress += str(e) + "," + self.trainingEpoch() + "\n"
        print("Testing the model with the testingData\n")
        self.testPrediction()
        DataLib.exportCSV()



#csv = DataLib()
#nn = NeuralNet(csv.csvToArray("../heart_disease_dataset.csv"),2,4)
#print(nn.trainingData)
#print(nn.testingData)
#nn.train(10)
