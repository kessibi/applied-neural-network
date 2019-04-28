#!/usr/bin/python

import numpy as np
import random
from DataLib import DataLib
from NNLib import NNLib

class NeuralNet:
    eta = 0.01

    def __init__(self,data,batchsize,K, hiddenlayers, hiddenlayersizes):
        self.nbInstances = data.shape[0]
        self.nbFeatures = data.shape[1]-1
        self.batchSize = batchsize
        self.nbClasses = K
        self.trainingData = data
        self.hiddenLayers = hiddenlayers
        self.hiddenLayersSizes = hiddenlayersizes
        self.trainingSize = (int)(0.74*self.nbInstances)
        self.testingSize = self.nbInstances - self.trainingSize
        self.trainingData = data[0:self.trainingSize][:]

        self.testingData = data[self.trainingSize:self.nbInstances][:]
        # self.testingData = data
        for i in range(self.hiddenLayers+1):
            if i==0:
                self.W = [np.random.rand(self.nbFeatures,self.hiddenLayersSizes[0])]
                self.b = [np.random.rand(self.batchSize,self.hiddenLayersSizes[0])]
            elif i==self.hiddenLayers:
                self.W = self.W + [np.random.rand(self.hiddenLayersSizes[i-1],self.nbClasses)]
                self.b = self.b + [np.random.rand(self.batchSize,self.nbClasses)]
            else:
                self.W = self.W + [np.random.rand(self.hiddenLayersSizes[i-1],self.hiddenLayersSizes[i])]
                self.b = self.b + [np.random.rand(self.batchSize,self.hiddenLayersSizes[i])]
        
        # to verify weight matrix sizes and bias matrix sizes
        # for i in range(self.hiddenLayers+1):
        #     print(self.W[i].shape)
        
        # for i in range(self.hiddenLayers+1):
        #     print(self.b[i].shape)

        self.X_train = np.empty([self.batchSize,self.nbFeatures])
        self.Y_train = np.empty([self.batchSize,self.nbClasses])
        self.X_test = np.empty([self.batchSize,self.nbFeatures])
        self.Y_test = np.empty([self.batchSize,self.nbClasses])

        self.result = 0
        self.falsePos = 0
        self.falseNeg = 0
        self.truePos = 0
        self.trueNeg = 0

        self.errors = []

    #done
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

    # done
    def loadAttributesAndLabels(self,dataSet,X,Y,dataIndex,batchSize):
        i = 0
        indexInBatch = dataIndex%batchSize
        for i in range(dataSet.shape[1]-1): # -1 because we do not want to take the label with the features
            X[indexInBatch][i] = dataSet[dataIndex][i]
        i += 1
        self.createOneHot((int)(dataSet[dataIndex][i]),indexInBatch,Y)
    
    # not finished
    def testPrediction(self):
        seenTestingData = 0
        offset = 0
        bSize = 1

        while seenTestingData<self.testingData.shape[0]:
            for i in range(bSize):
                self.loadAttributesAndLabels(self.testingData,self.X_test,self.Y_test,offset+i,bSize)           
            offset += bSize

            Z = []
            A = []


            #Forward propagation
            # testing if the result matches
            # (1,0) = not sick, (0,1) = sick

            for i in range(self.hiddenLayers+1):
                if i == 0:
                    Z.append((self.X_test @ self.W[0]) + self.b[0])
                else:
                    Z.append((A[i-1] @ self.W[i]) + self.b[i])
                
                if i == self.hiddenLayers:
                    A.append(NNLib.softMax(Z[i]))
                else:
                    A.append(NNLib.tanh(Z[i]))


            # testing if the result matches
            if self.Y_test[0][0]==1 and A[self.hiddenLayers][0][0]>=0.5:
                self.result += 1
                self.trueNeg += 1
            elif self.Y_test[0][1]==1  and A[self.hiddenLayers][0][1]>=0.5 :
                self.result += 1
                self.truePos += 1
            elif self.Y_test[0][0]==0 and A[self.hiddenLayers][0][0]>=0.5 :
                self.falseNeg += 1
            else:
                self.falsePos += 1
            
            seenTestingData += bSize

        print("TOTAL OF CORRECT PREDICTIONS: " + str(self.result))
        print("TOTAL OF PREDICTIONS: " + str(self.testingData.shape[0]))
        print("PERCENTAGE OF CORRECT PREDICTIONS: " + str((self.result/self.testingData.shape[0])*100) + " %")
        print("TRUE POSITIVES : " + str(self.truePos) + ", Percentage: "+str(self.truePos*100/self.testingData.shape[0])+ " %")
        print("TRUE NEGATIVES : " + str(self.trueNeg) + ", Percentage: "+str(self.trueNeg*100/self.testingData.shape[0])+ " %")
        print("FALSE POSITIVES : " + str(self.falsePos) + ", Percentage: "+str(self.falsePos*100/self.testingData.shape[0])+ " %")
        print("FALSE NEGATIVES : " + str(self.falseNeg) + ", Percentage: "+str(self.falseNeg*100/self.testingData.shape[0])+ " %")
        if self.truePos + self.trueNeg + self.falsePos + self.falseNeg == self.testingData.shape[0]:
            print("NUMBERS MATCHES")

    # done
    def trainingEpoch(self):
        seenTrainingData = 0
        trainingError = 0.0
        testingError = 0.0
        offset = 0
        self.error = []
        
        self.shuffleTainingData()

        while seenTrainingData<self.trainingData.shape[0]:
            for i in range(self.batchSize):
                self.loadAttributesAndLabels(self.trainingData,self.X_train,self.Y_train,offset+i,self.batchSize)
            offset += self.batchSize
            seenTrainingData += self.batchSize

            Z = []
            A = []
            delta = []
            dW = []
            db = []

            #Forward propagation
            for i in range(self.hiddenLayers+1):
                if i == 0: # if first layer
                    Z.append((self.X_train @ self.W[i]) + self.b[i])
                else:
                    Z.append((A[i-1] @ self.W[i]) + self.b[i])
                
                if i != self.hiddenLayers: # if hidden layer
                    A.append(NNLib.tanh(Z[i]))
                else: # if last layer
                    A.append(NNLib.softMax(Z[i]))
                    
                    #Error computing
                    trainingError = NNLib.crossEntropy(A[i],self.Y_train)
                    self.errors = np.append(self.errors,trainingError)  
            
            #Retropropagation of error
            for i in range(self.hiddenLayers+1):
                if i == 0: # if last layer
                    delta.append(A[self.hiddenLayers] - self.Y_train)
                else: # if hidden layer
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

        return np.mean(self.errors)


    # done
    def train(self,nbEpoch):
        trainingProgress = ""
        for e in range(nbEpoch):
            if e % 100 == 0:
                print(" Epoch "+str(e))
            error = self.trainingEpoch()
            # writes error with epoch number into new CSV to make gnuplot graph
            DataLib.writeToCSV(error,e)
        print("Testing the model with the testingData\n")
        self.testPrediction()
