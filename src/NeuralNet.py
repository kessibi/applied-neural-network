#!/usr/bin/python

import numpy as np
import random
from DataLib import DataLib
from NNLib import NNLib

class NeuralNet:
    eta = 0.01

    def __init__(self,data,batchsize,K):
        self.nbInstances = data.shape[0]
        self.nbFeatures = data.shape[1]-1
        self.batchSize = batchsize
        self.nbClasses = K
        self.trainingData = data
        self.trainingSize = (int)(0.75*self.nbInstances)
        self.testingSize = self.nbInstances - self.trainingSize
        self.trainingData = data[0:self.trainingSize][:]
        #self.testingData = data[self.trainingSize:self.nbInstances][:]
        self.testingData = data
        self.W1 = np.random.rand(self.nbFeatures,5)
        self.W2 = np.random.rand(5,self.nbClasses)
        self.b1 = np.full((1,5),0.0)
        self.b2 = np.full((1,2),0.0)
        self.X_train = np.empty([self.batchSize,self.nbFeatures])
        self.Y_train = np.empty([self.batchSize,self.nbClasses])
        self.X_test = np.empty([self.batchSize,self.nbFeatures])
        self.Y_test = np.empty([self.batchSize,self.nbClasses])

        self.result = 0
        self.falsePos = 0
        self.falseNeg = 0
        self.truePos = 0
        self.trueNeg = 0

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
    def testPredicition(self):
        currentDataIndex = 0
        cost = 0.0
        countPredicitions = 0

        seenTestingData = 0
        offset = 0

        while seenTestingData<self.testingData.shape[0]:
            for i in range(self.batchSize):
                self.loadAttributesAndLabels(self.testingData,self.X_test,self.Y_test,offset+i,self.batchSize)
            offset += self.batchSize


            #Forward propagation
            Z1 = np.add(np.dot(self.X_test,self.W1),self.b1)
            A1 = NNLib.tanh(Z1)
            Z2 = np.add(np.dot(A1,self.W2),self.b2)
            A2 = NNLib.softMax(Z2)

            # testing if the result matches
            # TODO: coutn falsePos, falseNeg etc...
            if abs(self.Y_test[0][0] - A2[0][0]) <= 0.5:
                self.result += 1
            # print otherwise for information purposes
            else:
                print("WRONG PREDICTION:")
                print(self.Y_test[0])
                print("Label founded: "+str(A2))
                print("")
            
            seenTestingData += self.batchSize

        print("TOTAL OF CORRECT PREDICITIONS: " + str(self.result))
        print("TOTAL OF PREDICITIONS: " + str(self.testingData.shape[0]))
        print("PERCENTAGE OF CORRECT PREDICITIONS: " + str((self.result/self.testingData.shape[0])*100) + " %")
        print("FALSE POSITIVES : " + str(self.falsePos))
        print("FALSE NEGATIVES : " + str(self.falseNeg))
        print("TRUE POSITIVES : " + str(self.truePos))
        print("TRUE NEGATIVES : " + str(self.trueNeg))

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
            
            #Forward propagation
            Z1 = (self.X_train @ self.W1) + self.b1
            A1 = NNLib.tanh(Z1)
            Z2 = (A1 @ self.W2) + self.b2
            A2 = NNLib.softMax(Z2)

            #Error calculation
            trainingError = NNLib.crossEntropy(A2,self.Y_train)

            #Retropropagation of error
            delta2 = A2 - self.Y_train
            dW2 = np.transpose(A1) @ delta2
            db2 = delta2

            delta1 = (delta2 @ np.transpose(self.W2)) * NNLib.tanhDeriv(Z1)
            dW1 = np.transpose(self.X_train) @ delta1
            db1 = delta1

            #Parameters update
            self.W2 = self.W2 - self.eta*dW2
            self.b2 = self.b2 - self.eta*db2

            self.W1 = self.W1 - self.eta*dW1
            self.b1 = self.b1 - self.eta*db1

            seenTrainingData += self.batchSize

        return trainingError

    # done
    def train(self,nbEpoch):
        trainingProgress = ""
        for e in range(nbEpoch):
            if e % 100 == 0:
                print(" Epoch "+str(e))
            error = self.trainingEpoch()
            # writes error with epoch number into new CSV to make gnuplot graph
            DataLib.writeToCSV(error,e)
            #trainingProgress += str(e) + "," + self.trainingEpoch() + "\n"
        print("Testing the model with the testingData\n")
        self.testPredicition()
        DataLib.exportCSV()



#csv = DataLib()
#nn = NeuralNet(csv.csvToArray("../heart_disease_dataset.csv"),2,4)
#print(nn.trainingData)
#print(nn.testingData)
#nn.train(10)
