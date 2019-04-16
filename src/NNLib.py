#!/usr/bin/python

import numpy as np
import math

# IMPORTANT NOTE: lines & columns may be inverted in all the following
# functions. For now, I have just translated the java code of TP4 in 
# python.

class NNLib:

    @staticmethod
    def relu(Z):
        activations = np.full((Z.shape[0],Z.shape[1]),0.0)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                activations[i][j] = Z[i][j] if Z[i][j]>0 else 0.0
        
        return activations

    @staticmethod
    def reluDeriv(A):
        C = np.full((A.shape[0],A.shape[1]),0.0)
        
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                C[i][j] = 1.0 if A[i][j]>0 else 0.0
        
        return C
    
    @staticmethod
    def softMax(Z):
        softA = np.full((A.shape[0],A.shape[1]),0.0)
        for i in range(softA.shape[1]):
            for j in range(softA.shape[0]):
                sum = 0.0
                for c in range(softA.shape[0]):
                    sum += math.exp(Z[c][i])
                softA[j][i] = (float)(math.exp(Z[j][i]/sum))

        return softA
    
    @staticmethod
    def checkPredictions(yHat,y,indexInBatch):
        predK = 0
        actualK = 0
        for k in range(1,yHat.shape[0]):
            if yHat[k][indexInBatch] > yHat[predK][indexInBatch]:
                predK = k
            if Y[k][indexInBatch] > yHat[actualK][indexInBatch]:
                actualK = k
        #print("Actual k: " + str(actualK) + " Predicted k: " + str(predK) + " (" + (actualK==predK) + ")")
        return actualK==predK

    @staticmethod
    def crossEntropy(yHat,y):
        K = y.size # number of classes
        cost = 0.0
        batchSize = yHat.shape[1]
        for i in range(K):
            for c in range(batchSize):
                cost += y[i][c]*math.log(yHat[i][c])
        
        return -(1/batchSize)*cost
