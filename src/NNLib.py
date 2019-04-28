#!/usr/bin/python

import numpy as np
import math

class NNLib:
    
    @staticmethod
    def tanh(Z):
        return np.tanh(Z)
    
    @staticmethod
    def tanhDeriv(A):
        return 1.0 - np.tanh(A)**2
    
    @staticmethod
    def sigmoid(A):
        return 1.0 / (1.0 + np.exp(-A))
    
    @staticmethod
    def sigmoidDeriv(A):
        return np.exp(-A)/pow(1.0 + np.exp(-A),2.0)

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
        
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                C[i][j] = 1.0 if A[i][j]>0 else 0.0
        
        return C
    
    @staticmethod
    def softMax(Z):
        softA = np.full((Z.shape[0],Z.shape[1]),0.0)
        for i in range(softA.shape[0]): # for each instance in current batch
            for k in range(softA.shape[1]): # for each class k
                sum = 0.0
                for c in range(softA.shape[1]):
                    sum += math.exp(Z[i][c])
                softA[i][k] = (float)(math.exp(Z[i][k]/sum))

        return softA

    @staticmethod
    def crossEntropy(yHat,y):
        K = y.shape[1]
        batchSize = y.shape[0]
        cost = 0.0
        for c in range(batchSize):
           for k in range(K):
               cost += y[c][k]*math.log(yHat[c][k])
        return -(1.0/batchSize)*cost


