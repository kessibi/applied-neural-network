#!/usr/bin/python

import numpy as np
import math

class DataLib:

    @staticmethod
    def csvToArray(filename):
        data = np.loadtxt(open(filename, "rb"), delimiter=";", skiprows=1)
        return data
    
    @staticmethod
    def normalizeData(data,columns):
        data_norm = np.copy(data)
        for i in range(columns.size):
            m = np.mean(data[:,columns[i]]) #mean of the column number column[i]
            sd = math.sqrt(np.var(data[:,columns[i]])) #standard deviation of the column number[i]
            for j in range(data.shape[0]):
                data_norm[j][columns[i]] = (data[j][columns[i]]-m)/sd
        
        return data_norm

    def shuffleData(self,data):
        nbInstances = data.shape[0]
        nbFeatures = data.shape[1]
    
    @staticmethod
    def exportCSV():
        print("exporting CSV")



# parser = DataLib()
# print(parser.csvToArray("heart_disease_dataset.csv").shape[1])