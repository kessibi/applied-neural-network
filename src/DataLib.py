#!/usr/bin/python

import numpy as np

class DataLib:

    @staticmethod
    def csvToArray(filename):
        data = np.loadtxt(open(filename, "rb"), delimiter=";", skiprows=1)
        return data
    
    def shuffleData(self,data):
        nbInstances = data.shape[0]
        nbFeatures = data.shape[1]
    
    @staticmethod
    def exportCSV():
        print("exporting CSV")



# parser = DataLib()
# print(parser.csvToArray("heart_disease_dataset.csv").shape[1])