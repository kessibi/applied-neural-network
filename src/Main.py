#!/usr/bin/python

import numpy as np
from DataLib import DataLib
from NeuralNet import NeuralNet
#from NNLib import NNLib

def main():
    data = DataLib.csvToArray("../heart_disease_dataset.csv")
    print(data)
    # 1 = batchsize
    # 2 = number of classes
    nn = NeuralNet(data,1,2)

    nn.train(200)


if __name__ == "__main__":
    main()


