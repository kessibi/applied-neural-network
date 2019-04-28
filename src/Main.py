#!/usr/bin/python

import numpy as np
import sys, os, argparse

from DataLib import DataLib
from NeuralNet import NeuralNet
from NNLib import NNLib

def main():

    if len(sys.argv) == 1:
        print("usage: python3.7 Main.py -h")
        sys.exit(1)
    # Argument parser
    parser = argparse.ArgumentParser(description='Neural network for heart disease.')
    parser.add_argument('--E', help='number of epochs')
    parser.add_argument('--B', help='size of batches')
    parser.add_argument('--L', nargs='+', help='sizes of hidden layers')
    args = parser.parse_args()
    sizes = list(map(int, args.L))


    # Data processing
    data = DataLib.csvToArray("../heart_disease_dataset.csv")
    DataLib.shuffleData(data)
    dataNorm = DataLib.normalizeData(data,np.array([0,3,4,7,9,11]))

    trainingSize = (int)(0.74*dataNorm.shape[0])
    if(trainingSize%int(args.B)!=0):
        parser.error('Batchsize must be a divisor of '+str(trainingSize)+" !")

    if os.path.exists("../out.csv"):
        os.remove("../out.csv")

    #Neural network
    
    # second argument = batchSize
    # third argument = number of classes
    # fourth argument = number of hidden layers
    nn = NeuralNet(dataNorm,int(args.B),2,len(args.L),sizes)

    nn.train(int(args.E))


if __name__ == "__main__":
    main()


