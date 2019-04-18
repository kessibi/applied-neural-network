#!/usr/bin/python

import numpy as np
import sys, os

#np.set_printoptions(threshold=np.nan)
from DataLib import DataLib
from NeuralNet import NeuralNet
from NNLib import NNLib
import argparse

def main():

    if len(sys.argv) == 1:
        print("usage: python3.7 Main.py -h")
        sys.exit(1)
    # Argument parser
    parser = argparse.ArgumentParser(description='Neural network for heart disease.')
    parser.add_argument('--epoch', help='number of epochs')
    parser.add_argument('--batchsize', help='size of batches')
    parser.add_argument('--nhlayers', help='number of hidden layers')
    parser.add_argument('--hlayerssizes', nargs='+', help='sizes of hidden layers')
    args = parser.parse_args()
    if int(args.nhlayers) != len(args.hlayerssizes):
        parser.error('hlayerssizes lengths must be equal to nhlayers.')
    sizes = list(map(int, args.hlayerssizes))

    # Data processing
    data = DataLib.csvToArray("../heart_disease_dataset.csv")
    DataLib.shuffleData(data)
    dataNorm = DataLib.normalizeData(data,np.array([0,3,4,7,9,11]))

    if os.path.exists("../out.csv"):
        os.remove("../out.csv")

    #Neural network
    
    # second argument = batchSize
    # third argument = number of classes
    # fourth argument = number of hidden layers
    nn = NeuralNet(dataNorm,int(args.batchsize),2,int(args.nhlayers),sizes)

    nn.train(int(args.epoch))


if __name__ == "__main__":
    main()


