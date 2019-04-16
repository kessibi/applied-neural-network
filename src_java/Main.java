package fnn;

public class Main {

  public static void main(String[] args) {
    
    // Copy data from file, shuffle them and write them in 2D array
    float[][] data = DataLib.copyDataToArray("iris_num.data" /*filename*/, "," /*separator*/);
    
    //DataLib.printData();
    
    NeuralNet nn = new NeuralNet(data, 4 /*batchSize*/, 3 /*nb classes*/);

    nn.train(800/*nb of epochs*/);
  }

}
