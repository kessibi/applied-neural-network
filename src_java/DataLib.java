package fnn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;


public class DataLib {
  
  private static ArrayList<ArrayList<String>> data = new ArrayList<ArrayList<String>>();

  /**
   * It's better to use dedicated libraries such as OpenCSV.
   * In the meantime, just make sure there's no other lines than data lines
   * (no header, no blank lines and so on)
   */
  private static ArrayList<ArrayList<String>> importCSV(String filename, String separator) {
    String line = "";
    try {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      while ((line = br.readLine()) != null){
        ArrayList<String> dataAsString = new ArrayList<String>(Arrays.asList(line.split(separator)));
        data.add(dataAsString);
      }
      br.close();
    }catch(IOException e) {e.printStackTrace();}
    return data;
  }
  
  /**
   * Read data from CSV file and return them, shuffled, in an 2D array
   * array[i]    -> instance no i
   * array[i][j] -> attribute j of instance i
   * @param fileName
   * @param separator
   * @return 2D array
   */
  public static float[][] copyDataToArray(String fileName, String separator){
    importCSV(fileName, separator);
    return shuffleData();
  }
  
  private static float[][] shuffleData(){
    int nbInstances = data.size();
    int nbFeatures  = data.get(0).size();
    // Shuffle the data
    Collections.shuffle(data);
    float[][] arrayData = new float[nbInstances][nbFeatures];
    for (int i = 0; i < nbInstances; i++) {
      for (int j = 0; j < nbFeatures; j++)
        arrayData[i][j] = Float.parseFloat(data.get(i).get(j));
    }
    return arrayData;
  }
  
  public static void printData(){
    int i = 0;
    for (ArrayList<String> entry : data){
      System.out.println("["+ i++ +"]"+entry);
    }
  }
  
  public static void exportDataToCSV(String filename, String data) {
    try {
      Writer out = new BufferedWriter(new FileWriter(filename));
      out.write(data);
      out.close();
    } catch (IOException e) { e.printStackTrace(); }
  }
  
}
