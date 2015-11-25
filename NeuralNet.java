import java.io.*;
import java.util.*;

public class NeuralNet{

  public static void main(String[] args) throws IOException, FileNotFoundException {
    Scanner input = new Scanner(System.in);
    Scanner initial, train;
    String output;
    boolean valid = false;
    //check for valid filename
    initial = checkFile(input, 0);
    train = checkFile(input, 1);
    train.close();
    System.out.print("Enter output filename: ");
    output = input.next(); //create a file with this name
    input.close();
    //initialize neural network with initial weights read from file
    Network network = readInitialFile(initial);

    printOutput(network, output);

  }
  //check for valid filename input
  public static Scanner checkFile(Scanner input, int type) throws FileNotFoundException{
    //exit out for -1 or something

    String print;
    switch(type){
      case 0: print = "Enter initial neural network filename: ";
        break;
      case 1: print = "Enter training set filename: ";
        break;
      default: print = "argument error";
        break;
    }

    System.out.print(print);
    while(true){
      try{
        Scanner sc = new Scanner(new File(input.next()));
        return sc;
      }
      catch(FileNotFoundException e){
        System.out.println("File Not Found :(");
        System.out.println(print);
      }
    }
  }
  public static Network readInitialFile(Scanner sc) throws FileNotFoundException{

    //input nodes
    int ni = sc.nextInt();
    //hidden nodes
    int nh = sc.nextInt();
    //output nodes
    int no = sc.nextInt();
    Network network = new Network(ni, nh, no);
    //read weights from inputs to hidden node
    for(int k = 0; k < nh; k++){
      // String w = sc.nextLine();
      // float[] weights = Float.parseFloat(w.split(" "));
      //instantiate new hidden node
      network.hiddenLayer[k] = new Node(ni+1);
      for(int i = 0; i < ni+1; i++){
        //read inputs from input layer to current hidden node
        network.hiddenLayer[k].input[i] = sc.nextFloat();
      }
    }

    //read weights from hidden node to output nodes
    for(int k = 0; k < no; k++){
      //String w = sc.nextLine();
      //float[] weights = Float.parseFloat(w.split(" "));
      //instantiate new hidden node
      network.outputLayer[k] = new Node(ni+1);
      for(int i = 0; i < nh+1; i++){
        //read inputs from input layer to current hidden node
        network.outputLayer[k].input[i] = sc.nextFloat();
      }
    }
    sc.close();
    return network;
  }
  //for trained neural network
  public static void printOutput(Network network, String output) throws IOException {
    PrintWriter pw = new PrintWriter(new FileWriter(output));
    int ni = network.inputLayer.length;
    int nh = network.hiddenLayer.length;
    int no = network.outputLayer.length;
    pw.println(ni + " " + nh + " " + no);
    //print weights from input to hidden layer (including bias weight)
    for(int k = 0; k < nh; k++){
        for(int i = 0; i < ni+1; i++){
          pw.print(network.hiddenLayer[k].input[i] + " ");
        }
      pw.println();
    }
    //print weights from hidden layer to outputs (including bias weight)
    for(int k = 0; k < no; k++){
      for (int i = 0; i < nh+1; i++) {
        pw.print(network.outputLayer[k].input[i] + " ");
      }
      pw.println();
    }
    pw.close();
  }
  private static class Network {
    Node[] inputLayer;
    Node[] hiddenLayer;
    Node[] outputLayer;
    public Network(int ni, int nh, int no){
      inputLayer = new Node[ni];
      hiddenLayer = new Node[nh];
      outputLayer = new Node[no];
    }
  }

  //for each
  private static class Node {
    float[] input;
    public Node(int numWeights){
      input = new float[numWeights];
    }
  }
}
