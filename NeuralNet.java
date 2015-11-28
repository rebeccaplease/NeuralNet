import java.io.*;
import java.util.*;

public class NeuralNet{
  public static int EPOCHS = 100;

  public static void main(String[] args) throws IOException, FileNotFoundException {
    Scanner in = new Scanner(System.in);
    Scanner initial, train;
    String output;
    boolean valid = false;
    //check for valid filename
    initial = checkFile(in, 0);
    train = checkFile(in, 1);

    System.out.print("Enter output filename: ");
    output = in.next(); //create a file with this name
    in.close();
    //initialize neural network with initial weights read from file
    Network network = readInitialFile(initial);

    //read in training data
    Example[] examples = readTrainingData(train);

    network = backPropLearning(examples, network);
    
    printOutput(network, output);

  }
  //check for valid filename input
  public static Scanner checkFile(Scanner in, int type) throws FileNotFoundException{
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
        Scanner sc = new Scanner(new File(in.next()));
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
        network.hiddenLayer[k].inputWeight[i] = sc.nextFloat();
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
        network.outputLayer[k].inputWeight[i] = sc.nextFloat();
      }
    }
    sc.close();
    return network;
  }

  public static Example[] readTrainingData(Scanner sc){
    int n = sc.nextInt();
    int ni = sc.nextInt();
    int no = sc.nextInt();
    Example[] training = new Example[];
    //loop through each training example
    for(int k = 0; k < n; k++){
      training[k] = new Node(ni, no);
      //input nodes
      for(int i = 0; i < ni; i++){
        training[k].inputWeight[i] = train.nextFloat();
      }
      //output nodes
      for(int i = 0; i < no; i++){
        training[k].output[i] = train.nextFloat();
      }
    }

    train.close();
    return training;
  }
  public static Network backPropLearning(Example[] examples, Network network){
    //repeat for 100 epochs instead of stopping condition
    for(int e = 0; e < EPOCHS; e++){
      //iterate through examples; propogate forward to calculate output
      for(int k = 0; k < examples.length; k++){
        //copy inputs from first example to the inputLayer of Network
        for(int i = 0; i < examples[k].inputWeight.length; i++){
          network.inputLayer[i] = examples[k].inputWeight[i];
        }
        //for each hidden layer (only 1 hidden layer)
        //for(int l = 2; l < network.numLayers; l++){

          //for each node in hidden layer
          for(int j = 0; j < network.hiddenLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
            float inj = 0;
            for(int i = 0; i < network.inputLayer.length; i++){
              //generalize to multiple hidden layers
              inj += network.inputLayer[i]*network.hiddenLayer[j].inputWeight[i];
            }
            network.hiddenLayer[j].setInput(inj);
            //compute activation weight of jth hidden node in this layer
            network.hiddenLayer[j].setActivation(activationFunction(inj));
          }

          //for each node in output layer
          for(int j = 0; j < network.outputLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
            float inj = 0;
            for(int i = 0; i < network.hiddenLayer.length; i++){
              //generalize to multiple hidden layers
              inj += network.hiddenLayer[i]*network.outputLayer[j].inputWeight[i];
            }
            network.outputLayer[j].setInput(inj);
            //compute activation weight of jth hidden node in this layer
            network.outputLayer[j].setActivation(activationFunction(inj));
          }


          //back propogate error
          //hidden layer to output layer
          float[] deltaJ = new float[network.outputLayer.length];
          for(int j = 0; j < network.outputLayer.length; j++){
            deltaJ[j] = derivActivationFunction(network.outputLayer[j].input)*
              (examples[j].output - examples[j].activation);
          }

          float[] deltaI = new float[network.hiddenLayer.length];
          //input layer to hidden layer
          for(int i = 0; i < network.hiddenLayer.length; i++){
            float err;
            for(int j = 0; j < network.outputLayer.length; j++){
              //loop through output layer error
              err += network.hiddenLayer[i].inputWeight[j]*deltaJ[j];
            }
            deltaI[i] = network.hiddenlayer[i].input*err;
          }

          //update errors from deltaI and deltaJ
          //input to hidden layer
          for(int i = 0; i < network.hiddenLayer.length; i++){
            for(int j = 0; j < network.hiddenLayer[i].length; j++){
              network.hiddenLayer[i].inputWeight[j] += deltaI[j];
            }
          }
          //hidden to output layer
          for(int i = 0; i < network.outputLayer.length; i++){
            for(int j = 0; j < network.outputLayer[i].length; j++){
              network.outputLayer[i].inputWeight[j] += deltaJ[j];
            }
          }
        //}
      }
    }
    return network;
  }
  public static float activationFunction(float in){
    return 0;
  }
  public static float derivActivationFunction(float in){

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
          pw.print(network.hiddenLayer[k].inputWeight[i] + " ");
        }
      pw.println();
    }
    //print weights from hidden layer to outputs (including bias weight)
    for(int k = 0; k < no; k++){
      for (int i = 0; i < nh+1; i++) {
        pw.print(network.outputLayer[k].inputWeight[i] + " ");
      }
      pw.println();
    }
    pw.close();
  }
  private static class Network {
    //arraylist
    Node[] inputLayer;
    Node[] hiddenLayer;
    Node[] outputLayer;
    int numLayers = 3;
    public Network(int ni, int nh, int no){
      inputLayer = new Node[ni];
      hiddenLayer = new Node[nh];
      outputLayer = new Node[no];
    }
  }
  // private static class Network {
  //   //arraylist
  //   ArrayList<Node[]> layers = new ArrayList<Node[]>();
  //   int numLayers = 3;
  //   public Network(int ni, HiddenLayers h, int no){
  //     layers.add(new Node[ni]); //inputLayer
  //     //hidden layer(s)
  //     for(k = 0; k < h.nodes.length; k++){
  //       hiddenLayer = new Node[nh];
  //       layers.add(new Node[h.])
  //     }
  //
  //     layers.add(new Node[no]); //outputLayer
  //   }
  // }
  // private static class HiddenLayers{
  //   int[] nodes;
  //   public HiddenLayers(int[] )
  //
  // }
  //for each
  private static class Node {
    float[] inputWeight;
    float activation = 0;
    float input = 0;
    public Node(int numWeights){
      input = new float[numWeights];
    }
    public void setInput(float i){
      input = i;
    }
    public void setActivation(float a){
      activation = a;
    }

  }
  private class Example extends Node{
    float[] output;
    public Example(int ni, int no){
      super.input = new float[ni];
      output = new float[no];
    }
  }
}
