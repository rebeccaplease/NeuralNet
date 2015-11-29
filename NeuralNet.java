import java.io.*;
import java.util.*;
//import src.*;

public class NeuralNet{
   public static int EPOCHS = 1;
  //learning rate
   public static double ALPHA = 0.1;

   public static void main(String[] args) throws IOException, FileNotFoundException {
      //Scanner in = new Scanner(System.in);
      Scanner initial, train;
      String output;
      boolean valid = false;
    //check for valid filename
      /*initial = checkFile(in, 0);
      train = checkFile(in, 1);
      System.out.print("Enter output filename: ");
      output = in.next(); //create a file with this name
      in.close();*/
      initial = new Scanner(new File("files/sample.NNWDBC.init.txt"));
      train = new Scanner(new File("files/mini/wdbc.mini_train.txt"));
      
      output = "output.txt";
    //initialize neural network with initial weights read from file
      Network network = readInitialFile(initial);
   
    //read in training data
      Node[] examples = readTrainingData(train);
   
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
      // double[] weights = double.parsedouble(w.split(" "));
      //instantiate new hidden node
         network.hiddenLayer[k] = new Node(ni+1);
         for(int i = 0; i < ni+1; i++){
         //read inputs from input layer to current hidden node
            network.hiddenLayer[k].inputWeight[i] = sc.nextDouble();
            if(k==0 && i < ni){
               network.inputLayer[i] = new Node();
            }
         }
      }
   
    //read weights from hidden node to output nodes
      for(int k = 0; k < no; k++){
      //String w = sc.nextLine();
      //double[] weights = double.parsedouble(w.split(" "));
      //instantiate new hidden node
         network.outputLayer[k] = new Node(ni+1);
         for(int i = 0; i < nh+1; i++){
         //read inputs from input layer to current hidden node
            network.outputLayer[k].inputWeight[i] = sc.nextDouble();
         }
      }
      sc.close();
      return network;
   }

   public static Node[] readTrainingData(Scanner sc){
      int n = sc.nextInt();
      int ni = sc.nextInt();
      int no = sc.nextInt();
      Node[] training = new Example[n];
    //loop through each training example
      for(int k = 0; k < n; k++){
         training[k] = new Example(ni, no);
      //input nodes
         for(int i = 0; i < ni; i++){
            training[k].inputWeight[i] = sc.nextDouble();
         }
      //output nodes
         for(int i = 0; i < no; i++){
            training[k].output = sc.nextDouble();
         }
      }
   
      sc.close();
      return training;
   }
   public static Network backPropLearning(Node[] examples, Network network){
    //repeat for 100 epochs instead of stopping condition
      for(int e = 0; e < EPOCHS; e++){
      //iterate through examples; propogate forward to calculate output
         for(int k = 0; k < examples.length; k++){
         //copy inputs from first example to the inputLayer of Network
            for(int i = 0; i < examples[k].inputWeight.length+1; i++){
               if(i == 0) {
                  network.inputLayer[0].output = -1;
               }
               else{
               //input for examples (not inputWeight)
                  network.inputLayer[i].output = examples[k].inputWeight[i-1];
               }
            }
         //for each hidden layer (only 1 hidden layer)
         //for(int l = 2; l < network.numLayers; l++){
         
          //for each node in hidden layer
            for(int j = 0; j < network.hiddenLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
               double inj = 0;
               //bias input and weight
               //inj += -1*network.hiddenLayer[j].inputWeight[0];
               for(int i = 0; i < network.inputLayer.length; i++){
               //generalize to multiple hidden layers
                  inj += network.inputLayer[i].output*network.hiddenLayer[j].inputWeight[i+1];
               }
               network.hiddenLayer[j].setInput(inj);
            //compute activation weight of jth hidden node in this layer
               network.hiddenLayer[j].setActivation(activationFunction(inj));
            }
         
          //for each node in output layer
            for(int j = 0; j < network.outputLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
               double inj = 0;
               //bias input and weight
               //inj += -1*network.outputLayer[j].inputWeight[0];
               for(int i = 0; i < network.hiddenLayer.length; i++){
               //generalize to multiple hidden layers
                  inj += network.hiddenLayer[i].output*network.outputLayer[j].inputWeight[i+1];
               }
               network.outputLayer[j].setInput(inj);
            //compute activation weight of jth hidden node in this layer
               network.outputLayer[j].setActivation(activationFunction(inj));
            }
         
         
          //back propogate error
          //hidden layer to output layer
            double[] deltaJ = new double[network.outputLayer.length+1];
            //deltaJ[0] = derivActivationFunction(-1)*
               //(examples[k].output - network.outputLayer[0].output);
            for(int j = 0; j < network.outputLayer.length; j++) {
               deltaJ[j+1] = derivActivationFunction(network.outputLayer[j+1].input)*
                  (examples[k].output - network.outputLayer[j+1].output);
            }
         
            double[] deltaI = new double[network.hiddenLayer.length+1];
          //input layer to hidden layer
            for(int i = 0; i < network.hiddenLayer.length; i++){
               double err = 0;
               for(int j = 0; j < network.outputLayer.length; j++){
               //loop through output layer error
                  err += network.hiddenLayer[i].inputWeight[j]*deltaJ[j];
               }
               deltaI[i] = derivActivationFunction(network.hiddenLayer[i].input)*err;
            }
         
          //update errors from deltaI and deltaJ
          //input to hidden layer
            for(int i = 0; i < network.hiddenLayer.length; i++){
               //network.hiddenLayer[i].inputWeight[0] += ALPHA*(-1)*
               for(int j = 0; j < network.hiddenLayer.length; j++){
                  network.hiddenLayer[i].inputWeight[j+1] += ALPHA*network.hiddenLayer[i].output*deltaI[j];
               }
            }
          //hidden to output layer
            for(int i = 0; i < network.outputLayer.length; i++){
               for(int j = 0; j < network.outputLayer.length; j++){
                  network.outputLayer[i].inputWeight[j+1] += ALPHA*network.outputLayer[i].output*deltaJ[j];
               }
            }
         //}
         }
      }
      return network;
   }
  //sigmoid function
   public static double activationFunction(double x){
      return 1.0/(1.0+Math.pow(Math.E, -x));
   }
   public static double derivActivationFunction(double x){
      double g = activationFunction(x);
      return g*(1.0-g);
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
            pw.printf("%.3f",network.hiddenLayer[k].inputWeight[i]);
            pw.print(" ");
            
         }
         pw.println();
      }
    //print weights from hidden layer to outputs (including bias weight)
      for(int k = 0; k < no; k++){
         for (int i = 0; i < nh+1; i++) {
            pw.printf("%.3f",network.outputLayer[k].inputWeight[i]);
            pw.print(" ");
         }
         pw.println();
      }
      pw.close();
   }
}
