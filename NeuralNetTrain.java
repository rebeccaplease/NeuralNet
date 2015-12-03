import java.io.*;
import java.util.*;

public class NeuralNetTrain{
   public static int EPOCHS = 100;
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
      //mini file
      //train = new Scanner(new File("files/mini/wdbc.mini_train.txt"));
      //output = "outputMiniTrain.txt";

      //full file
      train = new Scanner(new File("files/wdbc.train.txt"));
      output = "outputTrain.txt";
    //initialize neural network with initial weights read from file
      Network network = readInitialFile(initial);

    //read in training data
      Node[] examples = readTrainingData(train);
      //System.out.println(examples.length);
      /*for (Node ex : examples){
        System.out.println(ex);
      }*/
      backPropLearning(examples, network);

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
      //+1 for initial and hidden for bias
      Network network = new Network(ni+1, nh+1, no);
    //read weights from inputs to hidden node
      for(int k = 0; k < nh+1; k++){
        
      //instantiate new hidden node
         network.hiddenLayer[k] = new Node(ni+1);
         for(int i = 0; i < ni+1; i++){
         //read inputs from input layer to current hidden node
            if(k > 0){
               network.hiddenLayer[k].inputWeight[i] = sc.nextDouble();
            }
             //initialize input layer
            if(k==0 && i < ni+1){
               network.inputLayer[i] = new Node();
            }
         }
      }

    //read weights from hidden node to output nodes
      for(int k = 0; k < no; k++){
      //instantiate new output node
         network.outputLayer[k] = new Node(nh+1);
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
      Node[] training = new Node[n];
    //loop through each training example
      for(int k = 0; k < n; k++){
         training[k] = new Node(ni, no);
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
   public static void backPropLearning(Node[] examples, Network network){
    //repeat for 100 epochs instead of stopping condition
      for(int e = 0; e < EPOCHS; e++){
      //iterate through examples; propogate forward to calculate output
         for(int k = 0; k < examples.length; k++){
         //copy inputs from first example to the inputLayer of Network
            for(int i = 0; i < examples[k].inputWeight.length+1; i++){
               if(i == 0) {
                  network.inputLayer[0].output = -1;
                  network.hiddenLayer[0].output = -1;
               }
               else{
               //input for examples (not inputWeight)
               //0 to inputWeight.length for inputWeight
                  network.inputLayer[i].output = examples[k].inputWeight[i-1];
               }
            }
         //for each hidden layer (only 1 hidden layer)
         //for(int l = 2; l < network.numLayers; l++){

          //for each node in hidden layer
            for(int j = 1; j < network.hiddenLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
               double inj = 0;
               // if(e == 0 && k == 0){
               //   System.out.print(" epoch="+e+ " example no. "+k);
               //   System.out.print(" output node: " + j +" ");
               // }

               //bias input and weight
               for(int i = 0; i < network.inputLayer.length; i++){
               //generalize to multiple hidden layers
                  double act = network.inputLayer[i].output;
                  double weight = network.hiddenLayer[j].inputWeight[i];
                  double add = act*weight;
                  inj += add;
                  // if(e == 0 && k == 0){
                  //   System.out.print("activation: " + act);
                  //   System.out.print(" weight: " + weight);
                  //   System.out.println(" multiply: " + add);
                  // }
               }

               network.hiddenLayer[j].setInput(inj);

            //compute activation weight of jth hidden node in this layer
               network.hiddenLayer[j].setActivation(activationFunction(inj));
               // if(e == 0 && k == 0){
               //   System.out.println("inJ " + inj);
               //   System.out.println("output: "+activationFunction(inj));
               // }

  
            }

          //for each node in output layer
            for(int j = 0; j < network.outputLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
               double inj = 0;
               //bias input and weight
               for(int i = 0; i < network.hiddenLayer.length; i++){
               //generalize to multiple hidden layers
                  inj += network.hiddenLayer[i].output*network.outputLayer[j].inputWeight[i];
               }
               network.outputLayer[j].setInput(inj);
            //compute activation weight of jth hidden node in this layer
               network.outputLayer[j].setActivation(activationFunction(inj));
        
            }


          //back propogate error
          // output layer error distribution
            double[] deltaJ = new double[network.outputLayer.length];
            //deltaJ[0] = derivActivationFunction(-1)*
               //(examples[k].output - network.outputLayer[0].output);
            //for each node in the output layer
            for(int j = 0; j < network.outputLayer.length; j++) {
               deltaJ[j] = derivActivationFunction(network.outputLayer[j].input)*
                  (examples[k].output - network.outputLayer[j].output);
                  // if(e == 0 && k == 0)
                  //   System.out.println("deltaJ: " + deltaJ[j]);
            }

            double[] deltaI = new double[network.hiddenLayer.length];
          // //input layer to hidden layer
          // //for each node in the hidden layer
          //   for(int i = 1; i < network.hiddenLayer.length; i++){
          //      
          //      double err = 0;
          //      for(int j = 0; j < network.outputLayer.length; j++){
          //      //loop through output layer error
          //         err += network.hiddenLayer[i].inputWeight[j]*deltaJ[j];
          //      
          //      }
          //      deltaI[i] = derivActivationFunction(network.hiddenLayer[i].input)*err;
          //      // if(e == 0 && k == 0)
          //      //      System.out.println("deltaI: " + deltaI[i]);
          //   }

            //hidden node error distribution
            for(int j = 0; j < network.outputLayer.length; j++){
               double err = 0;
               for(int i = 1; i < network.hiddenLayer.length; i++){
               //loop through output layer error
                  err = network.outputLayer[j].inputWeight[i]*deltaJ[j];
                  deltaI[i] = derivActivationFunction(network.hiddenLayer[i].input)*err;
                }
            }


          //update errors from deltaI and deltaJ for every weight
          //hidden to output layer
          //loop through output layers
            for(int i = 0; i < network.outputLayer.length; i++){
            //loop through hidden layer outputs (activation)
               for(int j = 0; j < network.hiddenLayer.length; j++){
                  network.outputLayer[i].inputWeight[j] += ALPHA*network.hiddenLayer[j].output*deltaJ[i];
               }
            }
          //input to hidden layer
          //loop through hidden nodes
            for(int i = 1; i < network.hiddenLayer.length; i++){
               //network.hiddenLayer[i].inputWeight[0] += ALPHA*(-1)*
               //loop through each output
               for(int j = 0; j < network.inputLayer.length; j++){
                  network.hiddenLayer[i].inputWeight[j] += ALPHA*network.inputLayer[j].output*deltaI[i];
               }
            }

         //}
         }
      }
      //return network;
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
      pw.println((ni-1) + " " + (nh-1) + " " + no);
    //print weights from input to hidden layer (including bias weight)
      for(int k = 1; k < nh; k++){
         for(int i = 0; i < ni; i++){
            pw.printf("%.3f",network.hiddenLayer[k].inputWeight[i]);
            pw.print(" ");

         }
         pw.println();
      }
    //print weights from hidden layer to outputs (including bias weight)
      for(int k = 0; k < no; k++){
         for (int i = 0; i < nh; i++) {
            pw.printf("%.3f",network.outputLayer[k].inputWeight[i]);
            pw.print(" ");
         }
         pw.println();
      }
      pw.close();
   }
}
