import java.io.*;
import java.util.*;

public class NeuralNetTrain{

   /**
    * Prompts user for initial neural network file, training set,
    * output file name, number of epochs, and learning rate.
    *
    * Prints a text file of the trained neural network with these parameters.
    */
   public static void main(String[] args) throws IOException, FileNotFoundException {
      Scanner in = new Scanner(System.in);
      System.out.println("---Neural Network Training Program---");
      Scanner initial, train;
      String output;
      boolean valid = false;
      int epochs;
      double alpha;

      //check for valid filename
      initial = ScannerMethods.checkFile(in, 0);
      train = ScannerMethods.checkFile(in, 1);
      System.out.print("Enter output filename: ");
      output = in.next(); //create a file with this name

      System.out.print("Enter number of epochs: ");
      while(!in.hasNextInt()){
        System.out.println("Invalid input - please enter an integer.");
        System.out.print("Enter number of epochs (integer): ");
        in.next();
      }
      epochs = in.nextInt();

      System.out.print("Enter learning rate: ");
      while(!in.hasNextDouble()){
        System.out.println("Invalid input - please enter a decimal number.");
        System.out.print("Enter learning rate (double): ");
        in.next();
      }
       alpha = in.nextDouble();

      in.close();

    //initialize neural network with initial weights read from file
      Network network = ScannerMethods.readWeights(initial);

    //read in training data
      Example[] examples = ScannerMethods.readExampleData(train);

      backPropLearning(examples, network, epochs, alpha);

      printTrainedNetwork(network, output);

   }

   /**
    * Trains neural network with training examples.
    * One hidden layer. Backpropogate error.
    * Epochs and learning rate are specified by user input.
    *
    * @param    example   array of training examples
    * @param    network   initial neural network
    * @param    epochs    number of iterations through the entire dataset
    * @param    alpha     learning rate
    */
   public static void backPropLearning(Example[] examples, Network network, int epochs, double alpha){
    //repeat for number of epochs instead of stopping condition
      for(int e = 0; e < epochs; e++){
      //iterate through examples; propogate forward to calculate output
         for(int k = 0; k < examples.length; k++){
         //copy inputs from first example to the inputLayer of Network
            for(int i = 0; i < examples[k].input.length+1; i++){
               if(i == 0) {
                 // set fixed inputs of -1
                  network.inputLayer[0].output = -1;
                  network.hiddenLayer[0].output = -1;
               }
               else{
               //input for examples (not inputWeight)
               //0 to inputWeight.length for inputWeight
                  network.inputLayer[i].output = examples[k].input[i-1];
               }
            }
         //for each hidden layer (only 1 hidden layer)
         //for(int l = 2; l < network.numLayers; l++){

          //for each node in hidden layer
            for(int j = 1; j < network.hiddenLayer.length; j++){
            //sum weights and inputs from input layer to hidden layer
               double inj = 0;
               //bias input and weight
               for(int i = 0; i < network.inputLayer.length; i++){
               //generalize to multiple hidden layers
                  double act = network.inputLayer[i].output;
                  double weight = network.hiddenLayer[j].inputWeight[i];
                  double add = act*weight;
                  inj += add;
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
            //for each node in the output layer
            for(int j = 0; j < network.outputLayer.length; j++) {
               deltaJ[j] = derivActivationFunction(network.outputLayer[j].input)*
                  (examples[k].output[j] - network.outputLayer[j].output);
            }

            double[] deltaI = new double[network.hiddenLayer.length];
          //input layer to hidden layer
          //for each node in the hidden layer
            for(int i = 1; i < network.hiddenLayer.length; i++){
               double err = 0;
               for(int j = 0; j < network.outputLayer.length; j++){
               //loop through output layer error
                  err += network.outputLayer[j].inputWeight[i]*deltaJ[j];
               }
                deltaI[i] = derivActivationFunction(network.hiddenLayer[i].input)*err;
            }

          //update errors from deltaI and deltaJ for every weight
          //hidden to output layer
          //loop through output nodes
            for(int i = 0; i < network.outputLayer.length; i++){
            //loop through hidden layer outputs (activation)
               for(int j = 0; j < network.hiddenLayer.length; j++){
                  network.outputLayer[i].inputWeight[j] += alpha*network.hiddenLayer[j].output*deltaJ[i];
               }
            }
          //input to hidden layer
          //loop through hidden nodes
            for(int i = 1; i < network.hiddenLayer.length; i++){
               //loop through each output
               for(int j = 0; j < network.inputLayer.length; j++){
                  network.hiddenLayer[i].inputWeight[j] += alpha*network.inputLayer[j].output*deltaI[i];
               }
            }
         //}
         }
      }
      //return network;
   }


   /**
    * Returns sigmoid function with x argument (for computing activation).
    *
    * @param   x input
    * @return  sigmoid(x)
    */
  //sigmoid function
   public static double activationFunction(double x){
      return 1.0/(1.0+Math.pow(Math.E, -x));
   }
   /**
    * Returns derivative of sigmoid function with x argument (for error propogation).
    *
    * @param   x  input
    * @return sigmoid'(x)
    */
   public static double derivActivationFunction(double x){
      double g = activationFunction(x);
      return g*(1.0-g);
   }

   /**
    * Prints the given network based on the text file format.
    *
    * @param    network   network representation
    * @param    output    file name for text file
    */
  //for trained neural network
   public static void printTrainedNetwork(Network network, String output) throws IOException {
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
