import java.io.*;
import java.util.*;

public class NeuralNetTrain{
   public static int EPOCHS = 1;
  //learning rate
   public static double ALPHA = 0.1;

   public static void main(String[] args) throws IOException, FileNotFoundException {
      Scanner in = new Scanner(System.in);
      System.out.println("Neural Network Training Program");
      Scanner initial, train;
      String output;
      boolean valid = false;

      System.out.println("Enter 1 for inputting your own filename. Anything else for dataset training.");
      int choice = 0;
      if(in.hasNextInt()){
        choice = in.nextInt();
      }
      //int choice = in.nextInt();

      //try catch
      if(choice == 1){
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
        EPOCHS = in.nextInt();

        System.out.print("Enter learning rate: ");
        while(!in.hasNextDouble()){
          System.out.println("Invalid input - please enter a decimal number.");
          System.out.print("Enter learning rate (double): ");
          in.next();
        }
        ALPHA = in.nextDouble();

        in.close();
      }

      //wdbc
      //initial = new Scanner(new File("files/wdbc/sample.NNWDBC.init.txt"));

      //mini file
      //train = new Scanner(new File("files/wdbc/mini/wdbc.mini_train.txt"));
      //output = "outputMiniTrain.txt";

      //full file
      //train = new Scanner(new File("files/wdbc/wdbc.train.txt"));
      //output = "outputTrain.txt";

      //grades
      // initial = new Scanner(new File("files/grades/sample.NNGrades.init.txt"));
      // train = new Scanner(new File("files/grades/grades.train.txt"));
      // output = "trainedGrades.txt";

      else{
      //dataset testing
        System.out.print("Enter number of hidden layers: ");
        int nh = in.nextInt();
        initial = initFile(in, nh);

        System.out.print("Enter number of epochs: ");
        while(!in.hasNextInt()){
          System.out.println("Invalid input - please enter an integer.");
          System.out.print("Enter number of epochs (integer): ");
          in.next();
        }
        EPOCHS = in.nextInt();

        System.out.print("Enter learning rate: ");
        while(!in.hasNextDouble()){
          System.out.println("Invalid input - please enter a decimal number.");
          System.out.print("Enter learning rate (double): ");
          in.next();
        }
        ALPHA = in.nextDouble();

        in.close();
        train = new Scanner(new File("files/dataset/trainingSet.txt"));
        output = "files/dataset/results/trainedDataset_"+nh+"_"+ALPHA+"_"+EPOCHS+".txt";
    }

    //initialize neural network with initial weights read from file
      Network network = ScannerMethods.readWeights(initial);

    //read in training data
      Example[] examples = ScannerMethods.readExampleData(train);
      //System.out.println(examples.length);
      /*for (Node ex : examples){
        System.out.println(ex);
      }*/
      backPropLearning(examples, network);

      printTrainedNetwork(network, output);

   }


   public static void backPropLearning(Example[] examples, Network network){
    //repeat for 100 epochs instead of stopping condition
      for(int e = 0; e < EPOCHS; e++){
      //iterate through examples; propogate forward to calculate output
         for(int k = 0; k < examples.length; k++){
         //copy inputs from first example to the inputLayer of Network
            for(int i = 0; i < examples[k].input.length+1; i++){
               if(i == 0) {
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
                  (examples[k].output[j] - network.outputLayer[j].output);
                  // if(e == 0 && k == 0)

                  // System.out.println("input: "+network.outputLayer[j].input);
                  // System.out.println("derivActivationFunction: "+ derivActivationFunction(network.outputLayer[j].input));
                  // System.out.println("example output: "+ examples[k].output[j]);
                  // System.out.println("activation: "+ network.outputLayer[j].output);
                  // System.out.println("deltaJ: " + deltaJ[j]);
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
               // if(e == 0 && k == 0)
                // System.out.println(i);
                // System.out.println("input: "+network.hiddenLayer[i].input);
                // System.out.println("derivActivationFunction: " + derivActivationFunction(network.hiddenLayer[i].input));
                // System.out.println("activation: " + network.hiddenLayer[i].output);
                // System.out.println("deltaI: " + deltaI[i]);
            }

          //update errors from deltaI and deltaJ for every weight
          //hidden to output layer
          //loop through output nodes
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

   public static Scanner initFile(Scanner in, int nh) throws FileNotFoundException, IOException{
    //exit out for -1 or something

      try{
          Scanner sc = new Scanner(new File("files/dataset/dataset.init_"+nh+".txt"));
          return sc;
      }
      catch(FileNotFoundException e){
          System.out.println("Creating initial file...");
          RandomFile.printInitialNetwork("files/dataset/dataset.init_"+nh+".txt",7,nh,3);
          Scanner sc = new Scanner(new File("files/dataset/dataset.init_"+nh+".txt"));
          return sc;
      }
   }

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
