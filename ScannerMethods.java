import java.io.*;
import java.util.*;

public class ScannerMethods{
	 //check for valid filename input
   public static Scanner checkFile(Scanner in, int type) throws FileNotFoundException{
    //exit out for -1 or something

      String print;
      switch(type){
         case 0: print = "Enter initial neural network filename: ";
            break;
         case 1: print = "Enter training set filename: ";
            break;
         case 2: print = "Enter trained neural network filename: ";
            break;
         case 3: print = "Enter test set filename: ";
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
            System.out.print(print);
         }
      }
   }
   public static Network readWeights(Scanner sc) throws FileNotFoundException{

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

   public static Example[] readExampleData(Scanner sc){
      int n = sc.nextInt();
      int ni = sc.nextInt();
      int no = sc.nextInt();
      Example[] training = new Example[n];
    //loop through each training example
      for(int k = 0; k < n; k++){
         training[k] = new Example(ni, no);
      //input nodes
         for(int i = 0; i < ni; i++){
            training[k].input[i] = sc.nextDouble();
         }
      //output nodes
         for(int i = 0; i < no; i++){
            training[k].output[i] = sc.nextDouble();
         }
      }

      sc.close();
      return training;
   }
}