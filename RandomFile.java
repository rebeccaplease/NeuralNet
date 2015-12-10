import java.io.*;
import java.util.*;

/**
 * Creates a files of random weights for an initial neural network [0,1] with
 * one hidden layer.
 *
 * Prompts user for number of initial, hidden, and output nodes.
 * Prompts for output text file name.
 *
 * Prints a text file of the pseudo random initial neural network.
 */
public class RandomFile{
	public static void main(String[] args) throws IOException, FileNotFoundException {
      Scanner in = new Scanner(System.in);
      int ni, nh, no;
      String output;

      System.out.println("Generate pseudo random weights for initial neural network with one hidden layer");
      System.out.print("Enter the number of initial nodes: ");
      while(!in.hasNextInt()){
         System.out.println("Not an integer!");
         System.out.print("Enter the number of initial nodes: ");
         in.next();
      }
      ni = in.nextInt();

      System.out.print("Enter the number of hidden nodes: ");
      while(!in.hasNextInt()){
         System.out.println("Not an integer!");
         System.out.print("Enter the number of hidden nodes: ");
         in.next();
      }
      nh = in.nextInt();

      System.out.print("Enter the number of output nodes: ");
      while(!in.hasNextInt()){
         System.out.println("Not an integer!");
         System.out.print("Enter the number of output nodes: ");
         in.next();
      }
      no = in.nextInt();

      System.out.print("Enter output filename: ");
      output = in.next(); //create a file with this name
      in.close();

      printInitialNetwork(output, ni, nh, no);
   }

  //print pseudo random initial neural network
   public static void printInitialNetwork(String output, int ni, int nh, int no) throws IOException {
      PrintWriter pw = new PrintWriter(new FileWriter(output));

      pw.println((ni) + " " + (nh) + " " + no);
    //print weights from input to hidden layer (including bias weight)
      for(int k = 0; k < nh; k++){
         for(int i = 0; i < ni+1; i++){
            pw.printf("%.3f",Math.random());
            pw.print(" ");

         }
         pw.println();
      }
    //print weights from hidden layer to outputs (including bias weight)
      for(int k = 0; k < no; k++){
         for (int i = 0; i < nh+1; i++) {
            pw.printf("%.3f",Math.random());
            pw.print(" ");
         }
         pw.println();
      }
      pw.close();
   }
}
