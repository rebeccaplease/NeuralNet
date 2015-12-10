/**
 * For holding parameters for each node.
 */
public class Node {
   double[] inputWeight;
   double output; //activation
   double input = 0;
   public Node (){
   }

   public Node(int numWeights) {
      inputWeight = new double[numWeights];
   }
   public Node(int ni, int no) {
      inputWeight = new double[ni];
      input = 0;
   }
   public void setInput(double i){
      input = i;
   }
   public void setActivation(double a){
      output = a;
   }
   public String toString(){
     String s = "";
     for( double i : inputWeight){
        s+= "inputWeight: " + i;
     }
     s += "\noutput " + output;
     return s;
   }
}
