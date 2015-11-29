public class Node {
   double[] inputWeight;
   double output;
   //double output = 0;
   double input = 0;
   public Node(){
   
   }
   public Node(int numWeights) {
      inputWeight = new double[numWeights];
   }
   public Node(int ni, int no) {
      input = 0;
   }
   public void setInput(double i){
      input = i;
   }
   public void setActivation(double a){
      output = a;
   }

}
