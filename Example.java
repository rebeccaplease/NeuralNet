public class Example {
   double[] input;
   double[] output; //activation 

   public Example(int ni, int no) {
      input = new double[ni];
      output = new double[no];
   }
   public String toString(){
     String s = "";
     for( double i : input){
        s+= "input: " + i;
     }
     s += "\n";
     for( double i : output){
       s += "output " + output;
     }
     return s;
   }
}
