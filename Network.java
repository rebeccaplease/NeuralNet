/**
 * For holding nodes in each layer (input, hidden, and output)
 */
public class Network {
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
