public class Network {
   //arraylist
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
// private static class Network {
//   //arraylist
//   ArrayList<Node[]> layers = new ArrayList<Node[]>();
//   int numLayers = 3;
//   public Network(int ni, HiddenLayers h, int no){
//     layers.add(new Node[ni]); //inputLayer
//     //hidden layer(s)
//     for(k = 0; k < h.nodes.length; k++){
//       hiddenLayer = new Node[nh];
//       layers.add(new Node[h.])
//     }
//
//     layers.add(new Node[no]); //outputLayer
//   }
// }
// private static class HiddenLayers{
//   int[] nodes;
//   public HiddenLayers(int[] )
//
// }
//for each
