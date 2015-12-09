import java.io.*;
import java.util.*;

public class NeuralNetTest{

	public static void main(String[] args) throws IOException, FileNotFoundException {

		Scanner in = new Scanner(System.in);
		System.out.println("Neural Network Testing Program");
		Scanner trained, test;
		String output;
		boolean valid = false;

		System.out.println("Enter 1 for inputting your own filename. Anything else for dataset training.");
		int choice = 0;
		if(in.hasNextInt()){
			choice = in.nextInt();
		}
		//check for valid filename
		if(choice == 1){
			trained = ScannerMethods.checkFile(in, 2);
			test = ScannerMethods.checkFile(in, 3);
			System.out.print("Enter output filename: ");
			output = in.next(); //create a file with this name
			in.close();
		}

		//full file
		//trained = new Scanner(new File("outputGrades.txt"));
		//test = new Scanner(new File("files/grades/grades.test.txt"));
		//output = "resultGrades.txt";

		//grades
		// trained = new Scanner(new File("trainedGrades.txt"));
		// test = new Scanner(new File("files/grades/grades.test.txt"));
		// output = "outputGrades.txt";

		//dataset
		else{
			System.out.print("Enter number of hidden layers: ");
			int nh = in.nextInt();

			System.out.print("Enter number of epochs: ");
			while(!in.hasNextInt()){
				System.out.println("Invalid input - please enter an integer.");
				System.out.print("Enter number of epochs (integer): ");
				in.next();
			}
			int epochs = in.nextInt();

			System.out.print("Enter learning rate: ");
			while(!in.hasNextDouble()){
				System.out.println("Invalid input - please enter a decimal number.");
				System.out.print("Enter learning rate (double): ");
				in.next();
			}
			double alpha = in.nextDouble();

			in.close();
			trained = new Scanner(new File("files/dataset/results/trainedDataset_"+nh+"_"+alpha+"_"+epochs+".txt"));
			test = new Scanner(new File("files/dataset/testSet.txt"));
			output = "files/dataset/results/outputDataset_"+nh+"_"+alpha+"_"+epochs+".txt";
		}


        //read trained weights
		Network network = ScannerMethods.readWeights(trained);

    	//read in test data
		Example[] examples = ScannerMethods.readExampleData(test);

		Metric[] metrics = testing(examples, network);

		printResults(metrics, network, output);
	}

	public static Metric[] testing(Example[] examples, Network network){
		Metric[] metrics = new Metric[network.outputLayer.length];
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

            //calculate A B C D
			for(int j = 0; j < network.outputLayer.length; j++){
				if(k == 0){
					metrics[j] = new Metric();
				}
				int predicted = (int) Math.round(network.outputLayer[j].output);

				int expected = (int) examples[k].output[j];
				if(predicted == 1 && expected == 1)
					metrics[j].a += 1;
				else if(predicted == 1 && expected == 0)
					metrics[j].b += 1;
				else if(predicted == 0 && expected == 1)
					metrics[j].c += 1;
				else if(predicted == 0 && expected == 0)
					metrics[j].d += 1;
			}
		}
		return metrics;
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
	public static void printResults(Metric[] metrics, Network network, String output) throws IOException {
		PrintWriter pw = new PrintWriter(new FileWriter(output));
		double accTotal,precisionTotal,recallTotal;
		accTotal = precisionTotal = recallTotal = 0;
		int aTotal,bTotal,cTotal,dTotal;
		aTotal = bTotal = cTotal = dTotal = 0;
      //for each output class/category
      //A B C D overallAccuracy Precision Recall F1
		double no = network.outputLayer.length;
		for(int i = 0; i < no; i++){
			metrics[i].calculate();

			pw.print(metrics[i].abcd());

			pw.print(metrics[i].values());
			pw.printf("%n");

			aTotal += metrics[i].a;
			bTotal += metrics[i].b;
			cTotal += metrics[i].c;
			dTotal += metrics[i].d;

			accTotal += metrics[i].overallAccuracy;
			precisionTotal += metrics[i].precision;
			recallTotal += metrics[i].recall;
		}
      //microaveraged metrics overallAccuracy Precision Recall F1 (each decision 0 or 1 weighed equally)
		Metric micro = new Metric(aTotal,bTotal,cTotal,dTotal);
		micro.calculate();
		pw.print(micro.values());
		pw.printf("%n");


      //macroaveraged metrics overallAccuracy Precision Recall F1 (every class/category weighed equally)
		Metric macro = new Metric(accTotal, precisionTotal, recallTotal);
		macro.calculateMacro(no);
		pw.print(macro.values());

		pw.close();
	}
}
