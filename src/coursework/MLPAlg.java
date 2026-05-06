/* Multilayer Perceptron Algorithm
/* Class that runs MLP Algorithm (calls all other classes that are to be used here)
 * !Hardcoded number of inputs, as network is set to work for specific dataset..
 * Brief runtime of this class: - initialisation stage
 * 								- train method - forward pass - calculating the output (activation) of each neuron
 * 											   - backprop - calculating gradient using those neuron's activation
 * 											   - update stage - updating each neuron's weight by their gradient
 * 								- compare method - compares output from the test dataset with real output and calculate accuracy 
 * One important to notice is cross-entropy loss calculation. It tells me how confident the network is about the prediction..
 * To calculate:	Suppose the correct label = 3
 * 					Output probs: 	0: 0.01
 * 									1: 0.02
 * 									2: 0.05
 * 									3: 0.80 ...
 * 					The code than get probability of the correct label, and calculate loss using cross-entropy function:
 * 									-loss = log(0.80) e.g. loss = -log(0.80)
 * 									loss = 0.223*/

package coursework;

class MLPAlg
{
	/* Main method to run the algorithm - initialise all layers, train and compare, returns result
	 * @par trainData - data to train neural network
	 * @par testData - data to test against calculated neural network
	 * @HIDDENLAYERS - number of hidden layers
	 * @NEURONS - array that stores number of neurons inside each hidden layer
	 * @LEARNINGRATE - learning rate for training
	 * @EPOCHS - number of epochs - how many times i will update weights with all of the data
	 * @ret - returns percentage of how many data were accurately classified from the test data*/
	public static double MLPClassify(DataSet trainData, DataSet testData, int HIDDENLAYERS, int[] NEURONS, 
			double LEARNINGRATE, int EPOCHS, int[][] confusionMatrix, int OUTPUTSNUMBER)
	{
		// define layers ( +1 for output layer)
		Layer[] hiddenLayers = new Layer[HIDDENLAYERS + 1];
		
		// define neurons
		for (int layerNumber = 0; layerNumber < hiddenLayers.length - 1; layerNumber++) {
			
			// initialise input size (for weights, either data inputs or number of neurons in the layer before)
			int inputSize;
			if (layerNumber == 0) inputSize = 64; // 64 inputs since first layer takes from the input
			else inputSize = hiddenLayers[layerNumber - 1].getNumNeurons();
			
			// hidden layer initialisation
			hiddenLayers[layerNumber] = new Layer(NEURONS[layerNumber], inputSize);
		}
		
		// output layer initialisation
		hiddenLayers[HIDDENLAYERS] = new OutputLayer(OUTPUTSNUMBER, hiddenLayers[HIDDENLAYERS - 1].getNumNeurons());
		
		// train using trainData
		train(hiddenLayers, trainData, LEARNINGRATE, EPOCHS);
		
		// compare using testData
		double result = compare(testData.arr2d, hiddenLayers, testData.arr1d, confusionMatrix);
		
		return result;
	}
	
	/* Method that is responsible for training neural network by the trainData
	 * @par hiddenLayers - topology of the neural network - holds the whole neural network (include neurons and their weights)
	 * @par trainData - data to train neural network with
	 * @LEARNINGRATE - learning rate - how fast should i be updating the weights with each training
	 * @EPOCHS - how many times shall i run training of all data*/
	private static void train(Layer[] hiddenLayers, DataSet trainData, double LEARNINGRATE, int EPOCHS)
	{
		// repeat for number of epochs
		for (int epoch = 0; epoch < EPOCHS; epoch++) {
			
			// repeat for each data array
			for (int dataNum = 0; dataNum < trainData.arr2d.length; dataNum++) {
				
				// forward passing until calculated gradients
				forwardPass(hiddenLayers, trainData.arr2d[dataNum]);
				// calculate loss of the correct number - calculated only for info ... not used for training
				// double loss = calculateLoss(hiddenLayers, trainData.arr1d[dataNum]);
				// calculating gradients for each neuron's weight
				backprop(hiddenLayers, trainData.arr1d[dataNum]);
				// update weights by gradients in each layer
				updateStage(hiddenLayers, LEARNINGRATE);
			}
			
			// reshuffle train data array
			trainData.shuffleDataSet();
		}
	}
	
	/* Method to compare test data against trained neural network
	 * @par testData - 2d array of test data to be tested against neural network
	 * @par hiddenLayers - hidden layers that store all of the neurons, therefore, here i m passing whole neural network
	 * @par testResult - 2d data's labels
	 * @ret double - percentage of how many data were accurately classified from the test data*/
	private static double compare(int[][] testData, Layer[] hiddenLayers, int[] testResults, int[][] confusionMatrix)
	{
		// store result - correct/incorrect classified data
		int correct = 0;
		int incorrect = 0;
		
		for (int dataNum = 0; dataNum < testData.length; dataNum++)	{
			
			// forward pass to test each data
			forwardPass(hiddenLayers, testData[dataNum]);
			
			// check the highest probability from the output
			double maxProb = -1.0;
			int bestProbability = 0; // storing best probability here
			
			// cast this layer to use its extended methods for output layer
			OutputLayer out = (OutputLayer) hiddenLayers[hiddenLayers.length - 1];
			
			for (int probabilityNumber = 0; probabilityNumber < out.getProbabilities().length; probabilityNumber++) {
			    double tempProb = out.getProbabilities()[probabilityNumber];
			    if (tempProb > maxProb) {
			        maxProb = tempProb;
			        bestProbability = probabilityNumber;
			    }
			}
			int result = bestProbability;
			
			// add values to the confusion matrix
			confusionMatrix[testResults[dataNum]][result]++;
			
			// compare result with the correct label
			if (result == testResults[dataNum]) correct++;
			else incorrect++;
		}
		
		System.out.println("\nCorrect findings: " + correct + "\nIncorrect findings: " + incorrect);
		
		return 100.0 / testResults.length * correct;
	}
	
	/* Method responsible for forward passing of the data, both train and test
	 * @par hiddenLayers - array of all hidden layers
	 * @par inputArr - input data */
	private static void forwardPass(Layer[] hiddenLayers, int[] inputArr)
	{
		// start calculation with the data input
		hiddenLayers[0].activateLayer(inputArr);
		
		// carry on with the data from each hidden layer
		for (int layerNumber = 1; layerNumber < hiddenLayers.length; layerNumber++) {
			hiddenLayers[layerNumber].activateLayer(hiddenLayers[layerNumber - 1].getNeurons());
		}
	}
	
	/* Method to calculate loss using cross-entropy function - not used, only for info
	 * @par hiddenLayers - array of all hidden layers
	 * @par expResult - expected result from the train data*/
	/*private static double calculateLoss(Layer[] hiddenLayers, int expResult)
	{
		// downcast necessary (since i have not used abstract class)
		OutputLayer out = (OutputLayer) hiddenLayers[hiddenLayers.length - 1];
	    return -Math.log(out.getProbabilities()[expResult]);
	}*/
	
	/* Method that is responsible for backpropagation through all hidden layers (from output layer to the first hidden layer)
	 * @par hiddenLayers - array of all hidden layers (including output layer)
	 * @par result - correct result of the train data
	 * */
	private static void backprop(Layer[] hiddenLayers, int result)
	{
		// set output layer deltas
		OutputLayer out = (OutputLayer) hiddenLayers[hiddenLayers.length - 1];
		out.setDeltas(result);
		
		// loop through hidden layers backwards, with focus at previous hidden layer, since last hidden layer
		// weights get updated from the output layer function
		for (int layerIndex = hiddenLayers.length - 2; layerIndex >= 0; layerIndex--) {
			Layer nextLayer = hiddenLayers[layerIndex + 1];
			hiddenLayers[layerIndex].setDeltas(nextLayer);
		}
	}

	/* Method that updates each neuron's weight by its newly calculated gradient
	 * @par hiddenLayers - array of all layer
	 * @par LEARNINGRATE - learning rate set as constant by user*/
	private static void updateStage(Layer[] hiddenLayers, double LEARNINGRATE)
	{
		// loop through each layer that stores neurons, that store their weights and gradients
		for (Layer layer : hiddenLayers) {
			for (Neuron neuron : layer.getNeurons()) {
				neuron.updateWeights(LEARNINGRATE);
			}
		}
	}
}