/* Neuron class
 * Class that holds all the methods and properties of each neuron
 * Each neuron stores their weights, that are initialised using He initialisation: weight = randomGausian * squareRoot(2.0 / inputSize)
 * where inputSize = 64 in my case... each bias is set to 0*/

package coursework;

import java.util.Random;

class Neuron 
{
	// datafield
	private double[] weights;		// weights for each neuron (including bias as last weight)
	private double activation;		// activation value
	private double delta;			// "blames" / deltas for all outputs
	private double[] gradients;		// gradients to update each weight (index same as the index of the weight they refer to)
	private double[] lastInputs;	// values of inputs from previous layer (previous layer activation)
	
	// constructor
	/*@par inputSize - size of the layer (input) before*/
	public Neuron(int inputSize)
	{
		// init weights
		weights = new double[inputSize + 1];
		gradients = new double[inputSize + 1];
		lastInputs = new double[inputSize];
		
		Random rand = new Random();
		
		for (int weightNum = 0; weightNum < inputSize; weightNum++) {
			weights[weightNum] = rand.nextGaussian() * Math.sqrt(2.0 / inputSize);
		}
		
		weights[inputSize] = 0.0; // bias
	}
	
	// Member methods
	/* Method to calculate gradients for each weight */
	public void setGradient()
	{
		for (int weightIndex = 0; weightIndex < weights.length - 1; weightIndex++) {
			gradients[weightIndex] = delta * lastInputs[weightIndex];
		}
		gradients[weights.length - 1] = delta; // bias = delta
	}
	
	/* Method to update weights by their gradients
	 * @par LEARNINGRATE - learning rate defined by the user as constant in main method*/
	public void updateWeights(double LEARNINGRATE)
	{
		for (int weightIndex = 0; weightIndex < weights.length; weightIndex++) {
			weights[weightIndex] = weights[weightIndex] - LEARNINGRATE * gradients[weightIndex];
		}
	}
	
	/* Method to return delta (blame) of that neuron*/
	public double getDelta()
	{
		return delta;
	}
	
	/* Method to return weight on of the index
	 * @par weightIndex - index of the weight to be returned*/
	public double getWeight(int weightIndex)
	{
		return weights[weightIndex];
	}
	
	/* Method to set Delta value (blame) of particular neuron
	 * @par newDelta - value of the delta to be set to*/
	public void setDelta(double newDelta)
	{
		delta = newDelta;
	}
	
	/* Method to return activation (output value) from the neuron*/
	public double getActivation()
	{
		return activation;
	}
	
	/* Method to preactivate neuron for further calculations
	 * @par inputVector - input values from the previous layer*/
	public void preactivateNeuron(int[] inputVector)
	{
		activation = 0;
		for (int weightIndex = 0; weightIndex < inputVector.length; weightIndex++) {
			// divide by 16 for data input normalisation (to assure they are in a range between 0 - 1 for accuracy)
			// 16 because that is the value the pixels can be
			lastInputs[weightIndex] = inputVector[weightIndex] / 16.0;
			activation += lastInputs[weightIndex] * weights[weightIndex];
		}
		activation += weights[weights.length - 1];
	}
	
	/* Overloaded member method 
	 * @par neurons - array of neurons from the previous layer to get their activation (output)*/
	public void preactivateNeuron(Neuron[] neurons)
	{
		activation = 0;
		for (int weightIndex = 0; weightIndex < neurons.length; weightIndex++) {
			lastInputs[weightIndex] = neurons[weightIndex].getActivation();
			activation += weights[weightIndex]*lastInputs[weightIndex];
		}
		activation += weights[weights.length - 1];
	}
	
	/* ReLU activation method*/
	public void reluActivation()
	{
		if (activation < 0) activation = 0;
	}
}