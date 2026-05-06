/* Class Layer that represent all of the layers (including output layer)
 * This class stores all neurons inside the layer and define their methods
 * So basically, this class serves as the whole network, since network is an array of layers
 * 
 * Important to note: 
 * 
 * Preactivation methods are here to calculate weight*input1 + weight*input2 ... + bias
 * There are 2 types, since in the first hidden layer, inputs are actual pixels, all other layers have input the
 * actual output of the previous layers... Therefore, preactivation works independently from the activation function (ReLU)
 * ReLU than  updates value to 0 if value is negative
 * 
 * setDeltas method than calculates blame of each neuron -> ReLU derivative means that if ReLU = 0, derivative = 0, else, derivative = 1
 * 															If ReLU derivative = 0, Delta blame = 0, else:
 * 															Delta = ReLU derivative * ((weight1 * delta1) + (weight2 * delta2)...)
 * 
 * setGradients calculating: inputIntoNeuron * delta, and gradient of bias becomes delta
 * 
 * Weights are than updates using: weight = weight - learningRate * gradient -> for each weight
 * 								   bias = bias - learningRate * delta
 * */

package coursework;

class Layer
{
	// datafield
	protected Neuron[] neurons; // number of neurons in the layer (hidden or output layer)
	
	// constructor
	/* @par neuronsNumber - number of neurons to be initialised
	 * @par inputSize - size of the inputs (weights - 1) or 64 for the first layer */
	public Layer(int neuronsNumber, int inputSize)
	{
		neurons = new Neuron[neuronsNumber];
		
		// initialise weights for each neuron here, inside Neuron constructor
		for (int neuronNum = 0; neuronNum < neuronsNumber; neuronNum++) {
			neurons[neuronNum] = new Neuron(inputSize);
		}
	}
	
	/* accessor for the number of neurons
	 * @ret int - returns number of neurons*/
	public int getNumNeurons()
	{
		return neurons.length;
	}
	
	/* accessor for the neurons array 
	 * ret Neuron[] - returns array of Neuron objects*/
	public Neuron[] getNeurons()
	{
		return neurons;
	}
	
	/* Preactivation method for the very first hidden layer, because it takes int[] as input
	 * @par int[] - array of inputs (64 in this case)*/
	public void activateLayer(int[] inputArr)
	{
		for (int neuronNum = 0; neuronNum < neurons.length; neuronNum++) {
			neurons[neuronNum].preactivateNeuron(inputArr);
			neurons[neuronNum].reluActivation();
		}
	}
	
	/* Overloaded preactivation method (for the array of neurons - each neuron's output)
	 * @par Neuron[] - array of all neurons inside the layer */
	public void activateLayer(Neuron[] neurons)
	{
		for (int neuronNum = 0; neuronNum < this.neurons.length; neuronNum++) {
			this.neurons[neuronNum].preactivateNeuron(neurons);
			this.neurons[neuronNum].reluActivation();
		}
	}
	
	/* Method to set delta for each neuron
	 * @par nextLayer - necessary to have weights from the next layer for calculations*/
	public void setDeltas(Layer nextLayer)
	{
	    Neuron[] nextNeurons = nextLayer.getNeurons(); // set up array of the next layer neurons

	    // loop through all neurons in the current layer
	    for (int neuronIndex = 0; neuronIndex < neurons.length; neuronIndex++) {
	        double activation = neurons[neuronIndex].getActivation(); // get activation for this 1 neuron

	        // ReLU derivative
	        if (activation == 0) { // if activation == 0 (neuron not active, set delta to 0)
	            neurons[neuronIndex].setDelta(0.0);
	        } else {
	            double sum = 0.0;
	            // sum over all neurons in next layer
	            for (Neuron nextNeuron : nextNeurons) {
	                sum += nextNeuron.getWeight(neuronIndex) * nextNeuron.getDelta();
	            }
	            neurons[neuronIndex].setDelta(sum);   // ReLU was 1 since activation > 0
	        }

	        neurons[neuronIndex].setGradient(); // uses lastInputs & delta
	    }
	}
}