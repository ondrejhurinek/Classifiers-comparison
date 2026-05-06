/* OutputLayer class that extends layer class 
 * The main difference between the Layer and OutputLayer are activation function. While Layer uses ReLU, OutputLayer uses
 * Softmax to calculate probabilities for each output. The difference is also how the Deltas are calculated!!
 * In practice:	Preactivation function works the same way as in Layer class, it returns logits for each neuron (output)
 * 				Example:	Digit 0 score = 2.0
 * 							Digit 1 score = 1.0
 * 							Digit 2 score = 0.1
 * 				maxLogit is stored, and i subtract from each logit: original logits: [2.0, 1.0, 0.1]
 * 																	subtracted logits: [0. -1.0, -1.9]
 * 				than, apply e^x:	[e^0.0, e^-1.0, e^-1.9] =
 * 									[1.0, 0.3679, 0.1496]
 * 				than, it adds them:	sumExp = 1.0 + 0.3679 + 0.1496 = 1.5175
 * 				than, normalisation:	1.0 / 1.5175 = 0.659	-> final probability = 65.9%
 * 										0.3679 / 1.5175 = 0.242 -> final probability = 24.2%
 * 										0.1496 / 1.5175 = 0.099 -> final probability = 9.9%
 * 				
 * 				Calculating Delta:	The real correct is result = 0 and probabilities: [0.659, 0.242, 0.099]
 * 																	the true label:	  [1,	  0, 	 0]
 * 									So now deltas are set: for digit 0 = 0.659 - 1 = -0.341
 * 																	 1 = 0.242
 * 																	 2 = 0.099
 * 									So this becomes array of deltas: [-0.341, 0.242, 0.099]
 * 				Gradient calculation:	gradient = delta * input from previous layer
 * 				weight = weight - learningRate * gradient
 * 				bias = bias - learningRate * delta
 * */

package coursework;

class OutputLayer extends Layer
{
	// datafield
	private double[] probabilities; // yHat for each output
	
	// constructor
	/* @par neuronsNumber - number of neurons inside output layer - number of possible outputs
	 * @par inputSize - number of neurons inside the previous (hidden) layer*/
	public OutputLayer(int neuronsNumber, int inputSize)
	{
		super(neuronsNumber, inputSize);
		probabilities = new double[neuronsNumber];
	}
	
	/* Getter for probabilities
	 * ret double[] array of probabilities calculated for each output (yHat for each output)*/
	public double[] getProbabilities()
	{
		return probabilities;
	}
	
	/* Overridden activation method (for the array of double's - each neuron's output)
	 * I have to make sure that overflow does not occur upon the activation, since ReLU might grow into big numbers,
	 * and than, if I calculate output, it could be big number */
	@Override
	public void activateLayer(Neuron[] inputNeurons) 
	{

	    // Compute raw logits z_k for each output neuron
	    for (int neuronNum = 0; neuronNum < neurons.length; neuronNum++) {
	        neurons[neuronNum].preactivateNeuron(inputNeurons);
	    }

	    // Find max logit for numerical stability -- to assure there is no over or undeflow in the final softmax
	    double maxLogit = Double.NEGATIVE_INFINITY;
	    for (int neuronNum = 0; neuronNum < neurons.length; neuronNum++) {
	        double prevOutput = neurons[neuronNum].getActivation();
	        if (prevOutput > maxLogit) maxLogit = prevOutput;
	    }

	    // Compute exp(z_k - maxLogit) and sum them
	    double sumExp = 0.0;
	    for (int neuronNum = 0; neuronNum < neurons.length; neuronNum++) {
	        double neuronOutput = neurons[neuronNum].getActivation();
	        double expShifted = Math.exp(neuronOutput - maxLogit);
	        probabilities[neuronNum] = expShifted;   // temporarily store exp(z_k - maxLogit)
	        sumExp += expShifted;
	    }

	    // Normalise to get probabilities
	    for (int probIndex = 0; probIndex < probabilities.length; probIndex++) {
	        probabilities[probIndex] /= sumExp;      // now probabilities[i] is y_hat_k
	    }
	}

	/* Set deltas (blame) - softmax + cross-entropy
	 * @par result - true result index (correct label) */
	public void setDeltas(int result)
	{
		// because this is output layer, for each neuron, set its delta to its probability (or probability - 1 for correct output probability)
		for (int neuronIndex = 0; neuronIndex < neurons.length; neuronIndex++) {
			
			if (neuronIndex == result) neurons[neuronIndex].setDelta(probabilities[neuronIndex] - 1);
			else neurons[neuronIndex].setDelta(probabilities[neuronIndex]);
			
			// than set gradient
			neurons[neuronIndex].setGradient();
		}
	}
}