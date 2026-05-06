package coursework;

/* Support Vector Machine (SVM) Algorithm 
 * This class implements a multi–class Support Vector Machine using the One-vs-Rest (OvR) approach. This means, i have 10 classifiers
 * becouse I have 10 digits.. for example, 1st classifier = 0 or other, 2nd = 1 or other...
 *
 * The algorithm uses:
 *      - Stochastic Gradient Descent (SGD)
 *      - Hinge-loss updates:  if y*(w·x + b) < 1, update weights and bias, othervise, result correct enough
 *      - Polynomial feature expansion (degree 2) to allow non-linear decision boundaries
 *        without using the dual formulation or kernel trick.
 *
 * Brief runtime:
 *      initialiseModel → Creates weight matrix and bias vector sized according to
 *                           the expanded polynomial feature dimension.
 *                           64 original features -> 64 squared features -> 64*63/2 = 2016 interaction terms
 *                           so I have 10 × (2144 weights + 1 bias) = 21,450 parameters
 *                        4 pixels example: [16, 0, 8, 16] -> normalised to numbers between 0 - 1 -> deviding by 16 (0 - 16 pixel values)
 *                        					[1.0, 0.0, 0.5, 1.0] -> now, all values are between 0 and 1
 *                        Polynomial expansion: squared features = [1.0^2, 0.0^2, 0.5^2, 1.0^2] = [1.0, 0.0, 0.25, 1.0]
 *                        Multiply all pairs (squared with non squared): x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4
 *                        Final vector contains original values, squared values and multiplied results:
 *                        [1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.25, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5] -> 14 total values for 4 pixels!
 *                        also, i need weights for each value, i initialise each to 0, and initialise each bias for output to 0 too
 *                        
 *      trainModel → Performs SGD training for a given number of epochs:
							Supose i m training digit 3 vs not digit 3 (3 vs others).. and suppose this digit is 3.
							lets simplify calculations to 4 values (instead 16 created):
							values x: [1.0, 0.0, 0.5, 1.0]
							weights w: [0.2, -0.1, 0.3, 0.5]
							bias b: -0.2
							Calculation of score: (0.2 * 1.0) + (-0.1 * 0.0) + (0.3 * 0.5) + (0.5 * 1.0) - 0.2 = 0.65
							
							Margin calculation: margin = y * score = (+1) * 0.65 = 0.65
							margin < 1, therefore, update is required
							
							update weight: w = w + learningRate * y * x -> suppose learningRate = 0.1
							w1 = 0.2 + 0.1 * 1 * 1.0 = 0.3
							w2 = -0.1 + 0.1 * 1 * 0.0 = -0.1
							...
							bias update: b = -0.2 + 0.1 * 1 = -0.1
							
							So basicly, during training, i have pushed line towards the correct direction!

 *      testModel → Evaluates the trained model on test data using:
 *                          - the same preprocessing
 *                          - One vs Rest scoring (pick class with highest w·x + b)*/

class SVMAlg
{
	/* Method to run whole SVM Algorithm
	 * @par trainingSet -> set to be trained on
	 * @par testingSet -> testing data to check accuracy
	 * @par numberOfClasses -> number of possible outputs
	 * @par learningRate -> learning rate set as constant by the user
	 * @par numberOfEpochs -> number of epochs to train the model for
	 * @par confusionMatrix -> reference to the confusionMatrix to be printed*/
    public static double SVMClassify(DataSet trainingSet, DataSet testingSet, int numberOfClasses,
            double learningRate, int numberOfEpochs, int[][] confusionMatrix)
    {
        int numberOfInputFeatures = trainingSet.arr2d[0].length; // total number of inputs

        // Determine polynomial-feature size
        double[] firstNormalisedVector = normaliseInput(trainingSet.arr2d[0]);
        int polynomialFeatureCount = poly2(firstNormalisedVector).length;

        // Initialise weight and bias vectors
        double[][] weightMatrix = new double[numberOfClasses][polynomialFeatureCount];
        double[] biasVector = new double[numberOfClasses];

        // Train model
        trainModel(trainingSet, numberOfClasses, learningRate, numberOfEpochs, weightMatrix, biasVector,
                numberOfInputFeatures, polynomialFeatureCount);

        // Test model
        return testModel(testingSet,numberOfClasses,weightMatrix,biasVector,numberOfInputFeatures,
        		polynomialFeatureCount,confusionMatrix);
    }

    /* Method to train SVM model from training data
     * @par trainingSet -> dataset used for training
     * @par numberOfClasses -> number of output classes
     * @par learningRate -> learning rate set by the user
     * @par numberOfEpochs -> how many full passes over training data
     * @par weightMatrix -> weights corresponding to each class and feature (updated here)
     * @par biasVector -> bias term for each class (updated here)
     * @par numberOfInputFeatures -> number of original features (before expansion)
     * @par polynomialFeatureCount -> number of features after polynomial expansion
     * */
    private static void trainModel(DataSet trainingSet, int numberOfClasses, double learningRate, int numberOfEpochs,
            double[][] weightMatrix, double[] biasVector, int numberOfInputFeatures, int polynomialFeatureCount)
    {
    	// loop over epochs
        for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
        	
            trainingSet.shuffleDataSet(); // shuffle training set every epoch for SGD
            
            // loop through each training sample
            for (int trainingSampleIndex = 0; trainingSampleIndex < trainingSet.rows; trainingSampleIndex++) {
            	
            	// raw input pixels and true label
                int[] rawPixelVector = trainingSet.arr2d[trainingSampleIndex];
                int trueLabel = trainingSet.arr1d[trainingSampleIndex];
                
                double[] normalisedVector = normaliseInput(rawPixelVector); // normalise input to range 0 - 1
                double[] polynomialFeatures = poly2(normalisedVector); // expand normalised input using polynomial features
 
                // perform one-vs-rest hinge-loss update for all classes
                updateOneVsRestClassifiers(polynomialFeatures, trueLabel, numberOfClasses, learningRate, weightMatrix,
                        biasVector, polynomialFeatureCount);
            }
        }
    }

    /* Method to update all one-vs-rest classifiers for a single training example
     * @par featureVector -> polynomially expanded feature vector for current sample
     * @par trueLabel -> true label of the current sample
     * @par numberOfClasses -> total number of classes (0 to numberOfClasses - 1)
     * @par learningRate -> learning rate set by the user
     * @par weightMatrix -> weight vectors for all classes
     * @par biasVector -> bias terms for all classes
     * @par polynomialFeatureCount -> number of features after polynomial expansion
     * */
    private static void updateOneVsRestClassifiers(double[] featureVector, int trueLabel, int numberOfClasses,
            double learningRate, double[][] weightMatrix, double[] biasVector, int polynomialFeatureCount)
    {
    	// loop through every class and treat it as binary classifier (one-vs-rest)
        for (int classIndex = 0; classIndex < numberOfClasses; classIndex++) {
        	
        	// y = +1 if this is the correct class, otherwise y = -1
            int targetOutput = (trueLabel == classIndex) ? 1 : -1;

            // compute decision score for this class: w_k · x + b_k
            double decisionScore = computeDecisionScore(weightMatrix[classIndex], biasVector[classIndex],
                    featureVector, polynomialFeatureCount);

            double margin = targetOutput * decisionScore; // margin = y * score, hinge loss = max(0, 1 - margin)

            // if margin is less than 1, apply hinge-loss gradient update
            if (margin < 1.0) {
                applyHingeLossUpdate(weightMatrix[classIndex], biasVector, classIndex, featureVector, targetOutput,
                        learningRate, polynomialFeatureCount);
            }
        }
    }

    /* Method to compute decision score for a single class
     * @par weightVector -> weight vector of that class
     * @par bias -> bias of that class
     * @par featureVector -> input feature vector (polynomially expanded)
     * @par featureCount -> number of features in the vector
     * @ret double -> linear score w · x + b
     * */
    private static double computeDecisionScore(double[] weightVector, double bias, double[] featureVector, int featureCount)
    {
        double score = 0.0;

    	// dot product between weight vector and feature vector
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            score += weightVector[featureIndex] * featureVector[featureIndex];
        }

        return score + bias;
    }

    /* Method to apply hinge-loss update to one class's weights and bias
     * @par weightVector -> weight vector of the updated class
     * @par biasVector -> array of all biases (we update the one for this class)
     * @par classIndex -> index of the class being updated
     * @par featureVector -> input feature vector (polynomially expanded)
     * @par targetOutput -> +1 if positive class, -1 if negative
     * @par learningRate -> learning rate set by the user
     * @par featureCount -> number of features in featureVector
     * */
    private static void applyHingeLossUpdate(double[] weightVector, double[] biasVector, int classIndex,
            double[] featureVector, int targetOutput, double learningRate, int featureCount)
    {
    	// w = w + learningRate * y * x
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            weightVector[featureIndex] += learningRate * targetOutput * featureVector[featureIndex];
        }

        // b = b + learningRate * y
        biasVector[classIndex] += learningRate * targetOutput;
    }

    /* Method to test trained SVM model on testing data
     * @par testingSet -> dataset to be tested
     * @par numberOfClasses -> number of possible outputs
     * @par weightMatrix -> trained weight vectors for each class
     * @par biasVector -> trained biases for each class
     * @par numberOfInputFeatures -> number of original features (before expansion) - used for normalisation
     * @par polynomialFeatureCount -> number of features after polynomial expansion
     * @par confusionMatrix -> matrix storing true vs predicted class counts
     * @ret double -> percentage of correctly classified testing data
     * */
    private static double testModel(DataSet testingSet, int numberOfClasses, double[][] weightMatrix, double[] biasVector,
            int numberOfInputFeatures, int polynomialFeatureCount, int[][] confusionMatrix)
    {
        int correct = 0;
        int incorrect = 0;

        // loop through all testing samples
        for (int testSampleIndex = 0; testSampleIndex < testingSet.rows; testSampleIndex++) {
        	// read raw pixels and true label
            int[] rawPixelVector = testingSet.arr2d[testSampleIndex];
            int trueLabel = testingSet.arr1d[testSampleIndex];

            // normalise input and expand to polynomial features
            double[] normalisedVector = normaliseInput(rawPixelVector);
            double[] polynomialFeatures = poly2(normalisedVector);

            // predict class for this sample
            int predictedClass = predictClass(polynomialFeatures, numberOfClasses, weightMatrix,
                    biasVector, polynomialFeatureCount);

            // update confusion matrix
            confusionMatrix[trueLabel][predictedClass]++;

            // check if prediction is correct
            if (predictedClass == trueLabel) correct++;
            else incorrect++;
        }
        
        System.out.println("Correct findings: " + correct);
        System.out.println("Incorrect findings: " + incorrect);
        System.out.println("Percentage of correct anaswers: " + 100.0 * correct / testingSet.rows);
        
        
        // return accuracy in percentage
        return 100.0 * correct / testingSet.rows;
    }

    /* Method to predict class label for a single sample
     * @par polynomialFeatures -> polynomially expanded feature vector for the sample
     * @par numberOfClasses -> total number of classes
     * @par weightMatrix -> weight vectors for all classes
     * @par biasVector -> bias values for all classes
     * @par polynomialFeatureCount -> length of polynomialFeatures array
     * @ret int -> predicted class (0 to numberOfClasses - 1)
     * */
    private static int predictClass(double[] polynomialFeatures, int numberOfClasses, double[][] weightMatrix,
            double[] biasVector, int polynomialFeatureCount)
    {
        double highestScore = Double.NEGATIVE_INFINITY;
        int bestClass = 0;

        // compute score for each class and choose the one with highest score
        for (int classIndex = 0; classIndex < numberOfClasses; classIndex++) {
            double score = computeDecisionScore(weightMatrix[classIndex], biasVector[classIndex],
                    polynomialFeatures, polynomialFeatureCount);

            if (score > highestScore) {
                highestScore = score;
                bestClass = classIndex;
            }
        }

        return bestClass;
    }

    /* Method to normalise input vector to range 0 - 1
     * @par rawPixelVector -> array of raw pixel values (0 - 16)
     * @ret double[] -> array of normalised values (0 - 1)
     * */
    private static double[] normaliseInput(int[] rawPixelVector)
    {
        double[] normalised = new double[rawPixelVector.length];

        for (int pixelIndex = 0; pixelIndex < rawPixelVector.length; pixelIndex++) {
            normalised[pixelIndex] = rawPixelVector[pixelIndex] / 16.0;
        }

        return normalised;
    }

    /* Method that calculates degree-2 polynomial expansion and returns expanded feature vector
     * Expansion includes:
     *  - original features
     *  - squared terms of each feature
     *  - pairwise interaction terms between different features
     * @par inputVector -> normalised input vector
     * @ret double[] -> expanded feature vector
     * */
    public static double[] poly2(double[] inputVector)
    {
        int inputSize = inputVector.length;
        
        // total size = original + squared + interaction terms
        int expandedSize = inputSize + inputSize + (inputSize * (inputSize - 1)) / 2;

        double[] expanded = new double[expandedSize];

        int writeIndex = 0;

        // Original features
        for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
            expanded[writeIndex++] = inputVector[featureIndex];
        }

        // Squared terms
        for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
            expanded[writeIndex++] = inputVector[featureIndex] * inputVector[featureIndex];
        }

        // Interaction terms x_i * x_j for i < j
        for (int firstFeature = 0; firstFeature < inputSize; firstFeature++) {
            for (int secondFeature = firstFeature + 1; secondFeature < inputSize; secondFeature++) {
                expanded[writeIndex++] = inputVector[firstFeature] * inputVector[secondFeature];
            }
        }

        return expanded;
    }
}