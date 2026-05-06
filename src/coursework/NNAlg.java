/* K - Nearest Neighbour and Nearest Neighbour algorithm using Euclidean distance.
 * Takes 2 datasets, and classify 2nd against 1st using shortest Euclidean distance, provide NN and KNN categorisation.
 * Class stores "K" best results and classify the result by the weight (inversed squared distance weighting), or
 * by majority vote (depends on the settings in main class)! Weighted result was added later, for that reason, there
 * are 2 sort array methods and 2 find most common methods!
 * Weight is calculated 1 / (distance)^2 -> and than add the same label's weights together, highest wins
 * Example for 3 pixel image: A = [1, 2, 3]
 * 							  B = [4, 5, 6]
 * 							  (1 - 4)^2 = 9
 * 							  (2 - 5)^2 = 9
 * 							  (3 - 6)^2 = 9
 * 							  Sum = 27
 * 							  Distance = sqrt(27)
 * 	
 * Example of weighted:
 * Label = 3, Distance = 1.0 → Weight = 1 / (1.0)^2 = 1.0	// since i m comparing my picture with other pictures,
 * Label = 3, Distance = 2.0 → Weight = 1 / (2.0)^2 = 0.25	// other pictures labels repeat, but they do not necesarily
 * Label = 7, Distance = 1.5 → Weight = 1 / (1.5)^2 ≈ 0.44	// have the same distance.. if the sum of weights higher on
 *															// even just 1 label, that 1 label wins
 * Sum weights per label:
 * Label 3 → 1.0 + 0.25 = 1.25
 * Label 7 → 0.44
 *
 * Result = Label with highest weight sum = 3 */

package coursework;

class NNAlg
{
	/* Method to take each line of pixels from the test set, and compare to all in train data set, calculating
	 * Euclidean distance and storing the shortest distance with Label. Method than prints Correct and Incorrect
	 * findings and returns result in percentage
	 * @par trainData -> data to be trained against
	 * @par testData -> data to be tested
	 * @par confusionMatrix - 2d array for the confusion matrix
	 * @ret double -> result to be returned from the calculation in percentage */
	public static double NNClassify(DataSet trainData, DataSet testData, int[][] confusionMatrix)
	{
		// final results initialisation
		int correct = 0;
		int incorrect = 0;
		
		// loop through each array in test data
		for (int testArrayNumber = 0; testArrayNumber < testData.arr2d.length; testArrayNumber++) {
			
			int trainLabel = Integer.MAX_VALUE;
			double distance = Double.MAX_VALUE;
			
			// loop through each array in train data to compare test data
			for (int trainArrayNumber = 0; trainArrayNumber < trainData.arr2d.length; trainArrayNumber++) {
				
				// calculate distance and compare
				double tempDistance = eucDistCalc(testData.arr2d[testArrayNumber], trainData.arr2d[trainArrayNumber]);
				
				if (tempDistance < distance) {
					distance = tempDistance;
					trainLabel = trainData.arr1d[trainArrayNumber];
				}
			}
			
			// add values to the confusion matrix
			confusionMatrix[testData.arr1d[testArrayNumber]][trainLabel]++;
			
			// compare label results and increment correct/incorrect
			if (trainLabel == testData.arr1d[testArrayNumber]) correct++;
			else incorrect++;
		}
		
		// print the results
		System.out.println("Correct findings: " + correct);
		System.out.println("Incorrect findings: " + incorrect);
		
		return 100.0 / testData.arr1d.length * correct;
	}
	
	/* Method to calculate Euclidean distance between two arrays
	 * @par arr1, arr2 -> arrays to calculate distances between
	 * @ret double -> return result (Euclidean distance)*/
	private static double eucDistCalc(int[] arr1, int[] arr2)
	{
		// result to be returned
		double result = 0;
		
		// do Euclidean distance calculation for each element in both arrays
		for (int arr_index = 0; arr_index < arr1.length; arr_index++) {
			result += Math.pow(arr1[arr_index] - arr2[arr_index], 2);
		}
		
		return Math.sqrt(result);
	}
	
	/* Method to take each line of pixels from the test set, and compare to all in train data set, calculating
	 * Euclidean distance and storing the shortest distance in an array of K shortest distances. 
	 * Method than compares the most common element in an array, prints correct and incorrect findings and returns result in percentage
	 * @par trainData -> data to be trained against
	 * @par testData -> data to be tested
	 * @par NEIGHBOURSNUMBER -> K (neighbours) number to store K shortest distances in an array
	 * @par confusionMatrix - 2d array for the confusion matrix
	 * @ret double -> result to be returned from the calculation in percentage */
	public static double KNNClassify(DataSet trainData, DataSet testData, int NEIGHBOURSNUMBER, int[][] confusionMatrix, boolean weighted)
	{
		System.out.println("K (neighbors) = " + NEIGHBOURSNUMBER + "\n");
		
		int correct = 0; // final results initialisations
		int incorrect = 0;

		if (NEIGHBOURSNUMBER > trainData.arr1d.length) return 0.0; // if neighboursnumber too large

		double knnArr[] = new double[NEIGHBOURSNUMBER]; // KNN arrays, 1 to store distance, 1 to store label
		int knnArrLBL[] = new int[NEIGHBOURSNUMBER];
		
		// loop through each array  in test data
		for (int testArrayNumber = 0; testArrayNumber < testData.arr2d.length; testArrayNumber++) {
			
			// initialise knn arrays with the first K distances and labels
			for (int trainArrNoInit = 0; trainArrNoInit < NEIGHBOURSNUMBER; trainArrNoInit++) {
				knnArr[trainArrNoInit] = eucDistCalc(testData.arr2d[testArrayNumber], trainData.arr2d[trainArrNoInit]);
				knnArrLBL[trainArrNoInit] = trainData.arr1d[trainArrNoInit];
			}
			
			// loop through each array in train data to compare test data
			for (int trainArrayNumber = NEIGHBOURSNUMBER; trainArrayNumber < trainData.arr2d.length; trainArrayNumber++) {
				// calculate distance and compare with other distances stored in comArr[]
				double tempDistance = eucDistCalc(testData.arr2d[testArrayNumber], trainData.arr2d[trainArrayNumber]);
				compareDistance(knnArr, knnArrLBL, tempDistance, trainData.arr1d[trainArrayNumber]);
			}
			
			// find the result
			int trainLabel = weighted ? findMostCommonWeighted(knnArr, knnArrLBL) : findMostCommon(knnArrLBL);
			
			// add values to the confusion matrix
			confusionMatrix[testData.arr1d[testArrayNumber]][trainLabel]++;
			
			if (trainLabel == testData.arr1d[testArrayNumber]) correct++;
			else incorrect++;
		}
		
		// print the results
		System.out.println("Correct findings: " + correct);
		System.out.println("Incorrect findings: " + incorrect);
		
		return 100.0 / testData.arr1d.length * correct;
	}
	
	/* Method to compare array of k elements stored for the result with the element I m comparing the test set
	 * @par knnArr -> array of resulted elements so far (distances)
	 * @par knnArrLBL -> array of labels representing knnArr
	 * @par tempDistance -> distance of the testing data
	 * @par testLBL -> label of the testing distance*/
	private static void compareDistance(double[] knnArr, int[] knnArrLBL, double tempDistance, int testLBL)
	{
		// set the first index from knnArr to 0 to start with
		int tempLowest = 0;
		
		// find the farthest distance from the knnArr
		for (int arrIndex = 0; arrIndex < knnArr.length; arrIndex++) {
			if (knnArr[arrIndex] > knnArr[tempLowest]) tempLowest = arrIndex;
		}
		
		// compare the farthest distance with tempDistance, if tempDistance closer, store tempDistance on the position of the farthest distance
		if (knnArr[tempLowest] > tempDistance) {
			knnArr[tempLowest] = tempDistance;
			knnArrLBL[tempLowest] = testLBL;
		}
	}
	
	/* Method to sort resulting array and find the most common element (non weighted result)
	 * @par knnArrLBL - array of labels I'm trying to find the most common element from
	 */
	private static int findMostCommon(int[] knnArrLBL) 
	{
	    sortArr(knnArrLBL); // sort the array

	    // current best (most common)
	    int mostCommonLabel = knnArrLBL[0];
	    int timesRepeated = 1;

	    // current run I am counting
	    int tempMostCommon = knnArrLBL[0];
	    int tempTimesRepeated = 1;

	    for (int index = 1; index < knnArrLBL.length; index++) { // loop through all elements, counting runs of equal values
	        if (knnArrLBL[index] == tempMostCommon) {
	            tempTimesRepeated++; // same as current run, increase count
	        } 
	        else { // value changed: check if the run we just finished is the new best
	            if (tempTimesRepeated > timesRepeated) {
	                timesRepeated = tempTimesRepeated;
	                mostCommonLabel = tempMostCommon;
	            }
	            // start counting new value
	            tempMostCommon = knnArrLBL[index];
	            tempTimesRepeated = 1;
	        }
	    }

	    // after the loop, we need to compare the last run as well
	    if (tempTimesRepeated > timesRepeated) {
	        mostCommonLabel = tempMostCommon;
	    }
	    return mostCommonLabel;
	}
	
	/* Simple sorting algorithm to sort an array (bubble sort)
	 * @par arr - array of integers to be sorted*/
	private static void sortArr(int[] arr)
	{
		int arrSize = arr.length;
		boolean swapped;
		
		// loop from the bottom
		for (int bottomIndex = 0; bottomIndex < arrSize - 1; bottomIndex++) {
			
			swapped = false;
			
			// loop from the top
			for (int topIndex = 0; topIndex < arrSize - bottomIndex - 1; topIndex++) {
				// swap if two elements are in wrong order
				if (arr[topIndex] > arr[topIndex + 1]) {
					int temporary = arr[topIndex];
					arr[topIndex] = arr[topIndex + 1];
					arr[topIndex + 1] = temporary;
					swapped = true;
				}
			}
			
			// if I did not swap anything, means all is correct
			if (!swapped) break;
		}
	}
	
	/* Method to sort resulting arrays and find the most common label using weighted vote (weighted result)
	 * @par knnDistances - distances of K nearest neighbours
	 * @par knnLabels - labels of K nearest neighbours
	 */
	private static int findMostCommonWeighted(double[] knnDistances, int[] knnLabels)
	{
	    sortByLabel(knnLabels, knnDistances); // sort labels, keep distances aligned
	    
	    int currentLabel = knnLabels[0]; // current run
	    double currentWeightSum = 0.0;
	    int bestLabel = currentLabel; // best (max weight sum)
	    double bestWeightSum = -1.0;  // something smaller than any real weight sum

	    for (int knnIndex = 0; knnIndex < knnLabels.length; knnIndex++) { // loop through each k elements
	    	
	        int label = knnLabels[knnIndex];
	        double distance = knnDistances[knnIndex];

	        // if we have a training point at exactly the same position as the test point,
	        // just return its label directly (avoid division by zero and it makes sense)
	        if (distance == 0.0) return label;

	        double weight = 1.0 / (distance * distance); // Example weighting: inverse squared distance

	        if (label == currentLabel) currentWeightSum += weight; // same label run, add weight
	        else {
	            // label changed, check if finished run is the best so far
	            if (currentWeightSum > bestWeightSum) {
	                bestWeightSum = currentWeightSum;
	                bestLabel = currentLabel;
	            }

	            // start new run
	            currentLabel = label;
	            currentWeightSum = weight;
	        }
	    }
	    
	    // after the loop, compare the last run as well
	    if (currentWeightSum > bestWeightSum) bestLabel = currentLabel;

	    return bestLabel;
	}
	
	/* Bubble sort sorting algorithm to sort labels and keep distances aligned with them for Weighted algorithm
	 * @par labels - array of labels to be sorted
	 * @par distances - array of distances aligned with labels
	 */
	private static void sortByLabel(int[] labels, double[] distances)
	{
	    int arrSize = labels.length;
	    boolean swapped;

	    // loop from the bottom
	    for (int bottomIndex = 0; bottomIndex < arrSize - 1; bottomIndex++) {
	    	
	        swapped = false;
	        
	        // loop from the top
	        for (int topIndex = 0; topIndex < arrSize - bottomIndex - 1; topIndex++) {
	        	
	            // swap if two elements are in wrong order
	            if (labels[topIndex] > labels[topIndex + 1]) {
	                // swap labels
	                int tempLabel = labels[topIndex];
	                labels[topIndex] = labels[topIndex + 1];
	                labels[topIndex + 1] = tempLabel;

	                // swap distances to stay in sync with labels
	                double tempDist = distances[topIndex];
	                distances[topIndex] = distances[topIndex + 1];
	                distances[topIndex + 1] = tempDist;

	                swapped = true;
	            }
	        }

	        // if I didn't swap anything, means all is correct
	        if (!swapped) break;
	    }
	}
}