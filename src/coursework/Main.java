/* This code implements and compares several machine-learning classification algorithms
 * on the optical handwritten-digit dataset (8×8 pixel inputs, 10 output classes).
 * 
 * NOTE! --> to set up the program correctly, change on/off settings, file names and each algorithm's parameters
 * 			 (e.g. K, learning rates, epochs, network topology) at the top of the Main class the way you want it.
 * 		 --> The code runs on the assumption that dataset loaded from the csv files are clean, with no empty lines and
 * 			 only containint integers. There were no safety checks developed!
 * 
 * Brief run time:
 * 	1. Initialise data
 *  2. Run each algorithm using settings:
 *  	- Run each algorithm using 2 fold test (train on set1, test set2 and vice versa)
 *  	- Print the results*/

package coursework;

import java.util.concurrent.*;	// allows concurency where finding the best mlp structure

/* ============================ Main method to run all of the calculations ============================
 * Brief runtime:
 * 	- loads data into set1 and set2
 *  - Categorising using Nearest Neighbor using Euclidean distance (both sets against each other)
 *  - Categorising using K Nearest Neighbor using Euclidean distance (also, both sets against each other)
 *  - Running MLP algorithm (both sets against each other)
 *  - Running SVM algorithm (also both sets against each other)
 *  - Printing the results and confusion matrixes
 * */
public class Main
{
	// ---------------- ON/OFF settings --------
	final static boolean CONFUSIONMATRIX 	= 	false; 		// turn confusion matrix on/off
	final static boolean NNALGORITHM 		= 	true; 		// turn NN Algorithm on/off
	final static boolean KNNALGORITHM 		= 	false; 		// turn KNN Algorithm on/off
	final static boolean WEIGHTEDKNN 		= 	true; 		// switch between weighted / first come majority tie-breaker KNN Algorithm
	final static boolean MLPALGORITHM 		= 	true; 		// turn MLP Algorithm on/off
	final static boolean SVMALGORITHM 		= 	false; 		// turn SVM Algorithm on/off
	final static boolean FINDMLPBEST 		= 	false; 		// turn best MLP on/off -> settings directly inside the method!
		
	// ------------------ File names --------------
	final static String FILE1 = "dataSet1.csv";
	final static String FILE2 = "dataSet2.csv";
	
	// K number in KNN algorithm ----------- KNN ---------------------
	final static int NEIGHBOURSNUMBER = 4; // must be at least 2 (best found with 4, than 6 for weighted)
	
	// number of neurons in each layer ------------------ MLP ---------------------
	final static int[] NEURONS = { 50, 50, 50 }; // must be at least 1 hidden layer!
	// learning rate
	final static double LEARNINGRATE = 0.01;
	// learning epochs
	final static int EPOCHS = 50;
	// number of hidden layers
	final static int HIDDENLAYERS = NEURONS.length;
	// learning rate for SVM -----------------------------SVM ---------------------
	final static double SVM_LEARNING_RATE = 0.001;
	final static int SVM_EPOCHS = 50;
	
	// number of possible outputs
	final static int OUTPUTSNUMBER = 10;
	
	public static void main(String[] args)
	{
		// datasets declaration
		DataSet set1 = null;
		DataSet set2 = null;
		
		// datasets initialisation
		set1 = DataLoader.loadData(FILE1, set1);
		set2 = DataLoader.loadData(FILE2, set2);
		
		// check if data and parameters for the algorithms are loaded correctly, than run each
		if (set1 != null && set2 != null && set1.cols == set2.cols && NEURONS.length == HIDDENLAYERS) {
			double nnAverage = runNNAlg(set1, set2);
			double knnAverage = 0;
			if (NEIGHBOURSNUMBER > 1) knnAverage = runKNNALG(set1, set2);
			double mlpAverage = 0;
			if (HIDDENLAYERS > 0) mlpAverage = runMLPAlg(set1, set2);
			double svmAverage = 0;
			if (SVMALGORITHM) svmAverage = runSVMAlg(set1, set2);
			if (FINDMLPBEST) runFindMlpBest(set1, set2);
			
			// print results
			System.out.println("\n<-------------->\nPrinting results\n<-------------->\n");
			System.out.println(NNALGORITHM ? "NNAlg average:  	" + nnAverage : "NN Algorithm skipped");
			System.out.println(KNNALGORITHM ? "KNNAlg average: 	" + knnAverage : "KNN Algorithm skipped");
			System.out.println((MLPALGORITHM && HIDDENLAYERS > 0) ? "MLPAlg average:		" + mlpAverage : "MLP Algorithm skipped");
			System.out.println(SVMALGORITHM ? "SVMAlg average:         " + svmAverage : "SVM Algorithm skipped");
		} else {
			System.out.println("Incorrect data, terminating...");
		}
		System.out.println("\nTerminated!");
	}
	
	/* Function to find best MLP parameters (bruteforcing each scenario -> extremely expensive!!!)
	 * @par set1, set2 -> datasets */
	public static void runFindMlpBest(DataSet set1, DataSet set2)
	{
	    System.out.println("Finding the best setting for MLP:\n");

	    // set parameters for training
	    int maxNeurons = 128;	// max amount of neurons to try
	    int minNeurons = 16; 	// min amount of neurons to try
	    int epochs = 20; 		// epochs to start training each model
	    int maxEpochs = 50; 	// max epochs to train each model
	    int startHidLay = 2; 	// start amount of hidden layers
	    int maxHidLay = 4; 		// end amount of hidden layers
	    int updateEpochs = 10; 	// update epochs each cycle by

	    // best-so-far (wrapped so lambdas can modify them)
	    final double[] bestAverage = { 0.0 };
	    final int[][] bestModel = { null };
	    final int[] bestEpochs = { -1 };
	    final Object bestLock = new Object();	// lock so multiple threads do not update variables

	    // number of threads
	    int threads = Runtime.getRuntime().availableProcessors();
	    // create thread pool
	    ExecutorService pool = Executors.newFixedThreadPool(threads);

	    try {
	    	// loop through different number of hidden layers
	        for (int noHidLay = startHidLay; noHidLay <= maxHidLay; noHidLay++) {
	            int changeEpochs = epochs; // init temp epochs so epochs stays unchanged
	            while (changeEpochs <= maxEpochs) {
	                // init array of hidden layers and fill with minNeurons
	                int[] hiddenLayers = new int[noHidLay];
	                for (int i = 0; i < hiddenLayers.length; i++) hiddenLayers[i] = minNeurons;

	                // loop through number of hidden neurons inside each hidden layer
	                while (true) {
	                    // SNAPSHOTS for multithread task (VERY IMPORTANT)
	                    final int[] hiddenSnapshot = hiddenLayers.clone();
	                    final int epochsSnapshot = changeEpochs;

	                    pool.execute(() -> {
	                        // Calculation of set 2 against set1 (copy train set - set1)
	                        DataSet set1copy = new DataSet(set1);
	                        int[][] confusionMatrix1 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER];
	                        double mlpPercentage1 = MLPAlg.MLPClassify(
	                                set1copy, set2,
	                                hiddenSnapshot.length, hiddenSnapshot,
	                                LEARNINGRATE, epochsSnapshot,
	                                confusionMatrix1, OUTPUTSNUMBER
	                        );

	                        // Calculation of set1 against set2 (copy train set - set2)
	                        DataSet set2copy = new DataSet(set2);
	                        int[][] confusionMatrix2 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER];
	                        double mlpPercentage2 = MLPAlg.MLPClassify(
	                                set2copy, set1,
	                                hiddenSnapshot.length, hiddenSnapshot,
	                                LEARNINGRATE, epochsSnapshot,
	                                confusionMatrix2, OUTPUTSNUMBER
	                        );

	                        // compute average between the calculations
	                        double mlpAverage = (mlpPercentage1 + mlpPercentage2) / 2.0;

	                        // better result found (used lock for synchronisation)
	                        synchronized (bestLock) {
	                            if (mlpAverage > bestAverage[0]) {
	                                bestAverage[0] = mlpAverage;
	                                bestModel[0] = hiddenSnapshot.clone(); // keep a stable copy
	                                bestEpochs[0] = epochsSnapshot;

	                                System.out.print("New best model: ");
	                                for (int i : bestModel[0]) System.out.print(i + " ");
	                                System.out.println("Epochs: " + bestEpochs[0] + " Average: " + bestAverage[0]);
	                            }
	                        }
	                    });

	                    // end of calculations (producer continues)

	                    // increment like an odometer from the last index
	                    int index = hiddenLayers.length - 1;

	                    while (index >= 0 && hiddenLayers[index] == maxNeurons) {
	                        hiddenLayers[index] = minNeurons;
	                        index--;
	                    }

	                    // if alg finished
	                    if (index < 0) break;

	                    hiddenLayers[index]++; // normal increment
	                }

	                changeEpochs += updateEpochs;
	            }
	        }
	    } finally {
	    	// end the thread pool
	        pool.shutdown();
	        try {
	            pool.awaitTermination(365, TimeUnit.DAYS);	// in case something else still running
	        } catch (InterruptedException e) {
	            Thread.currentThread().interrupt();
	        }
	    }

	    // final best print (optional)
	    System.out.println("\nDONE.");
	    if (bestModel[0] != null) {
	        System.out.print("Best model: ");
	        for (int v : bestModel[0]) System.out.print(v + " ");
	        System.out.println("Epochs: " + bestEpochs[0] + " Average: " + bestAverage[0]);
	    } else {
	        System.out.println("No model improved over 0.0.");
	    }
	}
		
	/* Method to call Nearest Neighbour algorithm from NNAlg class
	 * @par set1, set2 -> datasets
	 * @ret double -> average between both sides calculations percentage*/
	public static double runNNAlg(DataSet set1, DataSet set2)
	{
		if (!NNALGORITHM) return 0;
		
		System.out.println("<---------------------------------------->\nNearest Neighbour Algorithm (NN Algorithm)\n<---------------------------------------->\n");
		
		int[][] confusionMatrix1 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER]; // confusion matrix declaration
		
		// Calculation of set2 against set1
		double nnPercentage1 = NNAlg.NNClassify(set1, set2, confusionMatrix1);
		
		printConfusionMatrix(confusionMatrix1);
		
		System.out.println("Percentage of correct anaswers: " + nnPercentage1);
		
		System.out.println("\nSwitching test sets...\n");
		
		int[][] confusionMatrix2 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER]; // confusion matrix declaration
		
		// Calculation of set1 against set2
		double nnPercentage2 = NNAlg.NNClassify(set2, set1, confusionMatrix2);
		
		printConfusionMatrix(confusionMatrix2);
		
		System.out.println("Percentage of correct anaswers: " + nnPercentage2);
		
		// compute average between the calculations
		double nnAverage = (nnPercentage1 + nnPercentage2) / 2;
		
		System.out.println("\nIn average, correct results using NN Algorithm are: " + nnAverage);
		
		return nnAverage;
	}
	
	/* Method to call K Nearest Neighbour algorithm from KNNAlg class
	 * @par set1, set2 -> datasets
	 * @ret double -> average between both side calculations percentage*/
	public static double runKNNALG(DataSet set1, DataSet set2)
	{
		if (!KNNALGORITHM) return 0;
		
		System.out.println("\n<-------------------------------------------->\nK - Nearest Neighbour Algorithm (NN Algorithm)\n<-------------------------------------------->\n");

		int[][] confusionMatrix1 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER]; // confusion matrix declaration
		
		// Calculation of set2 against set1
		double knnPercentage1 = NNAlg.KNNClassify(set1, set2, NEIGHBOURSNUMBER, confusionMatrix1, WEIGHTEDKNN);
		
		printConfusionMatrix(confusionMatrix1);

		System.out.println("Percentage of correct anaswers: " + knnPercentage1);
		
		System.out.println("\nSwitching test sets...\n");
		
		int[][] confusionMatrix2 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER]; // confusion matrix declaration
		
		// Calculation of set1 against set2
		double knnPercentage2 = NNAlg.KNNClassify(set2, set1, NEIGHBOURSNUMBER, confusionMatrix2, WEIGHTEDKNN);
		
		printConfusionMatrix(confusionMatrix2);
		
		System.out.println("Percentage of correct anaswers: " + knnPercentage2);
		
		// compute average between the calculations
		double knnAverage = (knnPercentage1 + knnPercentage2) / 2;
		
		System.out.println("\nIn average, correct results using K-NN Algorithm are: " + knnAverage);
		
		return knnAverage;
	}

	/* Method to call MLP algorithm from MLPAlg class
	 * @par set1, set2 -> datasets
	 * @ret double -> average between both sides calculations percentage
	 * */
	public static double runMLPAlg(DataSet set1, DataSet set2)
	{
		if (!MLPALGORITHM) return 0;
		
		System.out.println("\n<---------------------------------------------->\nMulti-Layer Perceptron Algorithm (MLP Algorithm)\n<---------------------------------------------->");
		
		int[][] confusionMatrix1 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER]; // confusion matrix declaration
		
		// Calculation of set 2 against set1 (I will give it copy of the train set - set1)
		DataSet set1copy = new DataSet(set1);
		double mlpPercentage1 = MLPAlg.MLPClassify(set1copy, set2, HIDDENLAYERS, NEURONS, LEARNINGRATE, EPOCHS, confusionMatrix1, OUTPUTSNUMBER);
		
		printConfusionMatrix(confusionMatrix1);
		
		System.out.println("Percentage of correct anaswers: " + mlpPercentage1);
		
		System.out.println("\nSwitching test sets...");
		
		int[][] confusionMatrix2 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER]; // confusion matrix declaration
		
		// Calculation of set1 against set2 (I will give it copy of the train set - set2)
		DataSet set2copy = new DataSet(set2);
		double mlpPercentage2 = MLPAlg.MLPClassify(set2copy, set1, HIDDENLAYERS, NEURONS, LEARNINGRATE, EPOCHS, confusionMatrix2, OUTPUTSNUMBER);
		
		printConfusionMatrix(confusionMatrix2);
		
		System.out.println("Percentage of correct anaswers: " + mlpPercentage2);
		
		// compute average between the calculations
		double mlpAverage = (mlpPercentage1 + mlpPercentage2) / 2;
		
		System.out.println("\nIn average, correct results using MLP Algorithm are: " + mlpAverage);
		
		return mlpAverage;
	}
	
	/* Method to call SVM algorithm from SVMAlg class
	 * @par set1, set2 -> datasets
	 * @ret double -> average between both sides calculations percentage
	 * */
	public static double runSVMAlg(DataSet set1, DataSet set2)
	{
	    System.out.println("\n<-------------------------->\nSupport Vector Machine (SVM)\n<-------------------------->\n");

	    int[][] confusionMatrix1 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER];
	    DataSet set1copy = new DataSet(set1);
	    
	    // calculation of set 2 against set 1
	    double svm1 = SVMAlg.SVMClassify(set1copy, set2, OUTPUTSNUMBER, SVM_LEARNING_RATE, SVM_EPOCHS, confusionMatrix1);
	    printConfusionMatrix(confusionMatrix1);

	    // switch sets
	    System.out.println("\nSwitching test sets...\n");
	    
	    int[][] confusionMatrix2 = new int[OUTPUTSNUMBER][OUTPUTSNUMBER];
	    DataSet set2copy = new DataSet(set2);
	    
	    // calculation of set 1 against set 2
	    double svm2 = SVMAlg.SVMClassify(set2copy, set1, OUTPUTSNUMBER, SVM_LEARNING_RATE, SVM_EPOCHS, confusionMatrix2);
	    printConfusionMatrix(confusionMatrix2);

	    double avg = (svm1 + svm2) / 2.0;
	    System.out.println("\nAverage SVM accuracy: " + avg);

	    return avg;
	}

	/* Method to print confusion matrix from the given 2d array
	 * @par result - matrix that stores all of the values 
	 * */
	public static void printConfusionMatrix(int[][] result)
	{
		if (!CONFUSIONMATRIX) return;
		
		System.out.println("\n----------------\nConfusion Matrix\n----------------");
		System.out.print("    ");
		
		// print first line (just numbers 0 - 9)
		for (int classNumber = 0; classNumber < OUTPUTSNUMBER; classNumber++) System.out.print(classNumber + "   ");
		
		System.out.println("    \n-----------------------------------------");
		
		// print all other lines with the results
		for (int rowIndex = 0; rowIndex < result.length; rowIndex++) {
			System.out.print(rowIndex + " | ");
			
			for (int columnIndex = 0; columnIndex < result[rowIndex].length; columnIndex++) {
				System.out.print(result[rowIndex][columnIndex] + 
						((result[rowIndex][columnIndex] > 9) ? ((result[rowIndex][columnIndex] > 99) ? " " : "  ") : "   "));
			}
			System.out.println("\n");
		}
		
		System.out.println();
	}
}
