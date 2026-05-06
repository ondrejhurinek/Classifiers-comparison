/* Class to store arrays and their sizes
 * stores arrays of training and testing data and their size
 * defines methods to be used when working with the data */

package coursework;

import java.util.Random;

public class DataSet
{
	// ---> datafield <---
	// number of rows and columns
	int rows = 0;
	int cols = 0;
	
	// 2d for 8x8 pixel matrix (actual pixels data)
	int[][] arr2d;
	
	// 1d for labels (labels of the data)
	int[] arr1d;
	
	// ---> constructors <---
	// default
	public DataSet() {}
	
	// copy constructor
	public DataSet(DataSet toBeCopied) 
	{
		rows = toBeCopied.rows;
		cols = toBeCopied.cols;
		arr2d = new int[toBeCopied.arr2d.length][toBeCopied.arr2d[0].length];
		arr1d = new int[toBeCopied.arr1d.length];
		
		for (int index = 0; index < arr1d.length; index++) {
			arr1d[index] = toBeCopied.arr1d[index];
		}
		
		for (int index = 0; index < arr2d.length; index++) {
			for (int index2 = 0; index2 < arr2d[index].length; index2++) {
				arr2d[index][index2] = toBeCopied.arr2d[index][index2];
			}
		}
	}
	
	// member methods
	/* Method to shuffle dataset
	 * Shuffling arr2d and arr1d same way*/
	public void shuffleDataSet()
	{
		Random rand = new Random();
		int temp2dArr[] = new int[cols];
		int temp1d;
		
		for (int index = 0; index < rows; index++) {
			// random value of an index that will be shuffled
			int indexReshuffle = rand.nextInt(rows);
			
			// store temporary values
			temp2dArr = arr2d[index];
			temp1d = arr1d[index];
			
			// assign new values
			arr2d[index] = arr2d[indexReshuffle];
			arr1d[index] = arr1d[indexReshuffle];
			
			// assign into random index
			arr2d[indexReshuffle] = temp2dArr;
			arr1d[indexReshuffle] = temp1d;
		}
	}
}