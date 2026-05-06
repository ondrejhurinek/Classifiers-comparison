/* Class to load sizes into dataset and to load data into an arrays 
 * class is responsible to load data from the files correctly 
 * class does not work with improper csv files (e.g. empty lines, non int values etc..) */

package coursework;

import java.io.File;
import java.io.IOException;
import java.util.InputMismatchException;
import java.util.Scanner;

class DataLoader
{
	/* Method to load data into the object of DataSet, also finds number of rows and columns in the dataset,
	 * if number of column already found, compares to assure that the amount of data is the same in
	 * each row (meaning, it is the same type data that i am comparing)
	 * @par FILE -> name of the file to be loaded
	 * @par set -> dataset to load data into
	 * @ret DataSet -> return updated dataset with the new data*/
	public static DataSet loadData(String FILE, DataSet set)
	{			
		// initialise dataset
		set = new DataSet();
				
		// to find number of rows and columns / if not found or any error e.g. file emptyp, return null
		if (!findNoRowCol(FILE, set)) return null;
		
		System.out.println("Number of rows:    " + set.rows);
		System.out.println("Number of columns: " + set.cols + "\nInitialising 2d array..");
		
		// if file read correctly, return it, otherwise, return null
		if (readFile(FILE, set)) return set;
		
		return null;
	}
	
	/* Method to find number of rows and columns in a file for arrays initialisations
	 * Method also checks if there is the same number of numbers in each row
	 * @par nameOfFile -> name of the file to be read
	 * @par set -> struct (object) to store number of rows and columns and arrays
	 * @ret bool -> if there was an error during file reading */
	private static boolean findNoRowCol(String nameOfFile, DataSet set)
	{
		try {
			File file = new java.io.File(nameOfFile);
			Scanner scanner = new Scanner(file); // scanner that reads each line
			
			System.out.println(nameOfFile + " file found! Reading data..\nPreparing to initialise 2d array..");
			
			if (!scanner.hasNextLine()) { // if there is no line to read - file empty
				scanner.close();
				return false;
			}
			
			while (scanner.hasNextLine()) { // read all of the lines
				String line = scanner.nextLine().trim();
				if (!readLine(line, set)) { // read all the numbers inside that line, return false if data read incorrectly
					scanner.close();
					return false;
				}
			}
			scanner.close();
		}
		// exception handleres in case data were read incorrectly
		catch (InputMismatchException | IOException exception) {
		    System.out.println("Error reading input file!");
		    set.rows = 0;
		    set.cols = 0;
		    return false;
		}
		return true;
	}

	/* Method to count number of integers in the line
	 * @par line -> line to read integers from
	 * @par set -> set to be initialised
	 * @ret bool -> true if number of integers correct, false if incorrect or reading interrupted*/
	private static boolean readLine(String line, DataSet set)
	{
		int tempInt = 0; // temporary int to store number of integers inside each line
		try (Scanner lineScanner = new Scanner(line)) { // read the line and use ',' as delimiter, than increase number of set.rows
			lineScanner.useDelimiter("[,]+");
			set.rows++;
			if (!lineScanner.hasNextInt()) { //if there is no integer to read
				set.rows = 0;
				set.cols = 0;
				return false;
			}
			if (set.cols == 0) { // count all integers in the first row
				while (lineScanner.hasNextInt()) {
					set.cols++;
					lineScanner.nextInt();
				}
			}
			else { // compare number of integers in the next lines
				while (lineScanner.hasNextInt()) { // for each line
					tempInt++; // store number of integers
					lineScanner.nextInt();
				}
				if (tempInt != set.cols) { // and compare the result
					System.out.println("Incorrect number of integers in the line " + set.rows);
					set.rows = 0;
					set.cols = 0;
					return false;
				}
				tempInt = 0;
			}
		}
		catch (InputMismatchException inputMismatchException) { // exception handler if incorrect data read
			System.out.println("Incorrect input encountered!");
			set.rows = 0;
			set.cols = 0;
			return false;
		}
		return true;
	}
	
	/* Method to read a file and initialise array from trainingData (sizes already found)
	 * @par nameOfFile -> name of file to initialise data from
	 * @par set -> set to be initialised
	 * @ret boolean -> return true if initialisation completed correctly, false otherwise*/
	private static boolean readFile(String nameOfFile, DataSet set)
	{
		try {
			// close scanner and reopen again and read from the beginning of the file for data
			File file = new java.io.File(nameOfFile);
			
			// for scanner to close if failed
			try (Scanner scanner = new Scanner(file)) {
				init2dArray(scanner, set);
			}
						
			// print2dArray(arr2d);
			System.out.println("2d Array initialised..");
			System.out.println("Done!\n");
		} catch (IOException inputOutputException) {
			System.out.println("No access to read data!");
			return false;
		}
		return true;
	}
	
	/* Method to initialise 2d array with pixel values and 1d array with their labels
	 * @par scanner -> scanner that is loaded with correct file
	 * @par set -> set to be initialised*/
	private static void init2dArray(Scanner scanner, DataSet set)
	{
		// rows and columns to loop through
		int row = 0;
		
		// initialise arrays inside set object
		set.arr2d = new int[set.rows][set.cols - 1];
		set.arr1d = new int[set.rows];
		
		while (scanner.hasNextLine()) {
			// save the line into line string
			String line = scanner.nextLine().trim();
			// create new scanner to read that line
			Scanner lineScanner = new Scanner(line);
			// set delimiter for the lineScanner
			lineScanner.useDelimiter("[,]+");
			
			// read 64 values in each line, init label after
			for (int column = 0; column < set.cols - 1; column++) {
				set.arr2d[row][column] = lineScanner.nextInt();
			}
			set.arr1d[row] = lineScanner.nextInt();

			row++;
			lineScanner.close();
		}
	}
}