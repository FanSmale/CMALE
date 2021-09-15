package algorithm;

import java.io.FileReader;
import java.util.Arrays;

import weka.core.Instances;

/**
 * Cost-sensitive multi-label active learning.
 * 
 * @author Fan Min
 *
 */
public class Cmale {

	  /**
     * The data.
     */
    Instances dataset = null;
    
    /**
	 * The number of instances.
	 */
	int numInstances;

    /**
	 * The number of conditions.
	 */
	int numConditions;

    /**
	 * The number of labels.
	 */
	int numLabels;

    /**
	 * The data matrix.
	 */
	double[][] dataMatrix;

	/**
	 * The label matrix.
	 */
	int[][] labelMatrix;

	/**
	 * The predicted label matrix.
	 */
	int[][] predictedLabelMatrix;

	/**
	 * The misclassification cost for classifying an instance without a label as
	 * with.
	 */
	double falsePositiveCost;

	/**
	 * The misclassification cost for classifying an instance with a label as
	 * without.
	 */
	double falseNegativeCost;

	/**
	 * The number of queries.
	 */
	int numQueries;

	/**
	 * The teacher cost for each query.
	 */
	double teacherCost;

	/**
	 * The nueual network for regression. AnnNetwork annRegressor;
	 */

	/**
	 * The matrix factorization regressor. MatrixFactorization mfRegressor;
	 */

	/**
	 * Representativeness of each instance.
	 */
	double[] representativenessArray;

	/**
	 * Representativeness of rank each instance.
	 */
	int[] representativenessRankArray;

	/**
	 * The number of queries for respective labels.
	 */
	int[] labelQueryCountArray;


    /**
     * Manhattan distance.
     */
    public static final int MANHATTAN = 0;

    /**
     * Euclidean distance.
     */
    public static final int EUCLIDEAN = 1;

    /**
     * The distance measure.
     */
    public int distanceMeasure = EUCLIDEAN;
    
	/**
	 ********************** 
	 * The first constructor. Data and labels are stored in two files.
	 * 
	 * @paraDataFilename The data filename.
	 ********************** 
	 */
	public Cmale(String paraDataFilename, int paraLabelFilename, int paraNumInstances, int paraNumConditions,
			int paraNumLabels) {
        try {
            FileReader tempReader = new FileReader(paraDataFilename);
            dataset = new Instances(tempReader);
            // The last attribute is the decision class.
            dataset.setClassIndex(dataset.numAttributes() - 1);
            tempReader.close();
        } catch (Exception ee) {
            System.out.println("Error occurred while trying to read \'" + paraDataFilename
                    + "\' in GeneralAnn constructor.\r\n" + ee);
            System.exit(0);
        }//of try

		numQueries = 0;
		representativenessArray = new double[paraNumInstances];
		representativenessRankArray = new int[paraNumInstances];
		labelQueryCountArray = new int[paraNumLabels];
		//annRegressor = new AnnRegressor(dataMatrix, quriedLabelMatrix); 
	}// Of the first constructor

	/**
	 ********************** 
	 * The second constructor. Data and labels are stored in one file.
	 * 
	 * @paraDataFilename The data filename.
	 ********************** 
	 */
	public Cmale(String paraArffFilename, int paraNumConditions, int paraNumLabels) {
        try {
            FileReader tempReader = new FileReader(paraArffFilename);
            dataset = new Instances(tempReader);
            // The last attribute is the decision class.
            dataset.setClassIndex(dataset.numAttributes() - 1);
            tempReader.close();
        } catch (Exception ee) {
            System.out.println("Error occurred while trying to read \'" + paraArffFilename
                    + "\' in GeneralAnn constructor.\r\n" + ee);
            System.exit(0);
        }//of try

        //Data matrix initialization.
		numInstances = dataset.numInstances();
		numConditions = paraNumConditions;
		numLabels = paraNumLabels;
		
        dataMatrix = new double[numInstances][numConditions];
        for (int i = 0; i < dataMatrix.length; i++) {
			for (int j = 0; j < dataMatrix[i].length; j++) {
				dataMatrix[i][j] = dataset.instance(i).value(j);
			}//Of for j
		}//Of for i
        
        //Label matrix initialization.
        labelMatrix = new int[numInstances][numLabels];
        for (int i = 0; i < labelMatrix.length; i++) {
			for (int j = 0; j < labelMatrix[0].length; j++) {
				labelMatrix[i][j] = (int)dataset.instance(i).value(numConditions + j);
			}//Of for j
		}//Of for i
        
        //Predicted label matrix initialization.
        predictedLabelMatrix = new int[numInstances][numLabels];

        numQueries = 0;
		representativenessArray = new double[numInstances];
		representativenessRankArray = new int[numInstances];
		labelQueryCountArray = new int[numLabels];
	}// Of the second constructor

	/**
	 ********************** 
	 * Compute the total cost. Including the teacher cost and misclassification
	 * cost.
	 ********************** 
	 */
	public double computeTotalCost() {
		double resultCost = 0;
		for (int i = 0; i < labelMatrix.length; i++) {
			for (int j = 0; j < dataMatrix[0].length; j++) {
				if ((predictedLabelMatrix[i][j] == -1) && (labelMatrix[i][j] == 1)) {
					resultCost += falseNegativeCost;
				} else if ((predictedLabelMatrix[i][j] == 1) && (labelMatrix[i][j] == -1)) {
					resultCost += falsePositiveCost;
				} else if ((predictedLabelMatrix[i][j] == 0)) {
					System.out.println("Internal error! The predicted label should be either -1 or +1, position: (" + i
							+ ", " + j + ").");
					System.exit(0);
				} // Of if
			} // Of for j
		} // Of for i

		resultCost += numQueries * teacherCost;
		return resultCost;
	}// Of computeTotalCost

	/**
	 ********************** 
	 * Predict for a given label of an instance.
	 * 
	 * @param paraI
	 *            The instance index.
	 * @param paraJ
	 *            The label index.
	 ********************** 
	 */
	public double predict(int paraI, int paraJ) {
		// double[] tempPredictions = classifier.predict(dataMatrix[paraI]);
		// return tempPredictions[paraJ];
		return 0;
	}// Of predict

	/**
	 ********************** 
	 * Predict for all labels.
	 ********************** 
	 */
	public double[][] predict() {
		double[][] resultMatrix = null;
		// for (int i = )
		// double[] tempPredictions = classifier.predict(dataMatrix[paraI]);
		// return tempPredictions[paraJ];
		return resultMatrix;
	}// Of predict

	/**
	 ********************** 
	 * Predict for all labels.
	 ********************** 
	 */
	public double[][] matrixFactorizationPredict() {
		double[][] resultMatrix = null;
		// resultMatrix = mfRegressor.predict();
		return resultMatrix;
	}// Of predict

	/**
	 ********************** 
	 * Train.
	 ********************** 
	 */
	public void train() {
		// Step 1. Train the ANN regressor
		// Step 2. Train the matrix factorization regressor
	}// Of train

	/**
	 ********************** 
	 * Compute the distance between two instances
	 * @param paraI The first index.
	 * @param paraI The second index.
	 ********************** 
	 */
    public double distance(int paraI, int paraJ) {
        int resultDistance = 0;
        double tempDifference;
        switch (distanceMeasure) {
            case MANHATTAN:
                for (int i = 0; i < numConditions; i++) {
                    tempDifference = dataMatrix[paraI][i] - dataMatrix[paraJ][i];
                    //Sum up the distance.
                    if (tempDifference < 0) {
                        resultDistance -= tempDifference;
                    } else {
                        resultDistance += tempDifference;
                    }//of if
                }//of for i
                break;

            case EUCLIDEAN:
                for (int i = 0; i < numConditions; i++) {
                    tempDifference = dataMatrix[paraI][i] - dataMatrix[paraJ][i];
                    resultDistance += tempDifference * tempDifference;
                }//of for i
                break;
            default:
                System.out.println("Unsupported distance measure: " + distanceMeasure);
        }//of switch
        return resultDistance;
    }//of distance

	/**
	 ********************** 
	 * Learn the classifier
	 * @param paraI The first index.
	 * @param paraI The second index.
	 ********************** 
	 */
    public void learn(int paraColdStartRounds, double paraDc) {
        // Step 1. Calculate the representativeness of each instance.
        // Calculate density using Gaussian kernel.
        double[] tempDensityArray = new double[numInstances];
        double tempDistance = 0;
		for (int i = 0; i < numInstances; i++) {
			tempDensityArray[i] = 0;
			for (int j = 0; j < numInstances; j++) {
				tempDistance = distance(i, j);
				tempDensityArray[i] += Math.exp(-tempDistance * tempDistance / paraDc / paraDc);
			} // Of for j
		} // Of for i

        // Calculate distance to its master.
        int[] tempMasterArray = new int[numInstances];
        double tempNewDistance = 0;
        double[] tempDistanceToMasterArray = new double[numInstances];
        for (int i = 0; i < numInstances; i++) {
        	tempMasterArray[i] = -1;
            double tempNearestDistance = 100000;
            for (int j = 0; j < numInstances; j++) {
            	if (tempDensityArray[j] <= tempDensityArray[i]) {
            		continue;
            	}//Of if
            	
                //Is this one closer?
            	tempNewDistance = distance(i, j);
                if (tempNewDistance < tempNearestDistance) {
                    tempNearestDistance = tempNewDistance;
                    tempDistanceToMasterArray[i] = tempNewDistance;
                    tempMasterArray[i] = j;
                }//of if
            }//Of for j
        }//Of for i

        // Representativeness.
        for (int i = 0; i < numInstances; i++) {
        	representativenessArray[i] = tempDensityArray[i] * tempDistanceToMasterArray[i];
        }//Of for i
        
        // Sort instances according to representativeness.
        representativenessRankArray =  mergeSortToIndices(representativenessArray);

        // Step 2. Cold start stage. Only consider instance representativeness
        // and label diversity
        for (int i = 0; i < paraColdStartRounds; i++) {
            // Cold start query here.
        } // Of for i

        // Step 3. Regular learning.
        double tempExpectedCostReduction = 1e5;
        while (true) {
            //Step 3.1 Query the most informative labels.
            numQueries++;

            //Step 3.2 Train the regressors.
            train();

            //Step 3.3 It is cost-effective?
            if (tempExpectedCostReduction < teacherCost) {
                break;
            } // Of if
        } // Of while
    }// Of learn

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged. The method should be tested further. <br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2]. <br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */
	public static int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];// For merge sort.

		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i

		// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;

		while (tempCurrentLength < tempLength) {
			// Divide into a number of groups.
			// Here the boundary is adaptive to array length not equal to 2^k.
			for (int i = 0; i < Math.ceil((tempLength + 0.0) / tempCurrentLength / 2); i++) {
				// Boundaries of the group
				tempFirstStart = i * tempCurrentLength * 2;

				tempSecondStart = tempFirstStart + tempCurrentLength;

				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {
					tempSecondEnd = tempLength - 1;
				} // Of if

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;

				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;
					} // Of for j
					break;
				} // Of if

				while ((tempFirstIndex <= tempSecondStart - 1)
						&& (tempSecondIndex <= tempSecondEnd)) {

					if (paraArray[resultMatrix[tempIndex
							% 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex
									% 2][tempSecondIndex]]) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempFirstIndex];
						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempSecondIndex];
						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;
				} // Of while

				// Remaining part
				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
							% 2][j];
					tempCurrentIndex++;
				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
							% 2][j];
					tempCurrentIndex++;
				} // Of for j
			} // Of for i

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices
	
	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
    public String toString() {
    	String resultString = "The data has " + numInstances + " instances, " + numConditions + " conditions, and " + numLabels + " labels.";
    	resultString += "\r\nData\r\n" + Arrays.deepToString(dataMatrix);
    	resultString += "\r\nLabel\r\n" + Arrays.deepToString(labelMatrix);
    	return resultString;
    }//Of toString
    
	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
    public static void readDataTest() {
    	Cmale tempCmale = new Cmale("D:/data/simpleiris.arff", 4, 1);
    	System.out.println("The data is:\r\n"  + tempCmale);
    }//Of readDataTest

    /**
	 ********************** 
	 * The entrance.
	 ********************** 
	 */
    public static void main(String[] args) {
    	readDataTest();
    }//Of main
    
}// Of class Cmale
