package algorithm;

import data.*;
import util.SimpleTools;

/**
 * Cost-sensitive multi-label active learning.
 * 
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 *
 */
public class Cmale {
    /**
	 * The dataset.
	 */
	MultiLabelData dataset;
	
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
	 ********************** 
	 * The first constructor. Data and labels are stored in one file.
	 * 
	 * @paraDataFilename The data filename.
	 ********************** 
	 */
	public Cmale(String paraArffFilename, int paraNumConditions, int paraNumLabels) {
		dataset = new MultiLabelData(paraArffFilename, paraNumConditions, paraNumLabels);
 
		//Data matrix initialization.
		numInstances = dataset.getNumInstances();
		numConditions = dataset.getNumConditions();
		numLabels = paraNumLabels;
		
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
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < numLabels; j++) {
				if ((predictedLabelMatrix[i][j] == -1) && (dataset.getLabel(i, j) == 1)) {
					resultCost += falseNegativeCost;
				} else if ((predictedLabelMatrix[i][j] == 1) && (dataset.getLabel(i, j) == -1)) {
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
				tempDistance = dataset.distance(i, j);
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
            	tempNewDistance = dataset.distance(i, j);
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
        representativenessRankArray = SimpleTools.mergeSortToIndices(representativenessArray);

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
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
    public String toString() {
    	return dataset.toString();
    }//Of toString
    
	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
    public static void readDataTest() {
    	//Cmale tempCmale = new Cmale("D:/data/simpleiris.arff", 4, 1);
    	Cmale tempCmale = new Cmale("D:/data/multilabel/flags.arff", 14, 12);
    	System.out.println("The data is:\r\n"  + tempCmale);
    }//Of readDataTest

    /**
	 ********************** 
	 * The entrance.
	 ********************** 
	 */
    public static void main(String[] args) {
    	readDataTest();
    	System.out.println("Finish.");
    }//Of main
    
}// Of class Cmale
