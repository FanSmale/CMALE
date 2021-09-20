package algorithm;

import java.io.File;
import java.io.RandomAccessFile;
import java.util.Arrays;

import algorithm.ann.MultiLabelAnn;
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
	 * The predicted label matrix. int[][] predictedLabelMatrix;
	 */

	/**
	 * The misclassification cost for classifying an instance without a label as
	 * with. double falsePositiveCost;
	 */

	/**
	 * The misclassification cost for classifying an instance with a label as
	 * without. double falseNegativeCost;
	 */

	/**
	 * The number of queries. int numQueries;
	 */

	/**
	 * The teacher cost for each query. double teacherCost;
	 */

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
	 * The number of queries for respective labels. int[] labelQueryCountArray;
	 */

	/**
	 ********************** 
	 * The first constructor. Data and labels are stored in one file.
	 * 
	 * @paraDataFilename The data filename.
	 ********************** 
	 */
	public Cmale(String paraArffFilename, int paraNumConditions, int paraNumLabels) {
		dataset = new MultiLabelData(paraArffFilename, paraNumConditions, paraNumLabels);

		// Data matrix initialization.
		numInstances = dataset.getNumInstances();
		numConditions = dataset.getNumConditions();
		numLabels = paraNumLabels;

		// Predicted label matrix initialization.
		// predictedLabelMatrix = new int[numInstances][numLabels];

		// numQueries = 0;
		representativenessArray = new double[numInstances];
		representativenessRankArray = new int[numInstances];
		// labelQueryCountArray = new int[numLabels];
	}// Of the second constructor

	/**
	 ********************** 
	 * Compute the total cost. Including the teacher cost and misclassification
	 * cost.
	 ********************** 
	 * public double computeTotalCost() { double resultCost = 0; for (int i = 0;
	 * i < numInstances; i++) { for (int j = 0; j < numLabels; j++) { if
	 * ((predictedLabelMatrix[i][j] == -1) && (dataset.getLabel(i, j) == 1)) {
	 * resultCost += falseNegativeCost; } else if ((predictedLabelMatrix[i][j]
	 * == 1) && (dataset.getLabel(i, j) == -1)) { resultCost +=
	 * falsePositiveCost; } else if ((predictedLabelMatrix[i][j] == 0)) {
	 * System.out.println("Internal error! The predicted label should be either
	 * -1 or +1, position: (" + i + ", " + j + ")."); System.exit(0); } // Of if
	 * } // Of for j } // Of for i
	 * 
	 * resultCost += numQueries * teacherCost; return resultCost; }// Of
	 * computeTotalCost
	 */

	/**
	 ********************** 
	 * Predict for a given label of an instance.
	 * 
	 * @param paraI
	 *            The instance index.
	 * @param paraJ
	 *            The label index.
	 ********************** 
	 *            public double predict(int paraI, int paraJ) { // double[]
	 *            tempPredictions = classifier.predict(dataMatrix[paraI]); //
	 *            return tempPredictions[paraJ]; return 0; }// Of predict
	 */

	/**
	 ********************** 
	 * Predict for all labels.
	 ********************** 
	 * public double[][] predict() { double[][] resultMatrix = null; // for (int
	 * i = ) // double[] tempPredictions =
	 * classifier.predict(dataMatrix[paraI]); // return tempPredictions[paraJ];
	 * return resultMatrix; }// Of predict
	 */

	/**
	 ********************** 
	 * Predict for all labels.
	 ********************** 
	 * public double[][] matrixFactorizationPredict() { double[][] resultMatrix
	 * = null; // resultMatrix = mfRegressor.predict(); return resultMatrix; }//
	 * Of predict
	 */

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
	 * Compute instance representativeness.
	 * 
	 * @param paraDc
	 *            The dc ratio.
	 ********************** 
	 */
	public void computeInstanceRepresentativeness(double paraDc) {
		// Step 1. Calculate the representativeness of each instance.
		// Calculate density using Gaussian kernel.
		double[] tempDensityArray = new double[numInstances];
		double tempDistance = 0;
		for (int i = 0; i < numInstances; i++) {
			tempDensityArray[i] = 0;
			for (int j = 0; j < numInstances; j++) {
				tempDistance = dataset.distance(i, j);
				// System.out.println("tempDistance = " + tempDistance);
				tempDensityArray[i] += Math.exp(-tempDistance * tempDistance / paraDc / paraDc);
				// System.out.printf("tempDensityArray[%d] = %f\r\n", i,
				// tempDensityArray[i]);
			} // Of for j
		} // Of for i

		// System.out.println("Gaussian, dc = " + paraDc);
		// System.out.println("tempDensityArray = " +
		// Arrays.toString(tempDensityArray));

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
				} // Of if

				// Is this one closer?
				tempNewDistance = dataset.distance(i, j);
				if (tempNewDistance < tempNearestDistance) {
					tempNearestDistance = tempNewDistance;
					tempDistanceToMasterArray[i] = tempNewDistance;
					tempMasterArray[i] = j;
				} // of if
			} // Of for j
		} // Of for i

		// Representativeness.
		// System.out.println("tempDensityArray = " +
		// Arrays.toString(tempDensityArray));
		// System.out.println("tempDistanceToMasterArray = " +
		// Arrays.toString(tempDistanceToMasterArray));
		for (int i = 0; i < numInstances; i++) {
			representativenessArray[i] = tempDensityArray[i] * tempDistanceToMasterArray[i];
		} // Of for i

		// Sort instances according to representativeness.
		representativenessRankArray = SimpleTools.mergeSortToIndices(representativenessArray);
	}// Of computeInstanceRepresentativeness

	/**
	 ********************** 
	 * Learn the classifier
	 * 
	 * @param paraColdStartRounds
	 *            Cold start rounds not considering label uncertainty.
	 * @param paraI
	 *            The second index.
	 ********************** 
	 */
	public void learn(int paraColdStartRounds, int paraNumAdditionalQueries,
			int paraNumQueryBatchSize, double paraDc, int[] paraFullConnectLayerNodes,
			int[] paraParallelLayerNodes, int paraPretrainRounds) throws Exception {
		// Step 1. Reset the dataset to clear learning inforamtion.
		dataset.reset();

		// Step 2. Prepare output file.
		File tempFile = new File("data/learn-results.txt");
		if (tempFile.exists()) {
			tempFile.delete();
		} // Of if
		RandomAccessFile tempRAFile = new RandomAccessFile("data/learn-results.txt", "rw");

		// Step 3. Calculate the representativeness of each instance.
		computeInstanceRepresentativeness(paraDc);

		// Step 4. Cold start stage. Only consider instance representativeness
		// and label diversity
		int[] tempLabelIndices = new int[paraNumQueryBatchSize];

		tempRAFile.writeBytes("Here is the whole process of learn(): \r\n");
		tempRAFile.writeBytes("Cold start: \r\n");
		for (int i = 0; i < paraColdStartRounds; i++) {
			// Cold start query here.
			tempLabelIndices = dataset.getScareLabels(paraNumQueryBatchSize);
			// System.out.println("tempLabelIndices = " +
			// Arrays.toString(tempLabelIndices));
			dataset.queryLabels(representativenessRankArray[i], tempLabelIndices);
			tempRAFile.writeBytes("Query instance #" + representativenessRankArray[i]
					+ " with labels #" + Arrays.toString(tempLabelIndices) + "\r\n");
		} // Of for i

		// Step 3. Pre-train an ANN.
		MultiLabelAnn tempAnn = new MultiLabelAnn(dataset, paraFullConnectLayerNodes,
				paraParallelLayerNodes, 0.05, 0.6, "sssss");
		for (int round = 0; round < paraPretrainRounds; round++) {
			if (round % 1000 == 999) {
				System.out.println("Round: " + round);
			} // Of if
			tempAnn.train();
		} // Of for n

		int[] tempInstanceIndices = new int[1];
		tempRAFile.writeBytes("Learning process: \r\n");
		// Step 4. Regular learning.
		for (int q = 0; q < paraNumAdditionalQueries; q++) {
			int[] tempIndices = tempAnn.getMostUncertainLabelIndices(paraNumQueryBatchSize);
			int[] tempLabelIndices2 = new int[paraNumQueryBatchSize];
			for (int j = 0; j < tempLabelIndices2.length; j++) {
				tempLabelIndices2[j] = tempIndices[j + 1];
			} // Of for j
			tempLabelIndices2[0] = tempIndices[1];
			dataset.queryLabels(tempIndices[0], tempLabelIndices2);
			tempRAFile.writeBytes("Query instance #" + tempIndices[0] + " with labels #"
					+ Arrays.toString(tempLabelIndices2) + "\r\n");

			tempInstanceIndices[0] = tempIndices[0];
			for (int round = 0; round < 1000; round++) {
				// if (round % 1000 == 999) {
				// System.out.println("EmphasizedTrain round: " + round);
				// } // Of if
				tempAnn.emphasizedTrain(10, tempInstanceIndices);
			} // Of for n
		} // Of for q

		double tempAccuray = tempAnn.test();
		System.out.println("The label scarcity array is: "
				+ Arrays.toString(dataset.computeLabelScarcityArray()));
		System.out.println("The label uncertainty matrix is: "
				+ Arrays.deepToString(tempAnn.computeLabelUncertaintyMatrix()));

		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("The training accuracy is: " + dataset.computeTrainingAccuracy());
		System.out.println("The total cost is: " + dataset.computeTotalCost());
		System.out.println(dataset.getCostDetail() + "\r\n");

		tempRAFile.writeBytes("The label scarcity array is: "
				+ Arrays.toString(dataset.computeLabelScarcityArray()) + "\r\n");
		tempRAFile.writeBytes("The label uncertainty matrix is: "
				+ Arrays.deepToString(tempAnn.computeLabelUncertaintyMatrix()) + "\r\n");
		tempRAFile.writeBytes("The accuracy is: " + tempAccuray + "\r\n");
		tempRAFile.writeBytes(
				"The training accuracy is: " + dataset.computeTrainingAccuracy() + "\r\n");
		tempRAFile.writeBytes("The total cost is: " + dataset.computeTotalCost() + "\r\n");
		tempRAFile.writeBytes(dataset.getCostDetail() + "\r\n");

		tempRAFile.close();
	}// Of learn

	/**
	 ********************** 
	 * Learn with randomly selected labels.
	 * 
	 * @param paraColdStartRounds
	 *            Cold start rounds not considering label uncertainty.
	 * @param paraI
	 *            The second index.
	 ********************** 
	 */
	public void randomSelectionLearn(int paraNumQueriedLabels, int[] paraFullConnectLayerNodes,
			int[] paraParallelLayerNodes, int paraTrainRounds) {
		dataset.reset();

		// Step 1. randomly select labels to query.
		// Here I present a trick to assure the number of queries.
		dataset.randomQuery(paraNumQueriedLabels);

		// Step 2. Train an ANN.
		MultiLabelAnn tempAnn = new MultiLabelAnn(dataset, paraFullConnectLayerNodes,
				paraParallelLayerNodes, 0.05, 0.6, "sssss");
		for (int round = 0; round < paraTrainRounds; round++) {
			if (round % 1000 == 999) {
				System.out.println("Round: " + round);
			} // Of if
			tempAnn.train();
		} // Of for n

		double tempAccuray = tempAnn.test();
		System.out.println("The label scarcity array is: "
				+ Arrays.toString(dataset.computeLabelScarcityArray()));
		System.out.println("The label uncertainty matrix is: "
				+ Arrays.deepToString(tempAnn.computeLabelUncertaintyMatrix()));

		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("The training accuracy is: " + dataset.computeTrainingAccuracy());
		System.out.println("The total cost is: " + dataset.computeTotalCost());
		System.out.println(dataset.getCostDetail() + "\r\n");
	}// Of randomSelectionLearn

	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
	public String toString() {
		return dataset.toString();
	}// Of toString

	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
	public static void readDataTest() {
		// Cmale tempCmale = new Cmale("D:/data/simpleiris.arff", 4, 1);
		Cmale tempCmale = new Cmale("data/mliris.arff", 4, 3);
		System.out.println("The data is:\r\n" + tempCmale);
	}// Of readDataTest

	/**
	 ********************** 
	 * Test on the iris dataset.
	 ********************** 
	 */
	public static void irisTest() {
		Cmale tempCmale = new Cmale("data/mliris.arff", 4, 3);
		int[] tempFullConnectLayerNodes = { 4, 8, 8 };
		int[] tempParallelLayerNodes = { 4, 2 };
		try {
			tempCmale.learn(30, 10, 1, 0.12, tempFullConnectLayerNodes, tempParallelLayerNodes,
					15000);
		} catch (Exception ee) {
			System.out.println(ee);
			System.exit(0);
		} // Of try

		tempCmale.randomSelectionLearn(40, tempFullConnectLayerNodes, tempParallelLayerNodes,
				15000);
	}// Of irisTest

	/**
	 ********************** 
	 * Test on the iris dataset.
	 ********************** 
	 */
	public static void flagTest() {
		Cmale tempCmale = new Cmale("data/flags.arff", 14, 12);
		int[] tempFullConnectLayerNodes = { 14, 14, 14 };
		int[] tempParallelLayerNodes = { 8, 2 };
		try {
			tempCmale.learn(150, 150, 1, 0.12, tempFullConnectLayerNodes, tempParallelLayerNodes,
					15000);
		} catch (Exception ee) {
			System.out.println(ee);
			System.exit(0);
		} // Of try
		tempCmale.randomSelectionLearn(300, tempFullConnectLayerNodes, tempParallelLayerNodes,
				15000);
	}// Of irisTest

	/**
	 ********************** 
	 * The entrance.
	 ********************** 
	 */
	public static void main(String[] args) {
		// readDataTest();
		irisTest();
		// flagTest();
		System.out.println("Finish.");
	}// Of main

}// Of class Cmale
