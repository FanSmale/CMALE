package algorithm;

import java.io.File;
import java.io.IOException;
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
	 * The output file for tracking the learning process.
	 */
	RandomAccessFile outputFile;

	/**
	 * The neural network for classification.
	 */
	MultiLabelAnn multiLabelAnn;

	/**
	 * Representativeness of each instance.
	 */
	double[] representativenessArray;

	/**
	 * Representativeness of rank each instance.
	 */
	int[] representativenessRankArray;

	/**
	 ********************** 
	 * The first constructor. Data and labels are stored in one file.
	 * 
	 * @param paraDataFilename
	 *            The data filename.
	 * @param paraNumConditions
	 *            The number of conditional attributes.
	 * @param paraNumLabels
	 *            The number of labels.
	 ********************** 
	 */
	public Cmale(String paraArffFilename, int paraNumConditions, int paraNumLabels) {
		// Step 1. Read the data from file
		dataset = new MultiLabelData(paraArffFilename, paraNumConditions, paraNumLabels);
		// Basic information of the data.
		numInstances = dataset.getNumInstances();
		numConditions = dataset.getNumConditions();
		numLabels = paraNumLabels;

		// Step 2. Compute instance representativeness.
		representativenessArray = new double[numInstances];
		representativenessRankArray = new int[numInstances];

		// Step 2. Prepare output file.
		File tempFile = new File("data/learn-results.txt");
		if (tempFile.exists()) {
			tempFile.delete();
		} // Of if
		try {
			outputFile = new RandomAccessFile("data/learn-results.txt", "rw");
		} catch (Exception ee) {
			System.out.println("Error occurred in Cmale constructor: " + ee);
			System.exit(0);
		} // Of try
	}// Of the first constructor

	/**
	 ********************** 
	 * Initialize ANN.
	 * 
	 * @param paraFullConnectLayerNodes
	 *            Full connect layer nodes, e.g., {4, 8, 6}. The first one
	 *            should be equal to numConditions.
	 * @param paraParallelLayerNodes
	 *            Parallel connect layer nodes, e.g., {4, 2}. The last one
	 *            should always be 2.
	 * @param paraLearningRate
	 *            The learning rate, such as 0.02.
	 * @param paraMobp
	 *            Mobp.
	 * @param paraActivators,
	 *            e.g., "sssss".
	 ********************** 
	 */
	public void initializeMultiLabelAnn(int[] paraFullConnectLayerNodes,
			int[] paraParallelLayerNodes, double paraLearningRate, double paraMobp,
			String paraActivators) {
		multiLabelAnn = new MultiLabelAnn(dataset, paraFullConnectLayerNodes,
				paraParallelLayerNodes, paraLearningRate, paraMobp, paraActivators);
	}// Of initializeMultiLabelAnn

	/**
	 ********************** 
	 * Bounded train. Control the training rounds.
	 * 
	 * @param paraLowerRounds
	 *            The training round lower bound.
	 * @param paraLowerRounds
	 *            The training round upper bound.
	 * @param paraCheckingRounds
	 *            For every some bounds we need to check the convergence of the
	 *            classifier.
	 * @param paraAccuracyThreshold
	 *            When the training accuracy threshold is reached, the training
	 *            process terminates.
	 ********************** 
	 */
	public void boundedTrain(int paraLowerRounds, int paraUpperRounds, int paraCheckingRounds,
			double paraAccuracyThreshold) throws IOException {
		System.out.printf("boundedTrain(%d, %d, %d, %f)\r\n", paraLowerRounds, paraUpperRounds,
				paraCheckingRounds, paraAccuracyThreshold);
		// Step 1. Train according to the lower bound.
		int round = 0;
		for (; round < paraLowerRounds; round++) {
			if (round % 1000 == 999) {
				System.out.println("Round: " + round);
			} // Of if
			multiLabelAnn.train();
		} // Of for round

		// Step 2. Train and check.
		double tempTrainingAccuracy;
		for (; round < paraUpperRounds; round++) {
			if (round % paraCheckingRounds == paraCheckingRounds - 1) {
				multiLabelAnn.test();
				tempTrainingAccuracy = dataset.computeTrainingAccuracy();
				System.out.printf("Regular round: %d, training accuracy = %f \r\n", round,
						tempTrainingAccuracy);
				outputFile.writeBytes("Regular round: " + round + ", training accuracy = "
						+ tempTrainingAccuracy + ".\r\n");

				if (tempTrainingAccuracy > paraAccuracyThreshold) {
					break;
				} // Of if
			} // Of if
			multiLabelAnn.train();
		} // Of for n
	}// Of boundedTrain

	/**
	 ********************** 
	 * Bounded train emphasizing on some instances.
	 * 
	 * @param paraLowerRounds
	 *            The training round lower bound.
	 * @param paraLowerRounds
	 *            The training round upper bound.
	 * @param paraCheckingRounds
	 *            For every some bounds we need to check the convergence of the
	 *            classifier.
	 * @param paraEmphasizeTimes
	 *            How many times the new instances should be emphasized.
	 * @param paraInstanceIndices
	 *            Which instances are emphasized.
	 * @param paraAccuracyThreshold
	 *            When the training accuracy threshold is reached, the training
	 *            process terminates.
	 ********************** 
	 */
	public void boundedEmphasizedTrain(int paraUpperRounds, int paraCheckingRounds,
			int paraEmphasizeTimes, int[] paraInstanceIndices, double paraAccuracyThreshold)
			throws IOException {
		// Step 1. Compute the original training accuracy.
		double tempTrainingAccuracy = dataset.computeTrainingAccuracy();
		System.out.printf("Emphasized train. Before retrain, training accuracy = %f \r\n",
				tempTrainingAccuracy);
		outputFile.writeBytes("Emphasized train. Before retrain, training accuracy = "
				+ tempTrainingAccuracy + "\r\n");

		// Step 2. Train and check.
		for (int round = 0; round < paraUpperRounds; round++) {
			if (round % paraCheckingRounds == paraCheckingRounds - 1) {
				multiLabelAnn.test();
				tempTrainingAccuracy = dataset.computeTrainingAccuracy();
				System.out.printf("Regular round: %d, training accuracy = %f \r\n", round,
						tempTrainingAccuracy);
				outputFile.writeBytes("Regular round: " + round + ", training accuracy = "
						+ tempTrainingAccuracy + "\r\n");
				if (tempTrainingAccuracy > paraAccuracyThreshold) {
					break;
				} // Of if
			} // Of if
			multiLabelAnn.emphasizedTrain(paraEmphasizeTimes, paraInstanceIndices);
		} // Of for n
	}// Of boundedEmphasizedTrain

	/**
	 ********************** 
	 * Compute instance representativeness.
	 * 
	 * @param paraDc
	 *            The dc ratio.
	 ********************** 
	 */
	public void computeInstanceRepresentativeness(double paraDc) {
		// Step 1. Calculate density using Gaussian kernel.
		double[] tempDensityArray = new double[numInstances];
		double tempDistance = 0;
		for (int i = 0; i < numInstances; i++) {
			tempDensityArray[i] = 0;
			for (int j = 0; j < numInstances; j++) {
				tempDistance = dataset.distance(i, j);
				tempDensityArray[i] += Math.exp(-tempDistance * tempDistance / paraDc / paraDc);
			} // Of for j
		} // Of for i

		// Step 2. Calculate distance to its master.
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

		// Step 3. Representativeness.
		for (int i = 0; i < numInstances; i++) {
			representativenessArray[i] = tempDensityArray[i] * tempDistanceToMasterArray[i];
		} // Of for i

		// Step 4. Sort instances according to representativeness.
		representativenessRankArray = SimpleTools.mergeSortToIndices(representativenessArray);
	}// Of computeInstanceRepresentativeness

	/**
	 ********************** 
	 * Learn the classifier
	 * 
	 * @param paraColdStartRounds
	 *            Cold start rounds not considering label uncertainty.
	 * @param paraNumAdditionalQueries
	 *            Additional queries.
	 * @param paraNumQueryBatchSize
	 *            How many labels are queried each time.
	 * @param paraDc
	 *            For representativeness computation.
	 * 
	 * @param paraI
	 *            The second index.
	 ********************** 
	 */
	public void twoStageLearn(int paraColdStartRounds, int paraNumAdditionalQueries,
			int paraNumQueryBatchSize, double paraDc, int paraPretrainRounds,
			double paraAccuracyThreshold) throws IOException {
		// Step 1. Reset the dataset to clear learning information.
		dataset.reset();

		// Step 2. Calculate the representativeness of each instance.
		computeInstanceRepresentativeness(paraDc);

		// Step 3. Cold start stage. Only consider instance representativeness
		// and label scarcity/diversity
		int[] tempLabelIndices = new int[paraNumQueryBatchSize];

		outputFile.writeBytes("Here is the whole process of learn(): \r\n");
		outputFile.writeBytes("Cold start: \r\n");
		// Query the scare k labels of most representative p instances.
		for (int i = 0; i < paraColdStartRounds; i++) {
			tempLabelIndices = dataset.getScareLabels(paraNumQueryBatchSize);
			dataset.queryLabels(representativenessRankArray[i], tempLabelIndices);
			outputFile.writeBytes("Query instance #" + representativenessRankArray[i]
					+ " with labels #" + Arrays.toString(tempLabelIndices) + "\r\n");
		} // Of for i

		// Pre-train an ANN. At least 1000 rounds.
		boundedTrain(1000, paraPretrainRounds, 200, paraAccuracyThreshold);

		// Step 4. Regular learning.
		// Now only one instance at a time.
		int[] tempInstanceIndices = new int[1];
		outputFile.writeBytes("Query and learning process: \r\n");
		int[] tempIndices;
		for (int q = 0; q < paraNumAdditionalQueries; q++) {
			tempIndices = multiLabelAnn.getMostUncertainLabelIndices(paraNumQueryBatchSize);
			for (int j = 0; j < tempLabelIndices.length; j++) {
				tempLabelIndices[j] = tempIndices[j + 1];
			} // Of for j
			dataset.queryLabels(tempIndices[0], tempLabelIndices);
			outputFile.writeBytes("Query instance #" + tempIndices[0] + " with labels #"
					+ Arrays.toString(tempLabelIndices) + "\r\n");

			tempInstanceIndices[0] = tempIndices[0];
			boundedEmphasizedTrain(5000, 200, 10, tempInstanceIndices, paraAccuracyThreshold);
		} // Of for q

		outputSummary();
	}// Of learn

	/**
	 ********************** 
	 * Learn with randomly selected labels.
	 * 
	 * @param paraNumQueriedLabels
	 *            The number of queried labels.
	 * @param paraTrainRounds
	 *            The number of train rounds, the upper bound.
	 * @param paraAccuracyThreshold
	 *            The training accuracy threshold. Terminate the learning
	 *            process if this threshold is reached.
	 * @param paraI
	 *            The second index.
	 ********************** 
	 */
	public void randomSelectionLearn(int paraNumQueriedLabels, int paraTrainRounds,
			double paraAccuracyThreshold) throws IOException {
		dataset.reset();

		// Step 1. randomly select labels to query.
		// Here I present a trick converting the matrix to an array to assure
		// the number of queries.
		dataset.randomQuery(paraNumQueriedLabels);

		// Step 2. Train an ANN.
		boundedTrain(1000, paraTrainRounds, 200, paraAccuracyThreshold);

		outputSummary();
	}// Of randomSelectionLearn

	/**
	 ********************** 
	 * Output the summary of the model.
	 ********************** 
	 */
	public void outputSummary() throws IOException {
		double tempAccuray = multiLabelAnn.test();
		System.out.println("The label scarcity array is: "
				+ Arrays.toString(dataset.computeLabelScarcityArray()));
		System.out.println("The label uncertainty matrix is: "
				+ Arrays.deepToString(multiLabelAnn.computeLabelUncertaintyMatrix()));

		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("The training accuracy is: " + dataset.computeTrainingAccuracy());
		System.out.println("The total cost is: " + dataset.computeTotalCost());
		System.out.println(dataset.getCostDetail() + "\r\n");

		outputFile.writeBytes("The label scarcity array is: "
				+ Arrays.toString(dataset.computeLabelScarcityArray()) + "\r\n");
		outputFile.writeBytes("The label uncertainty matrix is: "
				+ Arrays.deepToString(multiLabelAnn.computeLabelUncertaintyMatrix()) + "\r\n");
		outputFile.writeBytes("The accuracy is: " + tempAccuray + "\r\n");
		outputFile.writeBytes(
				"The training accuracy is: " + dataset.computeTrainingAccuracy() + "\r\n");
		outputFile.writeBytes("The total cost is: " + dataset.computeTotalCost() + "\r\n");
		outputFile.writeBytes(dataset.getCostDetail() + "\r\n");
	}// Of outputSummary

	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
	public void closeOutputFile() {
		try {
			outputFile.close();
		} catch (IOException ee) {
			System.out.println("Error occurred in Cmale.closeOutputFile(): " + ee);
			System.exit(0);
		} // Of try
	}// Of closeOutputFile

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
		int[] tempFullConnectLayerNodes = { 4, 8, };
		int[] tempParallelLayerNodes = { 4, 2 };
		try {
			tempCmale.initializeMultiLabelAnn(tempFullConnectLayerNodes, tempParallelLayerNodes,
					0.02, 0.6, "ssssss");
			tempCmale.twoStageLearn(30, 10, 1, 0.12, 20000, 0.99);

			tempCmale.initializeMultiLabelAnn(tempFullConnectLayerNodes, tempParallelLayerNodes,
					0.02, 0.6, "ssssss");
			tempCmale.randomSelectionLearn(40, 15000, 0.99);
		} catch (Exception ee) {
			System.out.println(ee);
			System.exit(0);
		} // Of try

		tempCmale.closeOutputFile();
	}// Of irisTest

	/**
	 ********************** 
	 * Test on the iris dataset.
	 ********************** 
	 */
	public static void flagTest() {
		Cmale tempCmale = new Cmale("data/flags.arff", 14, 12);
		int[] tempFullConnectLayerNodes = { 14, 14, 14 };
		int[] tempParallelLayerNodes = { 2 };

		try {
			tempCmale.initializeMultiLabelAnn(tempFullConnectLayerNodes, tempParallelLayerNodes,
					0.02, 0.6, "ssssss");
			tempCmale.twoStageLearn(150, 150, 2, 0.12, 15000, 0.99);

			tempCmale.initializeMultiLabelAnn(tempFullConnectLayerNodes, tempParallelLayerNodes,
					0.02, 0.6, "ssssss");
			tempCmale.randomSelectionLearn(300, 15000, 0.99);
		} catch (Exception ee) {
			System.out.println(ee);
			System.exit(0);
		} // Of try

		tempCmale.closeOutputFile();
	}// Of flagTest

	/**
	 ********************** 
	 * The entrance.
	 ********************** 
	 */
	public static void main(String[] args) {
		// readDataTest();
		// irisTest();
		flagTest();
		System.out.println("Finish.");
	}// Of main

}// Of class Cmale
