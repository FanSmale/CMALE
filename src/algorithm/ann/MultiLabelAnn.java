package algorithm.ann;

import java.util.Arrays;

import data.MultiLabelData;
import util.SimpleTools;

/**
 * Full ANN with a number of layers.
 * 
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 */
public class MultiLabelAnn {

	/**
	 * The whole dataset.
	 */
	MultiLabelData dataset;

	/**
	 * Number of layers. It is counted according to nodes instead of edges.
	 */
	int numLayers;

	/**
	 * Momentum coefficient.
	 */
	public double mobp;

	/**
	 * Learning rate.
	 */
	public double learningRate;

	/**
	 * The layers.
	 */
	GeneralAnnLayer[] layers;

	/**
	 ********************
	 * The first constructor.
	 * 
	 * @param paraFilename
	 *            The arff filename.
	 * @param paraLayerNumNodes
	 *            The number of nodes for each layer (may be different).
	 * @param paraLearningRate
	 *            Learning rate.
	 * @param paraMobp
	 *            Momentum coefficient.
	 * @param paraActivators
	 *            The storing the activators of each layer.
	 ********************
	 */
	public MultiLabelAnn(MultiLabelData paraDataset, int[] paraFullConnectLayerNumNodes,
			int[] paraParallelLayerNumNodes, double paraLearningRate, double paraMobp,
			String paraActivators) {
		dataset = paraDataset;
		int tempNumParts = paraDataset.getNumLabels();

		// Step 2. Accept parameters.
		numLayers = paraFullConnectLayerNumNodes.length + paraParallelLayerNumNodes.length;
		// Adjust if necessary.
		learningRate = paraLearningRate;
		mobp = paraMobp;

		// Initialize layers.
		layers = new GeneralAnnLayer[numLayers - 1];
		// System.out.println("numLayers = " + numLayers);
		for (int i = 0; i < paraFullConnectLayerNumNodes.length - 1; i++) {
			// System.out.println("Building full connect layer " + i);
			layers[i] = new FullConnectAnnLayer(paraFullConnectLayerNumNodes[i],
					paraFullConnectLayerNumNodes[i + 1], paraActivators.charAt(i), paraLearningRate,
					paraMobp);
		} // Of for i

		// System.out.println(
		// "Building ParallelAnnLayer " + (paraFullConnectLayerNumNodes.length -
		// 1));
		layers[paraFullConnectLayerNumNodes.length - 1] = new FullConnectAnnLayer(
				paraFullConnectLayerNumNodes[paraFullConnectLayerNumNodes.length - 1],
				paraParallelLayerNumNodes[0] * tempNumParts,
				paraActivators.charAt(paraFullConnectLayerNumNodes.length - 1), paraLearningRate,
				paraMobp);

		for (int i = 0; i < paraParallelLayerNumNodes.length - 1; i++) {
			// System.out.println(
			// "Building ParallelAnnLayer " +
			// (paraFullConnectLayerNumNodes.length + i));
			layers[paraFullConnectLayerNumNodes.length + i] = new ParallelAnnLayer(tempNumParts,
					paraParallelLayerNumNodes[i], paraParallelLayerNumNodes[i + 1],
					paraActivators.charAt(paraFullConnectLayerNumNodes.length + i),
					paraLearningRate, paraMobp);
		} // Of for i
	}// Of the first constructor

	/**
	 ********************
	 * Train using the dataset.
	 ********************
	 */
	public void train() {
		double[] tempInput;
		int[] tempTarget;
		int tempInstance;
		for (int i = 0; i < dataset.getNumQueriedInstances(); i++) {
			tempInstance = dataset.getQueriedInstanceIndex(i);
			// Step 1. Fill the data.
			tempInput = dataset.getData(tempInstance);

			// Step 3. Fill the class label. Unknown labels are INVALID_VALUE.
			tempTarget = dataset.getQueriedLabel(tempInstance);

			// Step 4. Train with this instance.
			forward(tempInput);
			backPropagation(tempTarget);
		} // Of for i
	}// Of train

	/**
	 ********************
	 * Train emphasizing on some instances. For example, if there are 100
	 * queried instances, we want to emphasize on 3 instances with 20 times,
	 * then we need to re-train these 3 instances for every 5 normal instances.
	 * This method is useful for incremental learning, where newly labelled
	 * instances should be emphasized.
	 * 
	 * @param paraTimes
	 *            The additional training times of the emphasized instances.
	 * @param paraEmphasizedInstances
	 *            Emphasized instances, the original indices.
	 ********************
	 */
	public void emphasizedTrain(int paraTimes, int[] paraEmphasizedInstances) {
		double[] tempInput;
		int[] tempTarget;
		int tempInstance;
		int tempNumQueriedInstance = dataset.getNumQueriedInstances();
		for (int i = 0; i < tempNumQueriedInstance; i++) {
			tempInstance = dataset.getQueriedInstanceIndex(i);
			// Step 1. Fill the data.
			tempInput = dataset.getData(tempInstance);

			// Step 3. Fill the class label. Unknown labels are INVALID_VALUE.
			tempTarget = dataset.getQueriedLabel(tempInstance);

			// Step 4. Train with this instance.
			forward(tempInput);
			backPropagation(tempTarget);

			// Step 5. Judge emphasized train or not.
			if (tempNumQueriedInstance % paraTimes != 0) {
				continue;
			} // Of if

			// Step 6. Train emphasized instances.
			for (int j = 0; j < paraEmphasizedInstances.length; j++) {
				// Step 6.1 Fill the data.
				tempInstance = paraEmphasizedInstances[j];
				tempInput = dataset.getData(tempInstance);

				// Step 6.2 Fill the class label. Unknown labels are
				// INVALID_VALUE.
				tempTarget = dataset.getQueriedLabel(tempInstance);

				// Step 6.3 Train with this instance.
				forward(tempInput);
				backPropagation(tempTarget);
			} // Of for j
		} // Of for i
	}// Of emphasizedTrain

	/**
	 ********************
	 * Test using the dataset.
	 * 
	 * @return The precision.
	 ********************
	 */
	public double test() {
		double[] tempInput;

		double[] tempPredictions;

		for (int i = 0; i < dataset.getNumInstances(); i++) {
			tempInput = dataset.getData(i);
			tempPredictions = forward(tempInput);

			for (int j = 0; j < dataset.getNumLabels(); j++) {
				if (tempPredictions[2 * j] > tempPredictions[2 * j + 1]) {
					dataset.setPredictedLabel(i, j, 0);
				} else {
					dataset.setPredictedLabel(i, j, 1);
				} // Of if
			} // Of for j
		} // Of for i

		return dataset.computeAccuracy();
	}// Of test

	/**
	 ********************
	 * Compute label uncertainty matrix.
	 * 
	 * @return The matrix.
	 ********************
	 */
	public double[][] computeLabelUncertaintyMatrix() {
		double[][] resultMatrix = new double[dataset.getNumInstances()][dataset.getNumLabels()];
		double[] tempInput;
		double[] tempPredictions;

		for (int i = 0; i < dataset.getNumInstances(); i++) {
			tempInput = dataset.getData(i);
			tempPredictions = forward(tempInput);

			for (int j = 0; j < dataset.getNumLabels(); j++) {
				// Queried label.
				if (dataset.getQueriedLabel(i, j) != MultiLabelData.INVALID_LABEL) {
					resultMatrix[i][j] = 0;
					continue;
				} // Of if

				if (tempPredictions[2 * j] > tempPredictions[2 * j + 1]) {
					resultMatrix[i][j] = 1 - tempPredictions[2 * j] + tempPredictions[2 * j + 1];
				} else {
					resultMatrix[i][j] = 1 + tempPredictions[2 * j] - tempPredictions[2 * j + 1];
				} // Of if
			} // Of for j
		} // Of for i

		return resultMatrix;
	}// Of computeLabelUncertaintyMatrix

	/**
	 ********************
	 * Get the most uncertain labels of an instance. Note that the result is a
	 * little complex since Java only supports one return value.
	 * 
	 * @return The instance index and the label indices in one array.
	 ********************
	 */
	public int[] getMostUncertainLabelIndices(int paraNumLabels) {
		int[] resultArray = new int[1 + paraNumLabels];
		double[][] tempMatrix = computeLabelUncertaintyMatrix();

		double tempMax = -1;
		double tempTotal;
		int[] tempSortedIndices;
		for (int i = 0; i < tempMatrix.length; i++) {
			tempSortedIndices = SimpleTools.mergeSortToIndices(tempMatrix[i]);
			if (dataset.getLabelQueried(i, tempSortedIndices[paraNumLabels - 1])) {
				// No enough unknown labels to query.
				continue;
			} // Of if

			tempTotal = 0;
			for (int j = 0; j < paraNumLabels; j++) {
				tempTotal += tempMatrix[i][tempSortedIndices[j]];
			} // Of for j

			if (tempMax < tempTotal) {
				tempMax = tempTotal;

				resultArray[0] = i;
				for (int j = 0; j < paraNumLabels; j++) {
					resultArray[j + 1] = tempSortedIndices[j];
				} // Of for j
			} // Of if
		} // Of for i

		System.out.println("Most uncertain: " + Arrays.toString(resultArray) + ": " + tempMax);
		return resultArray;
	}// Of getMostUncertainLabelIndices

	/**
	 ********************
	 * Compute a batch of uncertain instances and respective labels.
	 * 
	 * @param paraInstanceBatch
	 *            How many instances should be selected.
	 * @param paraLabelBatch
	 *            How many labels should be selected for each instance.
	 * @return Instance indices and label indices in one matrix, where the first
	 *         column is for the instance indices.
	 ********************
	 */
	public int[][] getUncertainLabelBatch(int paraInstanceBatch, int paraLabelBatch) {
		int[][] resultMatrix = new int[paraInstanceBatch][1 + paraLabelBatch];

		// Step 1. Get the uncertainty for all instance-label pairs.
		double[][] tempUncertaintyMatrix = computeLabelUncertaintyMatrix();

		// Step 2. Get the uncertainty sum for each instance according to the
		// label batch.
		double[] tempInstanceUncertainArray = new double[dataset.getNumInstances()];
		int[] tempSortedLabelIndices;
		for (int i = 0; i < tempInstanceUncertainArray.length; i++) {
			tempSortedLabelIndices = SimpleTools.mergeSortToIndices(tempUncertaintyMatrix[i]);
			if (dataset.getLabelQueried(i, tempSortedLabelIndices[paraLabelBatch - 1])) {
				// No enough unknown labels to query.
				tempInstanceUncertainArray[i] = 0;
			} else {
				for (int j = 0; j < paraLabelBatch; j++) {
					tempInstanceUncertainArray[i] += tempUncertaintyMatrix[i][tempSortedLabelIndices[j]];
				} // Of for j
			} // Of if
		} // Of for i

		// Step 3. Sort the instance uncertainty.
		int[] tempSortedInstanceIndices = SimpleTools
				.mergeSortToIndices(tempInstanceUncertainArray);

		// Step 4. Copy data.
		for (int i = 0; i < resultMatrix.length; i++) {
			resultMatrix[i][0] = tempSortedInstanceIndices[i];

			tempSortedLabelIndices = SimpleTools
					.mergeSortToIndices(tempUncertaintyMatrix[tempSortedInstanceIndices[i]]);
			for (int j = 0; j < paraLabelBatch; j++) {
				resultMatrix[i][j + 1] = tempSortedLabelIndices[j];
			}//Of for j
		} // Of for i

		System.out.print("Most uncertain: " + Arrays.deepToString(resultMatrix));
		for (int i = 0; i < resultMatrix.length; i++) {
			for (int j = 1; j < resultMatrix[0].length; j++) {
				System.out.print(", " + tempUncertaintyMatrix[resultMatrix[i][0]][resultMatrix[i][j]]);
			}//Of for j
		}//Of for i
		System.out.println();
		
		return resultMatrix;
	}// Of getUncertainLabelBatch

	/**
	 ********************
	 * Forward prediction. This is just a stub and should be overwritten in the
	 * subclass.
	 * 
	 * @param paraInput
	 *            The input data of one instance.
	 * @return The data at the output end.
	 ********************
	 */
	public double[] forward(double[] paraInput) {
		double[] resultArray = paraInput;
		for (int i = 0; i < layers.length; i++) {
			resultArray = layers[i].forward(resultArray);
		} // Of for i
		return resultArray;
	}// Of forward

	/**
	 ********************
	 * Back propagation. This is just a stub and should be overwritten in the
	 * subclass.
	 * 
	 * @param paraTarget
	 *            For 3-class data, it is [0, 0, 1], [0, 1, 0] or [1, 0, 0].
	 ********************
	 */
	public void backPropagation(int[] paraTarget) {
		// Pre-processing.
		int[] tempTarget = new int[paraTarget.length * 2];
		for (int i = 0; i < paraTarget.length; i++) {
			if (paraTarget[i] == 0) {
				tempTarget[2 * i] = 1;
				tempTarget[2 * i + 1] = 0;
			} else if (paraTarget[i] == 1) {
				tempTarget[2 * i] = 0;
				tempTarget[2 * i + 1] = 1;
			} else {
				tempTarget[2 * i] = MultiLabelData.INVALID_LABEL;
				tempTarget[2 * i + 1] = MultiLabelData.INVALID_LABEL;
			} // Of if
		} // Of for i

		double[] tempErrors = layers[layers.length - 1].getLastLayerErrors(tempTarget);
		for (int i = layers.length - 1; i >= 0; i--) {
			tempErrors = layers[i].backPropagation(tempErrors);
		} // Of for i
	}// Of backPropagation

	/**
	 ********************
	 * Show me.
	 ********************
	 */
	public String toString() {
		String resultString = "I am a full ANN with " + numLayers + " layers";
		return resultString;
	}// Of toString

	/**
	 ********************
	 * Test the algorithm.
	 ********************
	 */
	public static void main(String[] args) {
		// Iris with one label and binary class
		// MultiLabelData tempDataset = new
		// MultiLabelData("data/binaryiris.arff", 4, 1);
		// int[] tempFullConnectLayerNodes = { 4, 8, 8 };
		// int[] tempParallelLayerNodes = { 2, 2 };

		// Iris with three labels
		MultiLabelData tempDataset = new MultiLabelData("data/mliris.arff", 4, 3);
		tempDataset.randomQuery(120);
		System.out.println("Number of queries: " + tempDataset.getNumQueriedLabels());
		int[] tempFullConnectLayerNodes = { 4, 8, 8 };
		int[] tempParallelLayerNodes = { 4, 2 };

		// Flag with multi-label
		// MultiLabelData tempDataset = new MultiLabelData("data/flags.arff",
		// 14, 12);
		// tempDataset.randomQuery(0.8);
		// int[] tempFullConnectLayerNodes = { 14, 14, 14 };
		// int[] tempParallelLayerNodes = { 8, 2 };

		MultiLabelAnn tempAnn = new MultiLabelAnn(tempDataset, tempFullConnectLayerNodes,
				tempParallelLayerNodes, 0.05, 0.6, "sssss");

		for (int round = 0; round < 10000; round++) {
			if (round % 1000 == 999) {
				System.out.println("Round: " + round);
			} // Of if
			tempAnn.train();
		} // Of for n

		double tempAccuray = tempAnn.test();
		System.out.println("The label scarcity array is: "
				+ Arrays.toString(tempDataset.computeLabelScarcityArray()));
		System.out.println("The label uncertainty matrix is: "
				+ Arrays.deepToString(tempAnn.computeLabelUncertaintyMatrix()));

		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("The training accuracy is: " + tempDataset.computeTrainingAccuracy());
		System.out.println("The total cost is: " + tempDataset.computeTotalCost());

		System.out.println("FullAnn ends.");
	}// Of main
}// Of class FullAnn
