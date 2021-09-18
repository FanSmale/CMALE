package algorithm.ann;

import java.util.Arrays;
import java.util.Random;

import data.MultiLabelData;

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
	 * The number of nodes for each layer, e.g., [3, 4, 6, 2] means that there
	 * are 3 input nodes (conditional attributes), 2 hidden layers with 4 and 6
	 * nodes, respectively, and 2 class values (binary classification). int[]
	 * layerNumNodes;
	 */

	/**
	 * Momentum coefficient.
	 */
	public double mobp;

	/**
	 * Learning rate.
	 */
	public double learningRate;

	/**
	 * For random number generation.
	 */
	Random random = new Random();

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
		// System.out.println("numLayers = " + numLayers);
		for (int i = 0; i < layers.length; i++) {
			// System.out.println("layer = " + i + ", resultArray = " +
			// Arrays.toString(resultArray));
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
		// System.out.println("backPropagation paraTarget = " +
		// Arrays.toString(paraTarget));
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
		// System.out.println("paraLabelKnownArray error = " +
		// Arrays.toString(paraLabelKnownArray));
		// System.out.println("original error = " +
		// Arrays.toString(tempErrors));
		for (int i = layers.length - 1; i >= 0; i--) {
			tempErrors = layers[i].backPropagation(tempErrors);

			// System.out.println("layer " + i + ", error = " +
			// Arrays.toString(tempErrors));
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
		tempDataset.randomQuery(0.3);
		System.out.println("Number of queries: " + tempDataset.getNumQueriedLabels());
		int[] tempFullConnectLayerNodes = { 4, 8, 8 };
		int[] tempParallelLayerNodes = { 4, 2 };

		// Flag with multi-label
		// MultiLabelData tempDataset = new MultiLabelData("data/flags.arff",
		// 14, 12);
		// tempDataset.randomQuery(0.8);
		// int[] tempFullConnectLayerNodes = { 14, 14, 14 };
		// int[] tempParallelLayerNodes = { 8, 2 };

		MultiLabelAnn tempNetwork = new MultiLabelAnn(tempDataset, tempFullConnectLayerNodes,
				tempParallelLayerNodes, 0.02, 0.6, "sssss");

		for (int round = 0; round < 20000; round++) {
			if (round % 1000 == 999) {
				System.out.println("Round: " + round);
			} // Of if
			tempNetwork.train();
		} // Of for n

		double tempAccuray = 0;
		tempAccuray = tempNetwork.test();
		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("The total cost is: " + tempDataset.computeTotalCost());
		
		System.out.println("FullAnn ends.");
	}// Of main
}// Of class FullAnn
