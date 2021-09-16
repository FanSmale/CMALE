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
		System.out.println("numLayers = " + numLayers);
		for (int i = 0; i < paraFullConnectLayerNumNodes.length - 1; i++) {
			// System.out.println("Building full connect layer " + i);
			layers[i] = new FullConnectAnnLayer(paraFullConnectLayerNumNodes[i],
					paraFullConnectLayerNumNodes[i + 1], paraActivators.charAt(i), paraLearningRate,
					paraMobp);
		} // Of for i

		// System.out.println(
		// "Building ParallelAnnLayer " + (paraFullConnectLayerNumNodes.length -
		// 1));
		layers[paraFullConnectLayerNumNodes.length - 1] = new ParallelAnnLayer(tempNumParts,
				paraFullConnectLayerNumNodes[paraFullConnectLayerNumNodes.length - 1]
						/ tempNumParts,
				paraParallelLayerNumNodes[0],
				paraActivators.charAt(paraFullConnectLayerNumNodes.length), paraLearningRate,
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
		double[] tempInput = new double[dataset.getNumConditions()];
		double[] tempTarget = new double[dataset.getNumLabels()];
		for (int i = 0; i < dataset.getNumInstances(); i++) {
			// Fill the data.
			for (int j = 0; j < tempInput.length; j++) {
				tempInput[j] = dataset.getData(i, j);
			} // Of for j

			// Fill the class label.
			for (int j = 0; j < dataset.getNumLabels(); j++) {
				tempTarget[j] = dataset.getLabel(i, j);
			} // Of for j

			// Train with this instance.
			forward(tempInput);
			backPropagation(tempTarget, dataset.getLabelKnown(i));
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

		double tempNumCorrect = 0;
		double[] tempPrediction;

		for (int i = 0; i < dataset.getNumInstances(); i++) {
			tempInput = dataset.getData(i);

			tempPrediction = forward(tempInput);
			//System.out.println("tempInput = " + Arrays.toString(tempInput) + "\r\n prediction = "
			//		+ Arrays.toString(tempPrediction));
			// tempPredictedClass = argmax(tempPrediction);

			for (int j = 0; j < dataset.getNumLabels(); j++) {
				// System.out.println("i = " + i + ", j = " + j);
				if ((int) (tempPrediction[j] + 0.5) == dataset.getLabel(i, j)) {
					tempNumCorrect++;
				} // Of if
			} // Of for j
		} // Of for i

		System.out.println("Correct: " + tempNumCorrect + " out of "
				+ (dataset.getNumInstances() * dataset.getNumLabels()));

		return tempNumCorrect / dataset.getNumInstances() / dataset.getNumLabels();
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
			System.out.println("layer = " + i + ", resultArray = " + Arrays.toString(resultArray));
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
	public void backPropagation(double[] paraTarget, boolean[] paraLabelKnownArray) {
		System.out.println("backPropagation paraTarget = " + Arrays.toString(paraTarget));
		double[] tempErrors = layers[layers.length - 1].getLastLayerErrors(paraTarget);
		System.out.println("original error = " + Arrays.toString(tempErrors));
		for (int i = layers.length - 1; i >= 0; i--) {
			tempErrors = layers[i].backPropagation(tempErrors);
			System.out.println("layer  " + i + ", error = " + Arrays.toString(tempErrors));
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
		MultiLabelData tempDataset = new MultiLabelData("D:/data/multilabel/flags.arff", 14, 12);
		int[] tempFullConnectLayerNodes = { 14, 14, 24 };
		int[] tempParallelLayerNodes = { 2, 1 };
		MultiLabelAnn tempNetwork = new MultiLabelAnn(tempDataset, tempFullConnectLayerNodes,
				tempParallelLayerNodes, 0.01, 0.1, "ssssss");

		for (int round = 0; round < 10; round++) {
			if (round % 1000 == 999) {
				System.out.println("Round: " + round);
			} // Of if
			tempNetwork.train();
		} // Of for n

		double tempAccuray = 0;
		//tempAccuray = tempNetwork.test();
		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("FullAnn ends.");
	}// Of main
}// Of class FullAnn
