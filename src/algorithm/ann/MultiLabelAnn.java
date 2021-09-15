package algorithm.ann;

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
			int paraNumParts, int[] paraParallelLayerNumNodes, double paraLearningRate,
			double paraMobp, String paraActivators) {
		dataset = paraDataset;

		// Step 2. Accept parameters.
		numLayers = paraFullConnectLayerNumNodes.length + paraParallelLayerNumNodes.length;
		// Adjust if necessary.
		learningRate = paraLearningRate;
		mobp = paraMobp;

		// Initialize layers.
		layers = new GeneralAnnLayer[numLayers - 1];
		for (int i = 0; i < paraFullConnectLayerNumNodes.length; i++) {
			layers[i] = new FullConnectAnnLayer(paraFullConnectLayerNumNodes[i],
					paraFullConnectLayerNumNodes[i + 1], paraActivators.charAt(i), paraLearningRate,
					paraMobp);
		} // Of for i

		for (int i = 0; i < paraParallelLayerNumNodes.length; i++) {
			layers[paraFullConnectLayerNumNodes.length + i] = new ParallelAnnLayer(paraNumParts,
					paraParallelLayerNumNodes[i], paraParallelLayerNumNodes[i + 1],
					paraActivators.charAt(i), paraLearningRate, paraMobp);
		} // Of for i
	}// Of the first constructor

	/**
	 ********************
	 * Train using the dataset.
	 ********************
	 */
	public void train() {
		double[] tempInput = new double[dataset.getNumConditions()];
		double[] tempTarget = new double[dataset.getNumLabels() * 2];
		for (int i = 0; i < dataset.getNumInstances(); i++) {
			// Fill the data.
			for (int j = 0; j < tempInput.length; j++) {
				tempInput[j] = dataset.getData(i, j);
			} // Of for j

			// Fill the class label.
			// Arrays.fill(tempTarget, 0);
			for (int j = 0; j < dataset.getNumLabels(); j++) {
				if (dataset.getLabel(i, j) == 0) {
					tempTarget[dataset.getLabel(i, j * 2)] = 1;
					tempTarget[dataset.getLabel(i, j * 2 + 1)] = 0;
				} else {
					tempTarget[dataset.getLabel(i, j * 2)] = 0;
					tempTarget[dataset.getLabel(i, j * 2 + 1)] = 1;
				} // Of if
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
		int tempPredictedClass = -1;

		for (int i = 0; i < dataset.getNumInstances(); i++) {
			tempInput = dataset.getData(i);

			tempPrediction = forward(tempInput);
			// System.out.println("prediction: " +
			// Arrays.toString(tempPrediction));
			// tempPredictedClass = argmax(tempPrediction);

			for (int j = 0; j < dataset.getNumLabels(); j++) {
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
		for (int i = 0; i < numLayers - 1; i++) {
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
		double[] tempErrors = layers[numLayers - 2].getLastLayerErrors(paraTarget);
		for (int i = numLayers - 2; i >= 0; i--) {
			tempErrors = layers[i].backPropagation(tempErrors);
		} // Of for i

		return;
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
		int[] tempFullConnectLayerNodes = { 4, 8, 8, 12 };
		int[] tempParallelLayerNodes = { 3, 3, 2 };
		MultiLabelAnn tempNetwork = new MultiLabelAnn(tempDataset, tempFullConnectLayerNodes, 3,
				tempParallelLayerNodes, 0.01, 0.6, "sss");

		for (int round = 0; round < 5000; round++) {
			tempNetwork.train();
		} // Of for n

		double tempAccuray = tempNetwork.test();
		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("FullAnn ends.");
	}// Of main
}// Of class FullAnn
