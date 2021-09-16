package algorithm.ann;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;

/**
 * Full connect ANN with a number of layers.
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class FullConnectAnn {
	/**
	 * The whole dataset.
	 */
	Instances dataset;

	/**
	 * Number of layers. It is counted according to nodes instead of edges.
	 */
	int numLayers;

	/**
	 * The number of nodes for each layer, e.g., [3, 4, 6, 2] means that there
	 * are 3 input nodes (conditional attributes), 2 hidden layers with 4 and 6
	 * nodes, respectively, and 2 class values (binary classification).
	 */
	int[] layerNumNodes;

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
	FullConnectAnnLayer[] layers;

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
	public FullConnectAnn(String paraFilename, int[] paraLayerNumNodes, double paraLearningRate,
			double paraMobp, String paraActivators) {
		// Step 1. Read data.
		try {
			FileReader tempReader = new FileReader(paraFilename);
			dataset = new Instances(tempReader);
			// The last attribute is the decision class.
			dataset.setClassIndex(dataset.numAttributes() - 1);
			tempReader.close();
		} catch (Exception ee) {
			System.out.println("Error occurred while trying to read \'" + paraFilename
					+ "\' in GeneralAnn constructor.\r\n" + ee);
			System.exit(0);
		} // Of try

		// Step 2. Accept parameters.
		layerNumNodes = paraLayerNumNodes;
		numLayers = layerNumNodes.length;
		// Adjust if necessary.
		layerNumNodes[0] = dataset.numAttributes() - 1;
		layerNumNodes[numLayers - 1] = dataset.numClasses();
		learningRate = paraLearningRate;
		mobp = paraMobp;

		// Initialize layers.
		layers = new FullConnectAnnLayer[numLayers - 1];
		for (int i = 0; i < layers.length; i++) {
			layers[i] = new FullConnectAnnLayer(layerNumNodes[i], layerNumNodes[i + 1],
					paraActivators.charAt(i), paraLearningRate, paraMobp);
		} // Of for i
	}// Of the first constructor

	/**
	 ********************
	 * Train using the dataset.
	 ********************
	 */
	public void train() {
		double[] tempInput = new double[dataset.numAttributes() - 1];
		int[] tempTarget = new int[dataset.numClasses()];
		for (int i = 0; i < dataset.numInstances(); i++) {
			// Fill the data.
			for (int j = 0; j < tempInput.length; j++) {
				tempInput[j] = dataset.instance(i).value(j);
			} // Of for j

			// Fill the class label.
			Arrays.fill(tempTarget, 0);
			tempTarget[(int) dataset.instance(i).classValue()] = 1;

			// Train with this instance.
			forward(tempInput);
			backPropagation(tempTarget);
		} // Of for i
	}// Of train

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
	 * 
	 ********************
	 */
	public void backPropagation(int[] paraTarget) {
		double[] tempErrors = layers[numLayers - 2].getLastLayerErrors(paraTarget);
		for (int i = numLayers - 2; i >= 0; i--) {
			tempErrors = layers[i].backPropagation(tempErrors);
		} // Of for i

		return;
	}// Of backPropagation

	/**
	 ********************
	 * Get the index corresponding to the max value of the array.
	 * 
	 * @return the index.
	 ********************
	 */
	public static int argmax(double[] paraArray) {
		int resultIndex = -1;
		double tempMax = -1e10;
		for (int i = 0; i < paraArray.length; i++) {
			if (tempMax < paraArray[i]) {
				tempMax = paraArray[i];
				resultIndex = i;
			} // Of if
		} // Of for i

		return resultIndex;
	}// Of argmax

	/**
	 ********************
	 * Test using the dataset.
	 * 
	 * @return The precision.
	 ********************
	 */
	public double test() {
		double[] tempInput = new double[dataset.numAttributes() - 1];

		double tempNumCorrect = 0;
		double[] tempPrediction;
		int tempPredictedClass = -1;

		for (int i = 0; i < dataset.numInstances(); i++) {
			// Fill the data.
			for (int j = 0; j < tempInput.length; j++) {
				tempInput[j] = dataset.instance(i).value(j);
			} // Of for j

			// Train with this instance.
			tempPrediction = forward(tempInput);
			// System.out.println("prediction: " +
			// Arrays.toString(tempPrediction));
			tempPredictedClass = argmax(tempPrediction);
			if (tempPredictedClass == (int) dataset.instance(i).classValue()) {
				tempNumCorrect++;
			} // Of if
		} // Of for i

		// System.out.println("Correct: " + tempNumCorrect + " out of " +
		// dataset.numInstances());
		return tempNumCorrect / dataset.numInstances();
	}// Of test

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
		int[] tempLayerNodes = { 4, 8, 8, 3 };
		FullConnectAnn tempNetwork = new FullConnectAnn("D:/data/iris.arff", tempLayerNodes, 0.01,
				0.6, "sss");

		for (int round = 0; round < 5000; round++) {
			tempNetwork.train();
		} // Of for n

		double tempAccuray = tempNetwork.test();
		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("FullConnectAnn ends.");
	}// Of main
}// Of class FullConnectAnn