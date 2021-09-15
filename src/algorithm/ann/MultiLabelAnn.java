package algorithm.ann;

import java.util.Arrays;

/**
 * Full ANN with a number of layers.
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class MultiLabelAnn extends GeneralAnn {

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
	public MultiLabelAnn(String paraFilename, int[] paraFullConnectLayerNumNodes, int paraNumParts,
			int[] paraParallelLayerNumNodes, double paraLearningRate, double paraMobp, String paraActivators) {
		super(paraFilename, paraFullConnectLayerNumNodes.length + paraParallelLayerNumNodes.length, paraLearningRate,
				paraMobp);

		// Initialize layers.
		layers = new GeneralAnnLayer[numLayers - 1];
		for (int i = 0; i < paraFullConnectLayerNumNodes.length; i++) {
			layers[i] = new FullConnectAnnLayer(paraFullConnectLayerNumNodes[i], paraFullConnectLayerNumNodes[i + 1],
					paraActivators.charAt(i), paraLearningRate, paraMobp);
		} // Of for i

		for (int i = 0; i < paraParallelLayerNumNodes.length; i++) {
			layers[paraFullConnectLayerNumNodes.length + i] = new ParallelAnnLayer(paraNumParts,
					paraParallelLayerNumNodes[i], paraParallelLayerNumNodes[i + 1], paraActivators.charAt(i),
					paraLearningRate, paraMobp);
		} // Of for i
	}// Of the first constructor

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
	public void backPropagation(double[] paraTarget) {
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
		int[] tempFullConnectLayerNodes = { 4, 8, 8, 12 };
		int[] tempParallelLayerNodes = { 3, 3, 2 };
		MultiLabelAnn tempNetwork = new MultiLabelAnn("D:/data/iris.arff", tempFullConnectLayerNodes, 3,
				tempParallelLayerNodes, 0.01, 0.6, "sss");

		for (int round = 0; round < 5000; round++) {
			tempNetwork.train();
		} // Of for n

		double tempAccuray = tempNetwork.test();
		System.out.println("The accuracy is: " + tempAccuray);
		System.out.println("FullAnn ends.");
	}// Of main
}// Of class FullAnn
