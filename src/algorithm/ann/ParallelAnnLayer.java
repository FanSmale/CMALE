package algorithm.ann;

import java.util.Arrays;
import java.util.Random;

/**
 * An ANN layer runs with a number of parts, each for a label.
 * 
 * @author minfanphd
 */
public class ParallelAnnLayer extends GeneralAnnLayer {
	/**
	 * Number of parts.
	 */
	int numParts;

	/**
	 * The number of input for each part.
	 */
	int numInputEachPart;

	/**
	 * The number of output for each part.
	 */
	int numOutputEachPart;

	/**
	 * The weight cubic.
	 */
	double[][][] weights;

	/**
	 * The delta weight cubic.
	 */
	double[][][] deltaWeights;

	/**
	 *********************
	 * The first constructor.
	 * 
	 * @param paraActivator
	 *            The activator.
	 *********************
	 */
	public ParallelAnnLayer(int paraNumParts, int paraNumInputEachPart, int paraNumOutputEachPart, char paraActivator,
			double paraLearningRate, double paraMobp) {
		super(paraActivator, paraLearningRate, paraMobp);

		numParts = paraNumParts;
		numInputEachPart = paraNumInputEachPart;
		numOutputEachPart = paraNumOutputEachPart;

		weights = new double[numParts][numInputEachPart + 1][numOutputEachPart];
		deltaWeights = new double[numParts][numInputEachPart + 1][numOutputEachPart];
		for (int i = 0; i < numParts; i++) {
			for (int j = 0; j < numInputEachPart + 1; j++) {
				for (int k = 0; k < numOutputEachPart; k++) {
					weights[i][j][k] = random.nextDouble();
				} // Of for k
			} // Of for j
		} // Of for i

		errors = new double[numParts * numInputEachPart];

		input = new double[numParts * numInputEachPart];
		output = new double[numParts * numOutputEachPart];
		activatedOutput = new double[numParts * numOutputEachPart];
	}// Of the first constructor

	/**
	 ********************
	 * Forward prediction.
	 * 
	 * @param paraInput
	 *            The input data of one instance.
	 * @return The data at the output end.
	 ********************
	 */
	public double[] forward(double[] paraInput) {
		//System.out.println("Parallel ANN forward");
		// Copy data.
		for (int i = 0; i < numParts * numInputEachPart; i++) {
			input[i] = paraInput[i];
		} // Of for i

		// Calculate the weighted sum for each output.
		for (int i = 0; i < numParts; i++) {
			for (int j = 0; j < numOutputEachPart; j++) {
				output[i * numOutputEachPart + j] = weights[i][numInputEachPart][j];
				for (int k = 0; k < numInputEachPart; k++) {
					output[i * numOutputEachPart + j] += input[i * numInputEachPart + k] * weights[i][k][j];
				} // Of for j

				activatedOutput[i * numOutputEachPart + j] = activator.activate(output[i * numOutputEachPart + j]);
			} // Of for i
		} // Of for i

		return activatedOutput;
	}// Of forward

	/**
	 ********************
	 * Back propagation and change the edge weights.
	 * 
	 * @param paraTarget
	 *            For 3-class data, it is [0, 0, 1], [0, 1, 0] or [1, 0, 0].
	 ********************
	 */
	public double[] backPropagation(double[] paraErrors) {
		//System.out.println("Parallel ANN backPropagation");
		// Step 1. Adjust the errors.
		for (int i = 0; i < paraErrors.length; i++) {
			paraErrors[i] = activator.derive(output[i], activatedOutput[i]) * paraErrors[i];
		} // Of for i

		// Step 2. Compute current errors.
		for (int i = 0; i < numParts; i++)
			for (int j = 0; j < numInputEachPart; j++) {
				errors[i * numInputEachPart + j] = 0;
				for (int k = 0; k < numOutputEachPart; k++) {
					//System.out.println("i * numInputEachPart + j = " + (i * numInputEachPart + j));
					//System.out.println("j * numOutputEachPart + k = " + (i * numOutputEachPart + k));
					errors[i * numInputEachPart + j] += paraErrors[i * numOutputEachPart + k] * weights[i][j][k];
					deltaWeights[i][j][k] = mobp * deltaWeights[i][j][k]
							+ learningRate * paraErrors[i * numOutputEachPart + k] * input[i * numInputEachPart + j];
					weights[i][j][k] += deltaWeights[i][j][k];
				} // Of for j
			} // Of for i

		return errors;
	}// Of backPropagation

	/**
	 ********************
	 * I am the last layer, set the errors.
	 * 
	 * @param paraTarget
	 *            For 3-class data, it is [0, 0, 1], [0, 1, 0] or [1, 0, 0].
	 ********************
	 */
	public double[] getLastLayerErrors(int[] paraTarget) {
		double[] resultErrors = new double[numParts * 2];
		
		for (int i = 0; i < resultErrors.length; i++) {
			resultErrors[i] = (paraTarget[i] - activatedOutput[i]);
		} // Of for i
		//System.out.println("Last layer errors: " + Arrays.toString(resultErrors));

		return resultErrors;
	}// Of getLastLayerErrors

	/**
	 ********************
	 * Unit test.
	 ********************
	 */
	public static void unitTest() {
		ParallelAnnLayer tempLayer = new ParallelAnnLayer(2, 2, 3, 's', 0.01, 0.1);
		double[] tempInput = { 1, 4, 2, 5 };

		System.out.println(tempLayer);

		double[] tempOutput = tempLayer.forward(tempInput);
		System.out.println("Forward, the output is: " + Arrays.toString(tempOutput));

		double[] tempError = tempLayer.backPropagation(tempOutput);
		System.out.println("Back propagation, the error is: " + Arrays.toString(tempError));
	}// Of unitTest

	/**
	 ********************
	 * Test the algorithm.
	 ********************
	 */
	public static void main(String[] args) {
		unitTest();
	}// Of main
}// Of class ParallelAnnLayer
