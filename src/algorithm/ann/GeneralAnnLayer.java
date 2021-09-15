package algorithm.ann;

import java.util.Random;

public abstract class GeneralAnnLayer {
	/**
	 * The learning rate.
	 */
	double learningRate;

	/**
	 * The mobp.
	 */
	double mobp;

	/**
	 * The errors.
	 */
	double[] errors;

	/**
	 * The inputs.
	 */
	double[] input;

	/**
	 * The outputs.
	 */
	double[] output;

	/**
	 * The output after activate.
	 */
	double[] activatedOutput;

	/**
	 * The inputs.
	 */
	Activator activator;

	/**
	 * Random instance.
	 */
	Random random = new Random();

	/**
	 *********************
	 * The first constructor.
	 * 
	 * @param paraActivator
	 *            The activator.
	 *********************
	 */
	public GeneralAnnLayer(char paraActivator, double paraLearningRate, double paraMobp) {
		learningRate = paraLearningRate;
		mobp = paraMobp;
		activator = new Activator(paraActivator);
	}// Of the first constructor
	
	/**
	 ********************
	 * Set parameters for the activator.
	 * 
	 * @param paraAlpha
	 *            Alpha. Only valid for certain types.
	 * @param paraBeta
	 *            Beta.
	 * @param paraAlpha
	 *            Alpha.
	 ********************
	 */
	public void setParameters(double paraAlpha, double paraBeta, double paraGamma) {
		activator.setAlpha(paraAlpha);
		activator.setBeta(paraBeta);
		activator.setGamma(paraGamma);
	}// Of setParameters

	/**
	 ********************
	 * Forward prediction.
	 * 
	 * @param paraInput
	 *            The input data of one instance.
	 * @return The data at the output end.
	 ********************
	 */
	public abstract double[] forward(double[] paraInput);
	
	/**
	 ********************
	 * Back propagation and change the edge weights.
	 * 
	 * @param paraTarget
	 *            For 3-class data, it is [0, 0, 1], [0, 1, 0] or [1, 0, 0].
	 ********************
	 */
	public abstract double[] backPropagation(double[] paraErrors);
	
	/**
	 ********************
	 * I am the last layer, set the errors.
	 * 
	 * @param paraTarget
	 *            For 3-class data, it is [0, 0, 1], [0, 1, 0] or [1, 0, 0].
	 ********************
	 */
	public abstract double[] getLastLayerErrors(double[] paraTarget);

}//Of class GeneralAnnLayer