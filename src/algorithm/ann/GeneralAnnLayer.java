package algorithm.ann;

import java.util.Random;

/**
 * General ANN.
 * 
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 */
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
	 * Set learning rate.
	 * @param paraLearningRate The new learning rate.
	 ********************
	 */
	public void setLearningRate(double paraLearningRate) {
		learningRate = paraLearningRate;
	}//Of setLearningRate
	
	/**
	 ********************
	 * Set mobp.
	 * @param paraMobp The new mobp.
	 ********************
	 */
	public void setMobp(double paraMobp) {
		mobp = paraMobp;
	}//Of setMobp
	
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
	 * @paraLabelKnownArray Which labels are known.
	 * @return Error array.
	 ********************
	 */
	public abstract double[] getLastLayerErrors(int[] paraTarget);
	//, boolean[] paraLabelKnownArray

}//Of class GeneralAnnLayer