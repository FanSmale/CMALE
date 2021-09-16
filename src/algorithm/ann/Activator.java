package algorithm.ann;

/**
 * Activator.
 * 
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 */

public class Activator {
	/**
	 * Arc tan.
	 */
	public final char ARC_TAN = 'a';

	/**
	 * Elu.
	 */
	public final char ELU = 'e';

	/**
	 * Gelu.
	 */
	public final char GELU = 'g';

	/**
	 * Hard logistic.
	 */
	public final char HARD_LOGISTIC = 'h';

	/**
	 * Identity.
	 */
	public final char IDENTITY = 'i';

	/**
	 * Leaky relu, also known as parametric relu.
	 */
	public final char LEAKY_RELU = 'l';

	/**
	 * Relu.
	 */
	public final char RELU = 'r';

	/**
	 * Soft sign.
	 */
	public final char SOFT_SIGN = 'o';

	/**
	 * Sigmoid.
	 */
	public final char SIGMOID = 's';

	/**
	 * Tanh.
	 */
	public final char TANH = 't';

	/**
	 * Soft plus.
	 */
	public final char SOFT_PLUS = 'u';

	/**
	 * Swish.
	 */
	public final char SWISH = 'w';

	/**
	 * The activator.
	 */
	private char activator;

	/**
	 * Alpha for elu.
	 */
	double alpha;

	/**
	 * Beta for leaky relu.
	 */
	double beta;

	/**
	 * Gamma for leaky relu.
	 */
	double gamma;

	/**
	 *********************
	 * The first constructor.
	 * 
	 * @param paraActivator
	 *            The activator.
	 *********************
	 */
	public Activator(char paraActivator) {
		activator = paraActivator;
	}// Of the first constructor

	/**
	 *********************
	 * Setter.
	 *********************
	 */
	public void setActivator(char paraActivator) {
		activator = paraActivator;
	}// Of setActivator

	/**
	 *********************
	 * Getter.
	 *********************
	 */
	public char getActivator() {
		return activator;
	}// Of getActivator

	/**
	 *********************
	 * Setter.
	 *********************
	 */
	void setAlpha(double paraAlpha) {
		alpha = paraAlpha;
	}// Of setAlpha

	/**
	 *********************
	 * Setter.
	 *********************
	 */
	void setBeta(double paraBeta) {
		beta = paraBeta;
	}// Of setBeta

	/**
	 *********************
	 * Setter.
	 *********************
	 */
	void setGamma(double paraGamma) {
		gamma = paraGamma;
	}// Of setGamma

	/**
	 *********************
	 * Activate according to the activation function.
	 *********************
	 */
	public double activate(double paraValue) {
		double resultValue = 0;
		switch (activator) {
		case ARC_TAN:
			resultValue = Math.atan(paraValue);
			break;
		case ELU:
			if (paraValue >= 0) {
				resultValue = paraValue;
			} else {
				resultValue = alpha * (Math.exp(paraValue) - 1);
			} // Of if
			break;
		// case GELU:
		// resultValue = ?;
		// break;
		// case HARD_LOGISTIC:
		// resultValue = ?;
		// break;
		case IDENTITY:
			resultValue = paraValue;
			break;
		case LEAKY_RELU:
			if (paraValue >= 0) {
				resultValue = paraValue;
			} else {
				resultValue = alpha * paraValue;
			} // Of if
			break;
		case SOFT_SIGN:
			if (paraValue >= 0) {
				resultValue = paraValue / (1 + paraValue);
			} else {
				resultValue = paraValue / (1 - paraValue);
			} // Of if
			break;
		case SOFT_PLUS:
			resultValue = Math.log(1 + Math.exp(paraValue));
			break;
		case RELU:
			if (paraValue >= 0) {
				resultValue = paraValue;
			} else {
				resultValue = 0;
			} // Of if
			break;
		case SIGMOID:
			resultValue = 1 / (1 + Math.exp(-paraValue));
			break;
		case TANH:
			resultValue = 2 / (1 + Math.exp(-2 * paraValue)) - 1;
			break;
		// case SWISH:
		// resultValue = ?;
		// break;
		default:
			System.out.println("Unsupported activator: " + activator);
			System.exit(0);
		}// Of switch

		return resultValue;
	}// Of activate

	/**
	 *********************
	 * Derive according to the activation function. Some use x while others use
	 * f(x).
	 * 
	 * @param paraValue
	 *            The original value x.
	 * @param paraActivatedValue
	 *            f(x).
	 *********************
	 */
	public double derive(double paraValue, double paraActivatedValue) {
		double resultValue = 0;
		switch (activator) {
		case ARC_TAN:
			resultValue = 1 / (paraValue * paraValue + 1);
			break;
		case ELU:
			if (paraValue >= 0) {
				resultValue = 1;
			} else {
				resultValue = alpha * (Math.exp(paraValue) - 1) + alpha;
			} // Of if
			break;
		// case GELU:
		// resultValue = ?;
		// break;
		// case HARD_LOGISTIC:
		// resultValue = ?;
		// break;
		case IDENTITY:
			resultValue = 1;
			break;
		case LEAKY_RELU:
			if (paraValue >= 0) {
				resultValue = 1;
			} else {
				resultValue = alpha;
			} // Of if
			break;
		case SOFT_SIGN:
			if (paraValue >= 0) {
				resultValue = 1 / (1 + paraValue) / (1 + paraValue);
			} else {
				resultValue = 1 / (1 - paraValue) / (1 - paraValue);
			} // Of if
			break;
		case SOFT_PLUS:
			resultValue = 1 / (1 + Math.exp(-paraValue));
			break;
		case RELU: // Updated
			if (paraValue >= 0) {
				resultValue = 1;
			} else {
				resultValue = 0;
			} // Of if
			break;
		case SIGMOID: // Updated
			resultValue = paraActivatedValue * (1 - paraActivatedValue);
			break;
		case TANH: // Updated
			resultValue = 1 - paraActivatedValue * paraActivatedValue;
			break;
		// case SWISH:
		// resultValue = ?;
		// break;
		default:
			System.out.println("Unsupported activator: " + activator);
			System.exit(0);
		}// Of switch

		return resultValue;
	}// Of derive

	/**
	 *********************
	 * Overrides the method claimed in Object.
	 *********************
	 */
	public String toString() {
		String resultString = "Activator with function '" + activator + "'";
		resultString += "\r\n alpha = " + alpha + ", beta = " + beta + ", gamma = " + gamma;

		return resultString;
	}// Of toString

	/**
	 ********************
	 * Test the class.
	 ********************
	 */
	public static void main(String[] args) {
		Activator tempActivator = new Activator('s');
		double tempValue = 0.6;
		double tempNewValue;
		tempNewValue = tempActivator.activate(tempValue);
		System.out.println("After activation: " + tempNewValue);

		tempNewValue = tempActivator.derive(tempValue, tempNewValue);
		System.out.println("After derive: " + tempNewValue);
	}// Of main
}// Of class Activator
