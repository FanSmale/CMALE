package data;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;
import util.SimpleTools;

/**
 * Multi-label data.
 * 
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 */
public class MultiLabelData {
	/**
	 * For random number generation.
	 */
	Random random = new Random();

	/**
	 * The invalid label.
	 */
	public static final int INVALID_LABEL = -100;

	/**
	 * The data.
	 */
	Instances dataset = null;

	/**
	 * The number of instances.
	 */
	int numInstances;

	/**
	 * The number of conditions.
	 */
	int numConditions;

	/**
	 * The number of labels.
	 */
	int numLabels;

	/**
	 * The data matrix.
	 */
	double[][] dataMatrix;

	/**
	 * The label matrix.
	 */
	int[][] labelMatrix;

	/**
	 * The predicted label matrix.
	 */
	int[][] predictedLabelMatrix;

	/**
	 * Which labels are known.
	 */
	boolean[][] labelQueriedMatrix;

	/**
	 * Does respective instances have label queried. If not, the instance cannot
	 * be used for train.
	 */
	private boolean[] hasLabelQueriedArray;

	/**
	 * The number of queried instances.
	 */
	int numQueriedInstances;

	/**
	 * The number of queried labels.
	 */
	int numQueriedLabels;

	/**
	 * Does respective instances have label queried. If not, the instance cannot
	 * be used for train.
	 */
	int[] queriedInstanceArray;

	/**
	 * Teacher cost for one query.
	 */
	double teacherCost = 1;

	/**
	 * Mis-classification cost for FP, i.e., prediction 0 as 1.
	 */
	double fpCost = 1;

	/**
	 * Mis-classification cost for FN, i.e., prediction 1 as 0.
	 */
	double fnCost = 2;

	/**
	 * Manhattan distance.
	 */
	public static final int MANHATTAN = 0;

	/**
	 * Euclidean distance.
	 */
	public static final int EUCLIDEAN = 1;

	/**
	 * The distance measure.
	 */
	public int distanceMeasure = EUCLIDEAN;

	/**
	 ********************** 
	 * The first constructor. Data and labels are stored in one file.
	 * 
	 * @paraDataFilename The data filename.
	 ********************** 
	 */
	public MultiLabelData(String paraArffFilename, int paraNumConditions, int paraNumLabels) {
		try {
			FileReader tempReader = new FileReader(paraArffFilename);
			dataset = new Instances(tempReader);
			// The last attribute is the decision class.
			dataset.setClassIndex(dataset.numAttributes() - 1);
			tempReader.close();
		} catch (Exception ee) {
			System.out.println("Error occurred while trying to read \'" + paraArffFilename
					+ "\' in GeneralAnn constructor.\r\n" + ee);
			System.exit(0);
		} // of try

		// Data matrix initialization.
		numInstances = dataset.numInstances();
		numConditions = paraNumConditions;
		numLabels = paraNumLabels;

		dataMatrix = new double[numInstances][numConditions];
		for (int i = 0; i < dataMatrix.length; i++) {
			for (int j = 0; j < dataMatrix[i].length; j++) {
				dataMatrix[i][j] = dataset.instance(i).value(j);
			} // Of for j
		} // Of for i

		// Normalize it.
		SimpleTools.normalize(dataMatrix);

		// Label matrix initialization.
		labelMatrix = new int[numInstances][numLabels];
		predictedLabelMatrix = new int[numInstances][numLabels];
		for (int i = 0; i < labelMatrix.length; i++) {
			for (int j = 0; j < labelMatrix[0].length; j++) {
				labelMatrix[i][j] = (int) dataset.instance(i).value(numConditions + j);
				predictedLabelMatrix[i][j] = INVALID_LABEL;
			} // Of for j
		} // Of for i

		hasLabelQueriedArray = new boolean[numInstances];
		labelQueriedMatrix = new boolean[numInstances][numLabels];
		numQueriedInstances = 0;
		queriedInstanceArray = new int[numInstances];
	}// Of the first constructor

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public int getNumInstances() {
		return numInstances;
	}// Of getNumInstances

	/**
	 ********************** 
	 * Getter.
	 ********************** 
	 */
	public int getNumConditions() {
		return numConditions;
	}// Of getNumConditions

	/**
	 ********************** 
	 * Getter.
	 ********************** 
	 */
	public int getQueriedInstanceIndex(int paraIndex) {
		return queriedInstanceArray[paraIndex];
	}// Of getQueriedInstanceIndex

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public int getNumLabels() {
		return numLabels;
	}// Of getNumConditions

	/**
	 ********************** 
	 * Getter. Get one row.
	 ********************** 
	 */
	public double[] getData(int paraRow) {
		return dataMatrix[paraRow];
	}// Of getData

	/**
	 ********************** 
	 * Getter. Get one datum.
	 ********************** 
	 */
	public double getData(int paraRow, int paraColumn) {
		return dataMatrix[paraRow][paraColumn];
	}// Of getData

	/**
	 ********************** 
	 * Getter. Get the labels of one instance.
	 ********************** 
	 */
	public int[] getLabel(int paraRow) {
		return labelMatrix[paraRow];
	}// Of getLabel

	/**
	 ********************** 
	 * Getter. Get one particular label.
	 ********************** 
	 */
	public int getLabel(int paraRow, int paraColumn) {
		return labelMatrix[paraRow][paraColumn];
	}// Of getLabel

	/**
	 ********************** 
	 * Set predicted label.
	 ********************** 
	 */
	public void setPredictedLabel(int paraRow, int paraColumn, int paraValue) {
		predictedLabelMatrix[paraRow][paraColumn] = paraValue;
	}// Of setPredictedLabel

	/**
	 ********************** 
	 * Getter.
	 ********************** 
	 */
	public double getTeacherCost() {
		return teacherCost;
	}// Of getTeacherCost

	/**
	 ********************** 
	 * Setter.
	 ********************** 
	 */
	public void setTeacherCost(double paraTeacherCost) {
		teacherCost = paraTeacherCost;
	}// Of setTeacherCost

	/**
	 ********************** 
	 * Getter.
	 ********************** 
	 */
	public double getFpCost() {
		return fpCost;
	}// Of getFpCost

	/**
	 ********************** 
	 * Setter.
	 ********************** 
	 */
	public void setFpCost(double paraFpCost) {
		fpCost = paraFpCost;
	}// Of setFpCost

	/**
	 ********************** 
	 * Getter.
	 ********************** 
	 */
	public double getFnCost() {
		return fnCost;
	}// Of getFnCost

	/**
	 ********************** 
	 * Setter.
	 ********************** 
	 */
	public void setFnCost(double paraFnCost) {
		fnCost = paraFnCost;
	}// Of setFnCost

	/**
	 ********************** 
	 * Getter. Get the labels of one instance. Not queried labels will be
	 * INVALID_LABEL.
	 ********************** 
	 */
	public int[] getQueriedLabel(int paraRow) {
		int[] resultLabels = new int[numLabels];
		for (int i = 0; i < resultLabels.length; i++) {
			if (labelQueriedMatrix[paraRow][i]) {
				resultLabels[i] = labelMatrix[paraRow][i];
			} else {
				resultLabels[i] = INVALID_LABEL;
			} // Of if
		} // Of for i

		return resultLabels;
	}// Of getQueriedLabel

	/**
	 ********************** 
	 * Getter. Get one particular label. If the label is not queried, return
	 * INVALID_LABEL.
	 * 
	 * @param paraRow
	 *            The instance index.
	 * @param paraColumn
	 *            The label index.
	 ********************** 
	 */
	public int getQueriedLabel(int paraRow, int paraColumn) {
		if (!labelQueriedMatrix[paraRow][paraColumn]) {
			return INVALID_LABEL;
		} // Of if

		return labelMatrix[paraRow][paraColumn];
	}// Of getQueriedLabel

	/**
	 ********************** 
	 * Randomly query a proportion of labels.
	 ********************** 
	 */
	public void randomQuery(double paraProportion) {
		int[] tempArray = new int[1];
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (random.nextDouble() < paraProportion) {
					tempArray[0] = j;
					queryLabels(i, tempArray);
				} // Of if
			} // Of for j
		} // Of for i

		System.out.println(
				"After query " + numQueriedInstances + " instances, the query instances are ");
		System.out.println(Arrays.toString(queriedInstanceArray));
	}// Of randomQuery

	/**
	 ********************** 
	 * Get the label known status for one instance.
	 ********************** 
	 */
	public boolean[] getLabelQueried(int paraRow) {
		return labelQueriedMatrix[paraRow];
	}// Of getLabelQueried

	/**
	 ********************** 
	 * Get the label quried status.
	 ********************** 
	 */
	public boolean getLabelQueried(int paraRow, int paraColumn) {
		return labelQueriedMatrix[paraRow][paraColumn];
	}// Of getLabelQueried

	/**
	 ********************** 
	 * Get the number of queried instances.
	 ********************** 
	 */
	public int getNumQueriedInstances() {
		return numQueriedInstances;
	}// Of getNumQueriedInstances

	/**
	 ********************** 
	 * Query labels. If the label is already queried, terminate the algorithm
	 * since it is an internal error.
	 * 
	 * @paraInstance The instance to query.
	 * @param paraLabelIndices
	 *            The indices of queried labels
	 ********************** 
	 */
	public void queryLabels(int paraInstance, int[] paraLabelIndices) {
		if (paraLabelIndices.length == 0) {
			System.out.println("Internal error occurred in MultiLabelData.queryLabels()."
					+ "Cannot query an empty set of labels.");
			System.exit(0);
		} // Of if

		for (int j = 0; j < paraLabelIndices.length; j++) {
			if (labelQueriedMatrix[paraInstance][paraLabelIndices[j]]) {
				System.out.println("Internal error occurred in MultiLabelData.queryLabels()."
						+ "Cannot query a label twice.");
				System.out.println(
						Arrays.toString(labelMatrix[paraInstance]) + ", " + paraLabelIndices[j]);
				System.exit(0);
			} // Of if

			labelQueriedMatrix[paraInstance][paraLabelIndices[j]] = true;
		} // Of for j

		// Update the queried instance array.
		if (!hasLabelQueriedArray[paraInstance]) {
			hasLabelQueriedArray[paraInstance] = true;
			queriedInstanceArray[numQueriedInstances] = paraInstance;
			numQueriedInstances++;
		} // Of if

		numQueriedLabels += paraLabelIndices.length;
	}// Of queryLabels

	/**
	 ********************** 
	 * Get the number of queried labels.
	 ********************** 
	 */
	public int getNumQueriedLabels() {
		return numQueriedLabels;
	}// Of getNumQueriedLabels

	/**
	 ********************** 
	 * Compute accuracy.
	 ********************** 
	 */
	public double computeAccuracy() {
		double tempCorrect = 0;

		for (int i = 0; i < labelMatrix.length; i++) {
			for (int j = 0; j < labelMatrix[0].length; j++) {
				// It is correct.
				if (predictedLabelMatrix[i][j] == labelMatrix[i][j]) {
					tempCorrect ++;
				} // Of if
			} // Of for j
		} // Of for i

		return tempCorrect / numInstances / numLabels;
	}// Of computeAccuracy
	
	/**
	 ********************** 
	 * Compute the total cost.
	 ********************** 
	 */
	public double computeTotalCost() {
		double tempTotalTeacherCost = numQueriedLabels * teacherCost;
		int tempNumFp = 0;
		int tempNumFn = 0;

		double tempTotalMisclassificationCost = 0;
		for (int i = 0; i < labelMatrix.length; i++) {
			for (int j = 0; j < labelMatrix[0].length; j++) {
				// It is correct.
				if (predictedLabelMatrix[i][j] == labelMatrix[i][j]) {
					continue;
				} // Of if

				if ((predictedLabelMatrix[i][j] == 0) && (labelMatrix[i][j] == 1)) {
					tempTotalMisclassificationCost += fnCost;
					tempNumFn ++;
				} else if ((predictedLabelMatrix[i][j] == 1) && (labelMatrix[i][j] == 0)) {
					tempTotalMisclassificationCost += fpCost;
					tempNumFp ++;
				} else {
					System.out.println("Error occurred in MultiLabelData.computeTotalCost()\r\n"
							+ "The label at [" + i + "][" + j + "] has not been predicted.\r\n"
							+ predictedLabelMatrix[i][j] + " vs. " + labelMatrix[i][j]);
					System.exit(0);
				} // Of if
			} // Of for j
		} // Of for i

		System.out.println("FP = " + tempNumFp + ", FN = "
				+ tempNumFn);
		System.out.println("Teacher cost = " + tempTotalTeacherCost + ", misclassification cost = "
				+ tempTotalMisclassificationCost);
		return tempTotalTeacherCost + tempTotalMisclassificationCost;
	}// Of computeTotalCost

	/**
	 ********************** 
	 * Compute the distance between two instances
	 * 
	 * @param paraI
	 *            The first index.
	 * @param paraI
	 *            The second index.
	 ********************** 
	 */
	public double distance(int paraI, int paraJ) {
		int resultDistance = 0;
		double tempDifference;
		switch (distanceMeasure) {
		case MANHATTAN:
			for (int i = 0; i < numConditions; i++) {
				tempDifference = dataMatrix[paraI][i] - dataMatrix[paraJ][i];
				// Sum up the distance.
				if (tempDifference < 0) {
					resultDistance -= tempDifference;
				} else {
					resultDistance += tempDifference;
				} // of if
			} // of for i
			break;

		case EUCLIDEAN:
			for (int i = 0; i < numConditions; i++) {
				tempDifference = dataMatrix[paraI][i] - dataMatrix[paraJ][i];
				resultDistance += tempDifference * tempDifference;
			} // of for i
			break;
		default:
			System.out.println("Unsupported distance measure: " + distanceMeasure);
		}// of switch
		return resultDistance;
	}// Of distance

	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
	public String toString() {
		String resultString = "The data has " + numInstances + " instances, " + numConditions
				+ " conditions, and " + numLabels + " labels.";
		resultString += "\r\nData\r\n" + Arrays.deepToString(dataMatrix);
		resultString += "\r\nLabel\r\n" + Arrays.deepToString(labelMatrix);
		return resultString;
	}// Of toString

	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
	public static void readDataTest() {
		MultiLabelData tempDataset = new MultiLabelData("D:/data/multilabel/flags.arff", 14, 12);
		System.out.println("The data is:\r\n" + tempDataset);
	}// Of readDataTest

	/**
	 ********************** 
	 * The entrance.
	 ********************** 
	 */
	public static void main(String[] args) {
		readDataTest();
		System.out.println("Finish.");
	}// Of main
}// Of class MultiLabelData
