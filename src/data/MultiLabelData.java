package data;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;
import util.SimpleTools;

/**
 * Multi-label data.
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 */
public class MultiLabelData {
	/**
	 * For random number generation.
	 */
	Random random = new Random();

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
	 * Which labels are known.
	 */
	boolean[][] labelKnownMatrix;
	
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
        }//of try

        //Data matrix initialization.
		numInstances = dataset.numInstances();
		numConditions = paraNumConditions;
		numLabels = paraNumLabels;
		
        dataMatrix = new double[numInstances][numConditions];
        for (int i = 0; i < dataMatrix.length; i++) {
			for (int j = 0; j < dataMatrix[i].length; j++) {
				dataMatrix[i][j] = dataset.instance(i).value(j);
			}//Of for j
		}//Of for i
        //Normalize it.
        SimpleTools.normalize(dataMatrix);
        
        //Label matrix initialization.
        labelMatrix = new int[numInstances][numLabels];
        for (int i = 0; i < labelMatrix.length; i++) {
			for (int j = 0; j < labelMatrix[0].length; j++) {
				labelMatrix[i][j] = (int)dataset.instance(i).value(numConditions + j);
			}//Of for j
		}//Of for i
        
        labelKnownMatrix = new boolean[numInstances][numLabels];
	}// Of the first constructor
	
	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public int getNumInstances() {
		return numInstances;
	}//Of getNumInstances

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public int getNumConditions() {
		return numConditions;
	}//Of getNumConditions

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public int getNumLabels() {
		return numLabels;
	}//Of getNumConditions

	/**
	 ********************** 
	 * Getter. Get one row.
	 ********************** 
	 */
	public double[] getData(int paraRow) {
		return dataMatrix[paraRow];
	}//Of getData
	
	/**
	 ********************** 
	 * Getter. Get one datum.
	 ********************** 
	 */
	public double getData(int paraRow, int paraColumn) {
		return dataMatrix[paraRow][paraColumn];
	}//Of getData
	
	/**
	 ********************** 
	 * Getter. Get the labels of one instance.
	 ********************** 
	 */
	public int[] getLabel(int paraRow) {
		return labelMatrix[paraRow];
	}//Of getLabel

	/**
	 ********************** 
	 * Getter. Get one particular label.
	 ********************** 
	 */
	public int getLabel(int paraRow, int paraColumn) {
		return labelMatrix[paraRow][paraColumn];
	}//Of getLabel

	/**
	 ********************** 
	 * Set the label as known.
	 ********************** 
	 */
	public void setLabelKnown(int paraRow, int paraColumn) {
		labelKnownMatrix[paraRow][paraColumn] = true;
	}//Of setLabelKnown

	/**
	 ********************** 
	 * Set a proportion of labels as known. For test only.
	 ********************** 
	 */
	public void randomizeLabelKnownMatrix(double paraProportion) {
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (random.nextDouble() < paraProportion) {
					labelKnownMatrix[i][j] = true;
				} else {
					labelKnownMatrix[i][j] = false;
				}//Of if
			}//Of for j
		}//Of for i
		//System.out.println("labelKnownMatrix = " + Arrays.deepToString(labelKnownMatrix));
	}//Of setLabelKnown

	/**
	 ********************** 
	 * Get the label known status for one instance.
	 ********************** 
	 */
	public boolean[] getLabelKnown(int paraRow) {
		return labelKnownMatrix[paraRow];
	}//Of getLabelKnown

	/**
	 ********************** 
	 * Get the label known status.
	 ********************** 
	 */
	public boolean getLabelKnown(int paraRow, int paraColumn) {
		return labelKnownMatrix[paraRow][paraColumn];
	}//Of getLabelKnown

	/**
	 ********************** 
	 * Compute the distance between two instances
	 * @param paraI The first index.
	 * @param paraI The second index.
	 ********************** 
	 */
    public double distance(int paraI, int paraJ) {
        int resultDistance = 0;
        double tempDifference;
        switch (distanceMeasure) {
            case MANHATTAN:
                for (int i = 0; i < numConditions; i++) {
                    tempDifference = dataMatrix[paraI][i] - dataMatrix[paraJ][i];
                    //Sum up the distance.
                    if (tempDifference < 0) {
                        resultDistance -= tempDifference;
                    } else {
                        resultDistance += tempDifference;
                    }//of if
                }//of for i
                break;

            case EUCLIDEAN:
                for (int i = 0; i < numConditions; i++) {
                    tempDifference = dataMatrix[paraI][i] - dataMatrix[paraJ][i];
                    resultDistance += tempDifference * tempDifference;
                }//of for i
                break;
            default:
                System.out.println("Unsupported distance measure: " + distanceMeasure);
        }//of switch
        return resultDistance;
    }//Of distance
    
	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
    public String toString() {
    	String resultString = "The data has " + numInstances + " instances, " + numConditions + " conditions, and " + numLabels + " labels.";
    	resultString += "\r\nData\r\n" + Arrays.deepToString(dataMatrix);
    	resultString += "\r\nLabel\r\n" + Arrays.deepToString(labelMatrix);
    	return resultString;
    }//Of toString

	/**
	 ********************** 
	 * Test reading data.
	 ********************** 
	 */
    public static void readDataTest() {
    	MultiLabelData tempDataset = new MultiLabelData("D:/data/multilabel/flags.arff", 14, 12);
    	System.out.println("The data is:\r\n"  + tempDataset);
    }//Of readDataTest

    /**
	 ********************** 
	 * The entrance.
	 ********************** 
	 */
    public static void main(String[] args) {
    	readDataTest();
    	System.out.println("Finish.");
    }//Of main
}//Of class MultiLabelData 
