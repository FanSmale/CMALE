package util;

import java.util.Random;

/**
 * Simple tools.
 * @author Fan Min. minfanphd@163.com, minfan@swpu.edu.cn.
 */
public class SimpleTools {
	public static Random random = new Random();
	
	/**
	 ********************************** 
	 * Normalize the data
	 * 
	 * @param paraMatrix
	 *            the original matrix. It will be changed.
	 ********************************** 
	 */
	public static void normalize(double[][] paraMatrix){
		for(int j = 0; j < paraMatrix[0].length; j ++) {
			double tempMin = 1e20;
			double tempMax = -1e20;
			
			//The min and max values.
			for (int i = 0; i < paraMatrix.length; i++) {
				if (tempMin > paraMatrix[i][j]) {
					tempMin = paraMatrix[i][j];
				}//Of if
				if (tempMax < paraMatrix[i][j]) {
					tempMax = paraMatrix[i][j];
				}//Of if
			}//Of for i
			
			//Handle the extreme case.
			if (tempMin == tempMax) {
				for (int i = 0; i < paraMatrix.length; i++) {
					paraMatrix[i][j] = 0;
				}//Of for i
				continue;
			}//Of if
			
			//Now change.
			for (int i = 0; i < paraMatrix.length; i++) {
				paraMatrix[i][j] = (paraMatrix[i][j] - tempMin) / (tempMax - tempMin);
			}//Of for i
		}//Of for j
	}//Of normalize

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged. The method should be tested further. <br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2]. <br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */
	public static int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];// For merge sort.

		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i

		// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;

		while (tempCurrentLength < tempLength) {
			// Divide into a number of groups.
			// Here the boundary is adaptive to array length not equal to 2^k.
			for (int i = 0; i < Math.ceil((tempLength + 0.0) / tempCurrentLength / 2); i++) {
				// Boundaries of the group
				tempFirstStart = i * tempCurrentLength * 2;

				tempSecondStart = tempFirstStart + tempCurrentLength;

				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {
					tempSecondEnd = tempLength - 1;
				} // Of if

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;

				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;
					} // Of for j
					break;
				} // Of if

				while ((tempFirstIndex <= tempSecondStart - 1)
						&& (tempSecondIndex <= tempSecondEnd)) {

					if (paraArray[resultMatrix[tempIndex
							% 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex
									% 2][tempSecondIndex]]) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempFirstIndex];
						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempSecondIndex];
						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;
				} // Of while

				// Remaining part
				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
							% 2][j];
					tempCurrentIndex++;
				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
							% 2][j];
					tempCurrentIndex++;
				} // Of for j
			} // Of for i

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices

	/**
	 ********************************** 
	 * Get a random order index array.
	 * 
	 * @param paraLength
	 *            The length of the array.
	 * @return A random order.
	 ********************************** 
	 */
	public static int[] getRandomOrder(int paraLength) {
		// Step 1. Initialize
		int[] resultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			resultArray[i] = i;
		} // Of for i

		// Step 2. Swap many times
		int tempFirst, tempSecond;
		int tempValue;
		for (int i = 0; i < paraLength * 10; i++) {
			tempFirst = random.nextInt(paraLength);
			tempSecond = random.nextInt(paraLength);

			tempValue = resultArray[tempFirst];
			resultArray[tempFirst] = resultArray[tempSecond];
			resultArray[tempSecond] = tempValue;
		} // Of for i

		return resultArray;
	}// Of getRandomOrder	
}//Of class SimpleTools
