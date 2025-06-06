import java.util.*;
import java.io.*;
/**
 * Tensor: A general-purpose multidimensional numerical array class for matrix and tensor operations.
 *
 * <p>This class serves as the unified data structure across all core JANET subsystems:
 * <b>CuBridge</b> (numerical computation), <b>DataBridge</b> (data preprocessing),
 * <b>JANET</b> (neural network operations), and <b>ExBridge</b> (visualization).
 *
 * <p>Internally, a tensor is a dense, flat double array accompanied by shape metadata,
 * capable of representing scalars, vectors, matrices, or N-dimensional arrays. It is designed
 * to be memory-efficient, extensible, and highly interoperable across modules.
 *
 * <h2>Role as a Bridge</h2>
 * <ul>
 *   <li><b>CuBridge:</b> Executes numerical and GPU-accelerated operations using Tensors as operands.</li>
 *   <li><b>DataBridge:</b> Converts structured table data (e.g., CSV) into Tensor format for analysis.</li>
 *   <li><b>JANET:</b> Handles layer input/output as Tensors in neural network flows.</li>
 *   <li><b>ExBridge:</b> Uses Tensor values to render graphs, previews, or dimensional plots.</li>
 * </ul>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Multiple constructors: scalar, 1D/ND array, CSV file</li>
 *   <li>Shape-aware printing and recursive formatting</li>
 *   <li>Random tensor creation (uniform, normal, Gaussian)</li>
 *   <li>Common patterns: zeros, ones, eye, arange, linspace</li>
 *   <li>Reshape and flattening utilities</li>
 *   <li>Conversion tools: head, printSize, toArray</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a 2D tensor of ones (shape: 3x3)
 * Tensor t = Tensor.ones(new int[] {3, 3});
 * 
 * // Flatten and inspect
 * Tensor flat = Tensor.flatten(t);
 * flat.printData();
 * 
 * // Load a dataset
 * Tensor csv = new Tensor("data/input.csv");
 * }</pre>
 *
 * <h2>Function Summary</h2>
 * <ul>
 *   <li><b>Creation:</b> filled, zeros, ones, rand, randn, eye, arange, linspace</li>
 *   <li><b>Structure:</b> reshape, flatten, getShape, getSize, getAxis</li>
 *   <li><b>IO/Inspect:</b> printData, printSize, head, toArray</li>
 * </ul>
 *
 * @author 배준호, 조선대 3학년
 * @since 1.0
 */
public class Tensor {
	private double[] data = null;
	private int[] shape = null;
	private int len = 0;

	
	
	private void print(String str) {
		System.out.println(str);
	}

	/**
	 * Calculates the total number of elements implied by the given shape.
	 *
	 * @param shape the array of shape dimensions
	 * @return the total number of elements (product of dimensions)
	 */
	private int getLenFromShape() {
		int size = 1;
		for (var tmp : shape)
			size *= tmp;
		return size;
	}

	
	/**
	 * Constructs an empty Tensor with no data or shape defined.
	 */
	Tensor() {
	}

	/**
	 * Constructs a 1-dimensional Tensor from a given array of doubles.
	 * <ul>
	 *   <li>The tensor will have shape (N), where N is the length of the input array.</li>
	 *   <li>The input values are deep-copied for internal storage.</li>
	 * </ul>
	 * @param data one-dimensional data values
	 */
	Tensor(double... data) {
		this.shape = new int[] { data.length };
		this.data = data.clone();
		this.len = data.length;
	}

	/**
	 * Constructs a Tensor with the given data and shape.
	 * <ul>
	 *   <li>Validates whether the number of data points matches the product of the provided shape dimensions.</li>
	 *   <li>If the shape is valid, it is applied directly.</li>
	 *   <li>If a mismatch is detected, a warning is printed and the shape is automatically set to 1D (N,) format.</li>
	 * </ul>
	 *
	 * @param data the flat data array
	 * @param shape the intended shape dimensions
	 */
	Tensor(double[] data, int... shape) {
		this.shape = shape.clone();
		this.data = data.clone();
		this.len = data.length;

		if (len != getLenFromShape()) {
		    System.err.println("Warning: Shape mismatch detected. Overriding shape to 1D.");
		    this.shape = new int[] { len };
		}
	}

	/**
	 * Loads a Tensor from a CSV file.
	 * <ul>
	 *   <li>The file must be structured as rows and columns of numerical data, comma-separated.</li>
	 *   <li>Empty fields are automatically converted to 0.0.</li>
	 *   <li>The resulting tensor will have shape (rows, columns) excluding the first line (assumed to be a header or label row).</li>
	 * </ul>
	 * @param path file path to the CSV file
	 */
	Tensor(String path) {// TODO : 만약 다축 텐서의 경우 어떻게 저장하는가? // 야 이 csv가 다축일리가 있겠냐?
		ArrayList<Double> tmpList = new ArrayList<Double>();
		Scanner sc = null;
		int col = 0;
		int row = 0;

		try {
			FileInputStream fin = new FileInputStream(path);
			BufferedInputStream bin = new BufferedInputStream(fin);
			sc = new Scanner(bin);

			String line = null;
			String[] tmp = null;

			line = sc.nextLine();
			tmp = line.split(",", -1);// 라벨 제거

			row = tmp.length;

			while ((line = sc.nextLine()) != null) {
				col++;
				tmp = line.split(",", -1);

				for (int r = 0; r < row; r++)
					if (tmp[r].equals(""))
						tmpList.add(0.0);
					else
						tmpList.add(Double.parseDouble(tmp[r]));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NoSuchElementException e) {
			sc.close();
		}

		this.len = tmpList.size();
		this.data = new double[this.len];
		this.shape = new int[] { col, row };

		for (int idx = 0; idx < this.len; idx++)
			this.data[idx] = tmpList.get(idx);

	}

	/**
	 * Returns a copy of the internal data array.
	 *
	 * @return cloned array containing tensor values
	 */
	double[] toArray() {
		return this.data.clone();
	}

	/**
	 * Returns the total number of elements in the tensor.
	 *
	 * @return the number of elements (length)
	 */
	int getSize() {
		return this.len;
	}

	/**
	 * Returns the shape of the tensor.
	 *
	 * @return cloned array representing tensor dimensions
	 */
	int[] getShape() {
		return this.shape.clone();
	}

	/**
	 * Returns the number of axes (dimensions) of the tensor.
	 *
	 * @return the number of dimensions
	 */
	int getAxis() {
		return this.shape.length;
	}
	
	/**
	 * Prints the size (shape) of the tensor in a flat format.
	 * Example: [3 4 5]
	 */
	void printSize() {
		for(int i = 0; i < shape.length; i++)
		System.out.print(this.shape[i]+" ");
		System.out.println();
	}

	/**
	 * Pretty-prints the tensor data with shape information.
	 * 
	 * If the tensor is empty, prints a notice. Otherwise, recursively prints nested values
	 * according to the tensor's shape structure.
	 */
	public void printData() {
		System.out.println("Tensor(shape=" + Arrays.toString(shape) + "):");
		if (len == 0) {
			System.out.println("(empty tensor)");
			return;
		}
		printRecursive(0, 0, "");
	}

	/**
	 * Helper method for recursive tensor printing by dimension.
	 *
	 * @param level  the current depth in the shape hierarchy
	 * @param offset the flat array offset at this level
	 * @param indent indentation string for formatting
	 */
	private void printRecursive(int level, int offset, String indent) {
		if (level == shape.length - 1) {
			// 마지막 차원: 실제 값 출력
			System.out.print(indent + "  [ ");
			for (int i = 0; i < shape[level]; i++) {
				System.out.printf("%6.3f", data[offset + i]);
				if (i < shape[level] - 1)
					System.out.print(", ");
			}
			System.out.println(" ]");
		} else {
			// 내부 차원: 블럭 나눠서 재귀 출력
			int stride = 1;
			for (int i = level + 1; i < shape.length; i++)
				stride *= shape[i];

			for (int i = 0; i < shape[level]; i++) {
//				System.out.println(indent + "[" + i + "]");
				printRecursive(level + 1, offset + i * stride, indent + "  ");
			}
		}
	}

	/**
	 * Returns a string that displays the shape and the first few rows of the tensor.
	 * 
	 * <ul>
	 *   <li>Shows up to 6 rows of data.</li>
	 *   <li>Each row displays values from the last dimension (columns).</li>
	 *   <li>Intended for quick inspection of tensor contents.</li>
	 * </ul>
	 *
	 * @return a formatted string preview of the tensor
	 */
	String head() {
		int dim = shape.length;
		if (dim == 0)
			return "Tensor(shape=[], data=[])";

		int cols = shape[dim - 1]; // 마지막 축 (열 수)
		int totalRows = len / cols; // 전체 행 수
		int rowCount = Math.min(6, totalRows); // 최대 6줄

		StringBuilder sb = new StringBuilder();
		sb.append("Tensor(shape=").append(Arrays.toString(shape)).append(", head=\n");

		for (int i = 0; i < rowCount; i++) {
			sb.append(" [");
			for (int j = 0; j < cols; j++) {
				sb.append(String.format("%6.2f", data[i * cols + j]));
				if (j < cols - 1)
					sb.append(", ");
			}
			sb.append("]");
			if (i < rowCount - 1)
				sb.append("\n");
		}

		sb.append("\n)");
		return sb.toString();
	}

	/**
	 * Returns a string representation of the tensor's shape.
	 *
	 * @return a string indicating the tensor's shape
	 */
	public String toString() {
		return "shape = " + Arrays.toString(shape);
	}
	
	/**
	 * Computes the total number of elements implied by the given shape.
	 * 
	 * @param shape the tensor shape as an array of dimension sizes
	 * @return the product of all dimension sizes (total number of elements)
	 */
	private static int getLenFromShape(int... shape) {
		int size = 1;
		for (int d : shape)
			size *= d;
		return size;
	}
	
	/**
	 * Creates a tensor filled with the specified constant value.
	 *
	 * @param value the constant value to fill
	 * @param shape the desired tensor shape
	 * @return a new tensor filled with {@code value}
	 */
	public static Tensor filled(double value, int... shape) {
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new double[t.len];

		for (int i = 0; i < t.len; i++)
			t.data[i] = value;

		return t;
	}
	
	/**
	 * Creates a tensor filled with random values in the range [0.0, 1.0).
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor with uniformly distributed random values
	 */
	public static Tensor rand(int... shape) {
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new double[t.len];
		
		for (int i = 0; i < t.len; i++)
			t.data[i] = Math.random();

		return t;
	}
	
	/**
	 * Creates a tensor filled with standard normally distributed random values (mean = 0, std = 1).
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor with normally distributed random values
	 */
	public static Tensor randn(int... shape) {
		Random random = new Random();
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new double[t.len];
		
		for (int i = 0; i < t.len; i++)
			t.data[i] = random.nextGaussian();

		return t;
	}
	
	/**
	 * Creates a tensor filled with normally distributed random values with specified mean and standard deviation.
	 *
	 * @param mean the mean of the distribution
	 * @param std the standard deviation of the distribution
	 * @param shape the desired tensor shape
	 * @return a new tensor with normally distributed values
	 */
	public static Tensor randn(double mean, double std, int... shape) {
		Random random = new Random();
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new double[t.len];
		
		for (int i = 0; i < t.len; i++)
			t.data[i] = std * random.nextGaussian() + mean;

		return t;
	}

	/**
	 * Creates a tensor filled with zeros.
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor filled with 0.0
	 */
	public static Tensor zeros(int... shape) {
		return filled(0.0, shape);
	}

	/**
	 * Creates a tensor filled with ones.
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor filled with 1.0
	 */
	public static Tensor ones(int... shape) {
		return filled(1.0, shape);
	}

	/**
	 * Creates an identity matrix of size {@code n x n}.
	 *
	 * @param n the number of rows and columns
	 * @return an identity matrix tensor
	 */
	public static Tensor eye(int n) {
		Tensor t = new Tensor();
		t.shape = new int[] { n, n };
		t.len = n * n;
		t.data = new double[t.len];

		for (int i = 0; i < n; i++)
			t.data[i * n + i] = 1.0;

		return t;
	}

	/**
	 * Creates a 1D tensor with values from {@code start} to less than {@code end}, with given {@code step}.
	 *
	 * @param start the starting value (inclusive)
	 * @param end the end value (exclusive)
	 * @param step the increment step (must be > 0)
	 * @return a 1D tensor with range values
	 */
	public static Tensor arange(double start, double end, double step) {
		if (start >= end) {
			System.out.printf("Error: Tensor.arange() - start (%.3f) must be less than end (%.3f)\n", start, end);
			return new Tensor();
		}
		if (step <= 0) {
			System.out.printf("Error: Tensor.arange() - step (%.3f) must be positive\n", step);
			return new Tensor();
		}

		int length = (int) Math.ceil((end - start) / step);
		Tensor t = new Tensor();
		t.shape = new int[] { length };
		t.len = length;
		t.data = new double[length];

		for (int i = 0; i < length; i++)
			t.data[i] = start + i * step;

		return t;
	}

	/**
	 * Creates a 1D tensor with {@code num} evenly spaced values between {@code start} and {@code end}.
	 *
	 * @param start the starting value (inclusive)
	 * @param end the end value (inclusive)
	 * @param num number of values to generate (must be ≥ 1)
	 * @return a 1D tensor with evenly spaced values
	 */
	public static Tensor linspace(double start, double end, int num) {
		Tensor t = new Tensor();

		if (num <= 0) {
			System.out.printf("Error: Tensor.linspace() - num (%d) must be positive\n", num);
			return t;
		}

		t.shape = new int[] { num };
		t.len = num;
		t.data = new double[num];

		if (num == 1) {
			t.data[0] = start;
			return t;
		}

		double step = (end - start) / (num - 1);

		for (int i = 0; i < num; i++)
			t.data[i] = start + i * step;

		return t;
	}

	/**
	 * Reshapes a given tensor into a new shape.
	 *
	 * @param src the original tensor
	 * @param newShape the new desired shape
	 * @return a reshaped tensor with the same data
	 * @throws IllegalArgumentException if total size mismatches
	 */
	public static Tensor reshape(Tensor src, int... newShape) {
		int newLen = getLenFromShape(newShape);
		if (newLen != src.len)
			throw new IllegalArgumentException("reshape size mismatch");

		Tensor t = new Tensor();
		t.data = src.data.clone();
		t.shape = newShape.clone();
		t.len = src.len;
		return t;
	}

	/**
	 * Flattens a tensor into a 1D tensor.
	 *
	 * @param src the tensor to flatten
	 * @return a 1D tensor with all elements
	 */
	public static Tensor flatten(Tensor src) {
		return reshape(src, src.getSize());
	}

	//rand, randn

}
