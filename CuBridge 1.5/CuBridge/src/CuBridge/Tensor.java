package CuBridge;

import javax.sound.sampled.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.io.*;

/**
 * The {@code Tensor} class represents a general-purpose, multidimensional numerical array
 * designed for matrix and tensor operations across all JANET subsystems.
 *
 * <p>This class serves as the unified data structure shared by:
 * <b>CuBridge</b> (numerical computation), <b>DataBridge</b> (data preprocessing),
 * <b>JANET</b> (neural network execution), and <b>ExBridge</b> (data visualization).
 *
 * <p>Internally, a tensor is implemented as a dense, contiguous {@code float[]} array
 * with explicit shape metadata, allowing it to represent scalars, vectors, matrices,
 * or arbitrary N-dimensional arrays. It is designed for memory efficiency, modularity,
 * and seamless interoperability across GPU and CPU environments.
 *
 * <h2>Integration Roles</h2>
 * <ul>
 *   <li><b>CuBridge:</b> Executes numerical and GPU-accelerated operations using {@code Tensor} operands.</li>
 *   <li><b>DataBridge:</b> Converts structured datasets (e.g., CSV, XLS, WAV) into tensor representations.</li>
 *   <li><b>JANET:</b> Manages layer input/output tensors during neural network forward and backward passes.</li>
 *   <li><b>ExBridge:</b> Uses tensor values as graphical data sources for plots and visual outputs.</li>
 * </ul>
 *
 * <h2>Core Features</h2>
 * <ul>
 *   <li>Multiple constructors for scalars, 1D/ND arrays, and file-based loading (CSV/XLS/WAV)</li>
 *   <li>Shape-aware printing with recursive visualization</li>
 *   <li>Randomized tensor generation (uniform, Gaussian, normal distributions)</li>
 *   <li>Common patterns: {@code zeros}, {@code ones}, {@code eye}, {@code arange}, {@code linspace}</li>
 *   <li>Structural utilities: {@code reshape}, {@code flatten}, {@code slice}, {@code stack}</li>
 *   <li>Data inspection tools: {@code head}, {@code printData}, {@code printSize}, {@code toArray}</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a 2D tensor of ones (shape: 3×3)
 * Tensor t = Tensor.ones(3, 3);
 *
 * // Flatten and inspect
 * Tensor flat = Tensor.flatten(t);
 * flat.printData();
 *
 * // Load data from CSV file
 * Tensor csv = new Tensor("csv", "data/input.csv");
 * }</pre>
 *
 * <h2>Function Overview</h2>
 * <ul>
 *   <li><b>Creation:</b> filled, zeros, ones, rand, randn, eye, arange, linspace</li>
 *   <li><b>Transformation:</b> reshape, flatten, slice, stack, vstack, hstack</li>
 *   <li><b>Inspection:</b> printData, printSize, head, getShape, getSize, getAxis</li>
 *   <li><b>Conversion:</b> toArray, fromFile, loadTable, loadWav</li>
 * </ul>
 *
 * @author  배준호
 * @version 1.0
 * @since   CuBridge v1.0 / JANET Core Integration
 */
public class Tensor {//바이트단위 wav 불러오기 추가
	private float[] data = null;
	private int[] shape = null;
	private int len = 0;

	private void print(String str) {
		System.out.println(str);
	}

	/**
	 * Calculates the total number of elements implied by the given shape.
	 *
	 * @return the total number of elements (product of dimensions)
	 */
	private int getLenFromShape() {
		int size = 1;
		for (var tmp : shape)
			size *= tmp;
		return size;
	}

	/**
	 * Constructs an empty Tensor with no data or defined shape.
	 *
	 * <p>
	 * This constructor creates an uninitialized tensor with:
	 * <ul>
	 *   <li>{@code data = null}</li>
	 *   <li>{@code shape = null}</li>
	 *   <li>{@code len = 0}</li>
	 * </ul>
	 *
	 * It serves as a placeholder for deferred initialization,
	 * commonly used in dynamic tensor creation or function results.
	 * </p>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = new Tensor();
	 * // t has no data until filled or reshaped later
	 * }</pre>
	 */
	public Tensor() {
	}

	/**
	 * Constructs a 1-dimensional Tensor from a given array of floats.
	 *
	 * <ul>
	 *   <li>The resulting tensor has shape {@code (N)}, where {@code N} is the length of the input array.</li>
	 *   <li>The input array is deep-copied to preserve immutability of internal data.</li>
	 *   <li>Useful for creating quick 1D vectors or constants.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = new Tensor(1.0f, 2.0f, 3.0f);
	 * // shape = [3]
	 * }</pre>
	 *
	 * @param data one-dimensional array of float values
	 */
	public Tensor(float... data) {
		this.shape = new int[] { data.length };
		this.data = data.clone();
		this.len = data.length;
	}

	/**
	 * Constructs a Tensor with explicit data and shape definitions.
	 *
	 * <ul>
	 *   <li>Validates whether the total number of elements in {@code data}
	 *       matches the product of the provided shape dimensions.</li>
	 *   <li>If valid, the given shape is applied directly.</li>
	 *   <li>If a mismatch is detected, a warning is printed and the shape
	 *       is automatically reset to a 1D vector {@code (N,)}.</li>
	 *   <li>All input arrays are deep-copied to maintain internal consistency.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * float[] arr = {1f, 2f, 3f, 4f, 5f, 6f};
	 * Tensor t = new Tensor(arr, 2, 3);
	 * // shape = [2, 3]
	 *
	 * // If shape mismatch:
	 * Tensor t2 = new Tensor(arr, 2, 4);
	 * // Warning → shape automatically set to [6]
	 * }</pre>
	 *
	 * @param data  flat data array (copied internally)
	 * @param shape intended tensor shape dimensions
	 */
	public Tensor(float[] data, int... shape) {
		this.shape = shape.clone();
		this.data = data.clone();
		this.len = data.length;

		if (len != getLenFromShape()) {
			System.err.println("Warning: Shape mismatch detected. Overriding shape to 1D.");
			this.shape = new int[] { len };
		}
	}

	/**
	 * Constructs a Tensor from a 2D float array.
	 *
	 * <ul>
	 *   <li>The input is assumed to represent a matrix-like structure, where
	 *       {@code data.length} is the number of rows and {@code data[0].length} is the number of columns.</li>
	 *   <li>The tensor is stored in row-major order, flattening the 2D structure into a 1D array.</li>
	 *   <li>The resulting tensor has shape (rows, cols).</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * float[][] arr = {
	 *     {1.0f, 2.0f, 3.0f},
	 *     {4.0f, 5.0f, 6.0f}
	 * };
	 * Tensor t = new Tensor(arr);
	 * // shape = [2, 3]
	 * }</pre>
	 *
	 * @param data a 2D array of float values representing rows and columns
	 */
	public Tensor(float[][] data){
		this.shape = new int[] { data.length, data[0].length };
		this.len = data.length * data[0].length;
		this.data = new float[this.len];

		int cols = data[0].length;

		for(int i = 0; i < this.len; i++)
			this.data[i] = data[i / cols][i % cols];
	}

	/**
	 * Loads a Tensor from a CSV file.
	 *
	 * <ul>
	 *   <li>The file must be composed of numeric values separated by commas (CSV format).</li>
	 *   <li>Each line represents one row, and each value represents one column.</li>
	 *   <li>Empty cells are automatically replaced with {@code 0.0}.</li>
	 *   <li>The resulting tensor has shape {@code (rows, columns)}.</li>
	 *   <li>The first line is treated as data (not header), ensuring full matrix loading.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * // data.csv
	 * // 1.0,2.0,3.0
	 * // 4.0,5.0,6.0
	 * Tensor t = new Tensor("data.csv");
	 * // shape = [2, 3]
	 * }</pre>
	 *
	 * @param path file path to the CSV file
	 */
	public Tensor(String path) {
		ArrayList<Float> tmpList = new ArrayList<Float>();
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
			tmp = line.split(",", -1);

			row = tmp.length;

			while ((line = sc.nextLine()) != null) {
				col++;
				tmp = line.split(",", -1);

				for (int r = 0; r < row; r++)
					if (tmp[r].equals(""))
						tmpList.add(0.0f);
					else
						tmpList.add(Float.parseFloat(tmp[r]));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NoSuchElementException e) {
			if (sc != null) sc.close();
		}

		this.len = tmpList.size();
		this.data = new float[this.len];
		this.shape = new int[] { col, row };

		for (int idx = 0; idx < this.len; idx++)
			this.data[idx] = tmpList.get(idx);
	}

	/**
	 * Constructs a Tensor by automatically determining the loading method based on type.
	 *
	 * <ul>
	 *   <li>If {@code type} is <b>"csv"</b>, loads a CSV file as a 2D tensor.</li>
	 *   <li>If {@code type} starts with <b>"wav"</b>:
	 *     <ul>
	 *       <li>"wav" → loads audio using default settings (16kHz, 32-bit).</li>
	 *       <li>"wav_16000_32" → loads audio resampled to 16kHz, 32-bit.</li>
	 *       <li>Folder path supported: all WAVs padded to the longest length.</li>
	 *     </ul>
	 *   </li>
	 *   <li>Unsupported types produce an empty tensor with a warning.</li>
	 * </ul>
	 *
	 * <h3>Examples</h3>
	 * <pre>{@code
	 * Tensor t1 = new Tensor("csv", "data/train.csv");   // shape = [rows, cols]
	 * Tensor t2 = new Tensor("wav", "audio.wav");        // default 16kHz, 32-bit
	 * Tensor t3 = new Tensor("wav_44100_16", "folder/"); // resampled to 44.1kHz, 16-bit
	 * }</pre>
	 */
	public Tensor(String type, String path) {
		type = type.toLowerCase().trim();

		if ("csv".equals(type)) {
			loadTable(path);
		} else if ("wav".equals(type)) {
			loadWav(path, 16000, 32); // 기본값
		} else if (type.startsWith("wav_")) {
			String[] parts = type.split("_");
			int targetRate = 16000;
			int targetBits = 32;

			try {
				if (parts.length >= 2) targetRate = Integer.parseInt(parts[1]);
				if (parts.length >= 3) targetBits = Integer.parseInt(parts[2]);
			} catch (NumberFormatException e) {
				System.err.printf("[Tensor] Invalid WAV format specifier: %s (using defaults)%n", type);
			}

			loadWav(path, targetRate, targetBits);
		} else {
			System.err.printf("[Tensor] Unsupported file type: %s%n", type);
			this.data = new float[0];
			this.shape = new int[]{0};
			this.len = 0;
		}
	}

	/**
	 * Constructs a Tensor by converting a 2D string array to a numerical tensor.
	 *
	 * <ul>
	 *   <li>Uses a default normalization factor of {@code 1} (no scaling applied).</li>
	 *   <li>All string values are parsed into {@code float} precision values.</li>
	 *   <li>The input must be rectangular (all rows of equal length).</li>
	 *   <li>Stored in column-major order: {@code data[c][r]} becomes {@code float[c * row + r]}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * String[][] s = {
	 *     {"1.0", "2.0", "3.0"},
	 *     {"4.0", "5.0", "6.0"}
	 * };
	 * Tensor t = new Tensor(s);
	 * // shape = [2, 3]
	 * }</pre>
	 *
	 * @param data the 2D string array representing numeric values
	 * @since 1.1
	 */
	public Tensor(String[][] data) {
		this(data, 1);
	}

	/**
	 * Constructs a Tensor by converting a 2D string array to a normalized numerical tensor.
	 *
	 * <ul>
	 *   <li>All string elements are parsed into {@code float} values and divided by {@code norm}.</li>
	 *   <li>Use this to scale raw string data (e.g., percentage normalization).</li>
	 *   <li>The input must be rectangular (each row of equal length).</li>
	 *   <li>Stored in column-major order: {@code data[c][r]} becomes {@code float[c * row + r]}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * String[][] s = {
	 *     {"10", "20"},
	 *     {"30", "40"}
	 * };
	 * Tensor t = new Tensor(s, 10f);
	 * // shape = [2, 2]
	 * // data = [1.0, 2.0, 3.0, 4.0]
	 * }</pre>
	 *
	 * @param data the 2D string array representing numeric values
	 * @param norm the normalization factor to divide each value by
	 * @since 1.1
	 */
	public Tensor(String[][] data, float norm) {
		int col = data.length;
		int row = data[0].length;
		this.data = new float[col * row];
		this.shape = new int[]{col, row};
		this.len = col * row;

		for (int c = 0; c < col; c++)
			for (int r = 0; r < row; r++)
				this.data[c * row + r] = Float.parseFloat(data[c][r]) / norm;
	}

	/**
	 * Returns a deep copy of the internal data array.
	 *
	 * <ul>
	 *   <li>Copies the tensor's internal flat array ({@code data}) to prevent external modification.</li>
	 *   <li>The returned array is a full-length, 1D representation of all tensor elements.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.ones(2, 3);
	 * float[] arr = t.toArray();
	 * // arr = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	 * }</pre>
	 *
	 * @return cloned array containing all tensor values
	 */
	public float[] toArray() {
		return this.data.clone();
	}

	/**
	 * Returns the total number of elements contained in this tensor.
	 *
	 * <ul>
	 *   <li>Equivalent to the product of all shape dimensions.</li>
	 *   <li>For example, a tensor of shape (2, 3, 4) has {@code 2 × 3 × 4 = 24} elements.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = new Tensor(new float[]{1, 2, 3, 4}, 2, 2);
	 * int size = t.getSize();
	 * // size = 4
	 * }</pre>
	 *
	 * @return the total number of elements in the tensor
	 */
	public int getSize() {
		return this.len;
	}

	/**
	 * Returns a copy of the tensor's shape array.
	 *
	 * <ul>
	 *   <li>The returned array represents the size of the tensor along each axis.</li>
	 *   <li>The array is cloned to prevent modification of the internal shape data.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = new Tensor(new float[]{1, 2, 3, 4}, 2, 2);
	 * int[] shape = t.getShape();
	 * // shape = [2, 2]
	 * }</pre>
	 *
	 * @return cloned array representing tensor dimensions
	 */
	public int[] getShape() {
		return this.shape.clone();
	}

	/**
	 * Returns the number of axes (dimensions) of this tensor.
	 *
	 * <ul>
	 *   <li>For example, a vector has 1 axis, a matrix has 2 axes, and a 3D tensor has 3 axes.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t1 = new Tensor(new float[]{1, 2, 3});
	 * // shape = [3], getAxis() = 1
	 *
	 * Tensor t2 = Tensor.ones(2, 3, 4);
	 * // shape = [2, 3, 4], getAxis() = 3
	 * }</pre>
	 *
	 * @return the number of dimensions (axes) in the tensor
	 */
	public int getAxis() {
		return this.shape.length;
	}

	/**
	 * Prints the size (shape) of the tensor in a flat format.
	 *
	 * <ul>
	 *   <li>Each dimension is printed sequentially in one line.</li>
	 *   <li>For example, a tensor with shape (3, 4, 5) prints as {@code 3 4 5}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.ones(3, 4, 5);
	 * t.printSize();
	 * // Output: 3 4 5
	 * }</pre>
	 */
	public void printSize() {
		for (int i = 0; i < shape.length; i++)
			System.out.print(this.shape[i] + " ");
		System.out.println();
	}

	/**
	 * Prints the tensor data with shape information.
	 *
	 * <ul>
	 *   <li>Displays the tensor's shape followed by formatted numerical data.</li>
	 *   <li>If the tensor is empty, prints {@code (empty tensor)}.</li>
	 *   <li>Multi-dimensional tensors are recursively printed in nested form.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.arange(0, 6, 1);
	 * Tensor reshaped = Tensor.reshape(t, 2, 3);
	 * reshaped.printData();
	 * // Output:
	 * // Tensor(shape=[2, 3]):
	 * //   [ 0.000, 1.000, 2.000 ]
	 * //   [ 3.000, 4.000, 5.000 ]
	 * }</pre>
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
	 * Prints a concise preview of this tensor’s contents to the console.
	 *
	 * <p>This method emulates the behavior of Python’s {@code head()} or slicing preview
	 * ({@code tensor[:6]}) and automatically adapts its output format according to the
	 * tensor’s dimensionality.</p>
	 *
	 * <h2>Display Rules by Dimension</h2>
	 * <ul>
	 *   <li><b>1D tensor:</b> Prints up to the first 6 elements in a single row.</li>
	 *   <li><b>2D tensor:</b> Prints up to the first 6 rows; all columns are shown per row.</li>
	 *   <li><b>3D tensor:</b> Prints up to the first 2 blocks along the first axis,
	 *       each block containing multiple rows and columns.</li>
	 *   <li><b>4D or higher:</b> Only the shape is printed with a message indicating
	 *       that preview for high-rank tensors is omitted.</li>
	 * </ul>
	 *
	 * <p>If the tensor is empty or uninitialized, prints {@code Tensor(empty)}.</p>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * // 1D example
	 * Tensor t1 = Tensor.arange(0f, 10f, 1f);
	 * t1.head();
	 * // Output:
	 * // Tensor(shape=[10], head=
	 * //  [ 0.000, 1.000, 2.000, 3.000, 4.000, 5.000, ...]
	 * // )
	 *
	 * // 2D example
	 * Tensor t2 = Tensor.arange(0f, 12f, 1f);
	 * Tensor r2 = Tensor.reshape(t2, 4, 3);
	 * r2.head();
	 * // Output:
	 * // Tensor(shape=[4, 3], head=
	 * //  [ 0.000, 1.000, 2.000 ]
	 * //  [ 3.000, 4.000, 5.000 ]
	 * //  [ 6.000, 7.000, 8.000 ]
	 * //  [ 9.000,10.000,11.000 ]
	 * // )
	 *
	 * // 3D example
	 * Tensor t3 = Tensor.arange(0f, 24f, 1f);
	 * Tensor r3 = Tensor.reshape(t3, 2, 3, 4);
	 * r3.head();
	 * // Output:
	 * // Tensor(shape=[2, 3, 4], head=
	 * //  Block 0:
	 * //   [ 0.000, 1.000, 2.000, 3.000 ]
	 * //   [ 4.000, 5.000, 6.000, 7.000 ]
	 * //   [ 8.000, 9.000,10.000,11.000 ]
	 * //
	 * //  Block 1:
	 * //   [12.000,13.000,14.000,15.000 ]
	 * //   [16.000,17.000,18.000,19.000 ]
	 * //   [20.000,21.000,22.000,23.000 ]
	 * // )
	 * }</pre>
	 */
	public void head() {
		int dim = shape.length;
		if (dim == 0 || len == 0) {
			System.out.println("Tensor(empty)");
			return;
		}

		System.out.println("Tensor(shape=" + Arrays.toString(shape) + ", head=");

		if (dim == 1) {
			// 1D: 앞 6개
			int n = Math.min(6, len);
			System.out.print(" [");
			for (int i = 0; i < n; i++) {
				System.out.printf("%6.3f", data[i]);
				if (i < n - 1) System.out.print(", ");
			}
			if (len > n) System.out.print(", ...");
			System.out.println(" ]");
		}

		else if (dim == 2) {
			// 2D: 앞 6행
			int rows = shape[0];
			int cols = shape[1];
			int rowCount = Math.min(6, rows);
			for (int i = 0; i < rowCount; i++) {
				System.out.print("  [");
				for (int j = 0; j < cols; j++) {
					System.out.printf("%6.3f", data[i * cols + j]);
					if (j < cols - 1) System.out.print(", ");
				}
				System.out.println(" ]");
			}
			if (rows > rowCount) System.out.println("  ...");
		}

		else if (dim == 3) {
			// 3D: 앞 2블록
			int d1 = shape[0], d2 = shape[1], d3 = shape[2];
			int blockCount = Math.min(2, d1);
			int stride2 = d2 * d3;
			for (int b = 0; b < blockCount; b++) {
				System.out.println(" Block " + b + ":");
				for (int i = 0; i < d2; i++) {
					System.out.print("  [");
					for (int j = 0; j < d3; j++) {
						int idx = b * stride2 + i * d3 + j;
						System.out.printf("%6.3f", data[idx]);
						if (j < d3 - 1) System.out.print(", ");
					}
					System.out.println(" ]");
				}
				if (b < blockCount - 1) System.out.println();
			}
			if (d1 > blockCount) System.out.println("  ...");
		}

		else {
			// 4차원 이상은 shape만 출력
			System.out.println("  (head preview not implemented for rank ≥ 4)");
		}

		System.out.println(")");
	}

	/**
	 * Returns a string representation of the tensor's shape.
	 *
	 * <ul>
	 *   <li>Displays only the shape information, e.g., {@code shape = [2, 3, 4]}.</li>
	 *   <li>Intended for simple summaries or debugging logs.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.zeros(2, 3, 4);
	 * System.out.println(t);
	 * // Output: shape = [2, 3, 4]
	 * }</pre>
	 *
	 * @return a string describing the tensor’s shape
	 */
	@Override
	public String toString() {
		return "shape = " + Arrays.toString(shape);
	}

	/**
	 * Saves the tensor in a human-readable text format (.txt or .tensor).
	 * The file can be loaded in Python using numpy.loadtxt().
	 */
	public void save(String path) {
		try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(path)))) {
			// ----- Header -----
			pw.println("# Tensor Save File (CuBridge)");
			pw.println("# shape=" + Arrays.toString(this.shape));
			pw.println("# len=" + this.len);
			pw.println("# dtype=float32");
			pw.println();

			// ----- Data -----
			if (this.shape.length == 1) {
				// 1D
				for (int i = 0; i < this.len; i++)
					pw.printf(Locale.US, "%.8f%n", this.data[i]);
			} else if (this.shape.length == 2) {
				// 2D
				int rows = this.shape[0];
				int cols = this.shape[1];
				for (int r = 0; r < rows; r++) {
					for (int c = 0; c < cols; c++) {
						pw.printf(Locale.US, "%.8f", this.data[r * cols + c]);
						if (c < cols - 1) pw.print(",");
					}
					pw.println();
				}
			} else {
				// N-D → flatten 출력
				for (int i = 0; i < this.len; i++)
					pw.printf(Locale.US, "%.8f%n", this.data[i]);
			}
			pw.flush();
			System.out.println("[Tensor] Saved as text → " + path);
		} catch (IOException e) {
			System.err.println("[Tensor] Save failed: " + e.getMessage());
		}
	}

	/**
	 * Loads a tensor saved in the same text format by save(path).
	 */
	public static Tensor load(String path) {
		ArrayList<Float> values = new ArrayList<>();
		int[] shape = null;

		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			String line;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#")) {
					if (line.startsWith("# shape=")) {
						// Extract shape info
						String shapeStr = line.substring(line.indexOf('=') + 1)
								.replace("[", "")
								.replace("]", "")
								.trim();
						String[] parts = shapeStr.split(",");
						shape = new int[parts.length];
						for (int i = 0; i < parts.length; i++)
							shape[i] = Integer.parseInt(parts[i].trim());
					}
					continue;
				}

				// Data line
				for (String s : line.split(",")) {
					if (!s.isEmpty()) values.add(Float.parseFloat(s));
				}
			}

			float[] arr = new float[values.size()];
			for (int i = 0; i < arr.length; i++) arr[i] = values.get(i);

			if (shape == null) shape = new int[]{arr.length};
			System.out.println("[Tensor] Loaded from text → " + path);
			return new Tensor(arr, shape);

		} catch (Exception e) {
			System.err.println("[Tensor] Load failed: " + e.getMessage());
			return new Tensor(new float[0]);
		}
	}

	/**
	 * Creates a tensor filled with a specified constant value.
	 *
	 * <ul>
	 *   <li>All elements are set to the same constant {@code value}.</li>
	 *   <li>The tensor's shape is defined by the provided {@code shape} array.</li>
	 *   <li>Useful for initializing tensors with bias values, masks, or fixed constants.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.filled(5.0f, 2, 3);
	 * // shape = [2, 3]
	 * // data = [5, 5, 5, 5, 5, 5]
	 * }</pre>
	 *
	 * @param value the constant value to fill
	 * @param shape the desired tensor shape
	 * @return a new tensor filled with {@code value}
	 */
	public static Tensor filled(float value, int... shape) {
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new float[t.len];

		for (int i = 0; i < t.len; i++)
			t.data[i] = value;

		return t;
	}

	/**
	 * Creates a tensor filled with uniformly distributed random values in the range [0.0, 1.0).
	 *
	 * <ul>
	 *   <li>Each element is a pseudo-random float generated by {@link Math#random()}.</li>
	 *   <li>Distribution is uniform across the interval [0.0, 1.0).</li>
	 *   <li>Useful for initializing weight tensors or sampling noise.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.rand(2, 2);
	 * // Example output:
	 * // shape = [2, 2]
	 * // data ≈ [0.12, 0.83, 0.47, 0.29]
	 * }</pre>
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor with uniformly distributed random values
	 */
	public static Tensor rand(int... shape) {
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new float[t.len];

		for (int i = 0; i < t.len; i++)
			t.data[i] = (float) Math.random();

		return t;
	}

	/**
	 * Creates a tensor filled with standard normally distributed random values.
	 *
	 * <ul>
	 *   <li>Samples are drawn from a normal distribution with mean = 0 and standard deviation = 1.</li>
	 *   <li>Uses {@link Random#nextGaussian()} internally for generation.</li>
	 *   <li>Ideal for weight initialization in deep learning models.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.randn(3, 3);
	 * // Example output:
	 * // shape = [3, 3]
	 * // data ≈ [-0.52, 0.11, 1.24, -0.33, ...]
	 * }</pre>
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor with standard normal random values
	 */
	public static Tensor randn(int... shape) {
		Random random = new Random();
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new float[t.len];

		for (int i = 0; i < t.len; i++)
			t.data[i] = (float) random.nextGaussian();

		return t;
	}

	/**
	 * Creates a tensor filled with normally distributed random values
	 * using the specified mean and standard deviation.
	 *
	 * <ul>
	 *   <li>Samples are drawn from a Gaussian distribution with parameters:
	 *       {@code mean} and {@code std}.</li>
	 *   <li>Each value = {@code mean + std × N(0, 1)}.</li>
	 *   <li>Useful for generating scaled random tensors for initialization or simulation.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.randn(0.0f, 0.1f, 2, 4);
	 * // Example output:
	 * // shape = [2, 4]
	 * // data ≈ [0.03, -0.07, 0.12, -0.01, ...]
	 * }</pre>
	 *
	 * @param mean  the mean of the normal distribution
	 * @param std   the standard deviation of the normal distribution
	 * @param shape the desired tensor shape
	 * @return a new tensor with normally distributed random values
	 */
	public static Tensor randn(float mean, float std, int... shape) {
		Random random = new Random();
		Tensor t = new Tensor();
		t.shape = shape.clone();
		t.len = getLenFromShape(shape);
		t.data = new float[t.len];

		for (int i = 0; i < t.len; i++)
			t.data[i] = (float) (std * random.nextGaussian() + mean);

		return t;
	}

	/**
	 * Creates a tensor filled entirely with zeros.
	 *
	 * <ul>
	 *   <li>All elements are set to {@code 0.0f}.</li>
	 *   <li>Equivalent to {@code Tensor.filled(0.0f, shape)}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.zeros(2, 3);
	 * // shape = [2, 3]
	 * // data = [0, 0, 0, 0, 0, 0]
	 * }</pre>
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor filled with zeros
	 */
	public static Tensor zeros(int... shape) {
		return filled(0.0f, shape);
	}

	/**
	 * Creates a tensor filled entirely with ones.
	 *
	 * <ul>
	 *   <li>All elements are set to {@code 1.0f}.</li>
	 *   <li>Equivalent to {@code Tensor.filled(1.0f, shape)}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.ones(2, 2);
	 * // shape = [2, 2]
	 * // data = [1, 1, 1, 1]
	 * }</pre>
	 *
	 * @param shape the desired tensor shape
	 * @return a new tensor filled with ones
	 */
	public static Tensor ones(int... shape) {
		return filled(1.0f, shape);
	}

	/**
	 * Creates an identity matrix of size {@code n × n}.
	 *
	 * <ul>
	 *   <li>Diagonal elements are {@code 1.0}, all others are {@code 0.0}.</li>
	 *   <li>The resulting tensor has shape {@code [n, n]}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.eye(3);
	 * // shape = [3, 3]
	 * // data =
	 * // [1, 0, 0,
	 * //  0, 1, 0,
	 * //  0, 0, 1]
	 * }</pre>
	 *
	 * @param n the number of rows and columns
	 * @return a square identity matrix tensor
	 */
	public static Tensor eye(int n) {
		Tensor t = new Tensor();
		t.shape = new int[] { n, n };
		t.len = n * n;
		t.data = new float[t.len];

		for (int i = 0; i < n; i++)
			t.data[i * n + i] = 1.0f;

		return t;
	}

	/**
	 * Creates a 1D tensor with values starting from {@code start}
	 * and increasing by {@code step} up to (but not including) {@code end}.
	 *
	 * <ul>
	 *   <li>Equivalent to Python’s {@code numpy.arange()} behavior.</li>
	 *   <li>All values are linearly spaced by {@code step}.</li>
	 *   <li>If {@code step <= 0} or {@code start >= end}, an empty tensor is returned.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.arange(0f, 5f, 1f);
	 * // shape = [5]
	 * // data = [0, 1, 2, 3, 4]
	 * }</pre>
	 *
	 * @param start the starting value (inclusive)
	 * @param end   the end value (exclusive)
	 * @param step  the increment step (must be > 0)
	 * @return a 1D tensor containing sequential values
	 */
	public static Tensor arange(float start, float end, float step) {
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
		t.data = new float[length];

		for (int i = 0; i < length; i++)
			t.data[i] = start + i * step;

		return t;
	}

	/**
	 * Creates a 1D tensor with {@code num} evenly spaced values
	 * between {@code start} and {@code end} (inclusive).
	 *
	 * <ul>
	 *   <li>Equivalent to Python’s {@code numpy.linspace()} behavior.</li>
	 *   <li>If {@code num == 1}, the tensor contains only the {@code start} value.</li>
	 *   <li>If {@code num <= 0}, an empty tensor is returned.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.linspace(0f, 1f, 5);
	 * // shape = [5]
	 * // data = [0.00, 0.25, 0.50, 0.75, 1.00]
	 * }</pre>
	 *
	 * @param start the starting value (inclusive)
	 * @param end   the end value (inclusive)
	 * @param num   number of values to generate (must be ≥ 1)
	 * @return a 1D tensor with evenly spaced values
	 */
	public static Tensor linspace(float start, float end, int num) {
		Tensor t = new Tensor();

		if (num <= 0) {
			System.out.printf("Error: Tensor.linspace() - num (%d) must be positive\n", num);
			return t;
		}

		t.shape = new int[] { num };
		t.len = num;
		t.data = new float[num];

		if (num == 1) {
			t.data[0] = start;
			return t;
		}

		float step = (end - start) / (num - 1);

		for (int i = 0; i < num; i++)
			t.data[i] = start + i * step;

		return t;
	}

	/**
	 * Reshapes a given tensor into a new shape.
	 *
	 * <ul>
	 *   <li>Returns a new tensor with the same data but a different shape.</li>
	 *   <li>The total number of elements must remain the same.</li>
	 *   <li>If the new shape's size does not match, an {@link IllegalArgumentException} is thrown.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.arange(0f, 6f, 1f);
	 * Tensor reshaped = Tensor.reshape(t, 2, 3);
	 * // shape = [2, 3]
	 * // data = [0, 1, 2, 3, 4, 5]
	 * }</pre>
	 *
	 * @param src      the original tensor
	 * @param newShape the desired new shape
	 * @return a new tensor reshaped to {@code newShape}
	 * @throws IllegalArgumentException if the total size does not match
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
	 * <ul>
	 *   <li>Converts any N-dimensional tensor into a 1D tensor while preserving data order.</li>
	 *   <li>Equivalent to {@code Tensor.reshape(src, src.getSize())}.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.eye(3);
	 * Tensor flat = Tensor.flatten(t);
	 * // shape = [9]
	 * // data = [1, 0, 0, 0, 1, 0, 0, 0, 1]
	 * }</pre>
	 *
	 * @param src the tensor to flatten
	 * @return a 1D tensor containing all elements
	 */
	public static Tensor flatten(Tensor src) {
		return reshape(src, src.getSize());
	}

	/**
	 * Splits a tensor into multiple sub-tensors along the specified axis using explicit index boundaries.
	 *
	 * <ul>
	 *   <li>Each segment is defined by consecutive pairs of indices in the {@code indices} array.</li>
	 *   <li>For example, {@code indices = {0, 2, 5}} produces two tensors:
	 *       {@code [0,2)} and {@code [2,5)}.</li>
	 *   <li>All resulting sub-tensors preserve the order and shape of other axes.</li>
	 *   <li>If any range is invalid or out of bounds, an exception is thrown.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.arange(0f, 6f, 1f);
	 * Tensor reshaped = Tensor.reshape(t, 2, 3);
	 * // shape = [2, 3]
	 *
	 * Tensor[] parts = Tensor.slice(reshaped, 1, new int[]{0, 2, 3});
	 * // parts[0].shape = [2, 2]
	 * // parts[1].shape = [2, 1]
	 * }</pre>
	 *
	 * @param src     the source tensor to split
	 * @param axis    the axis along which to perform the split
	 * @param indices the array of split boundaries (must be sorted, length ≥ 2)
	 * @return an array of sub-tensors defined by {@code indices}
	 * @throws IllegalArgumentException if {@code indices} are invalid or out of range
	 */
	public static Tensor[] slice(Tensor src, int axis, int[] indices) {
		int rank = src.shape.length;
		axis = normalizeAxis(axis, rank);

		int partCount = indices.length - 1;
		Tensor[] results = new Tensor[partCount];

		int stride = 1;
		for (int i = axis + 1; i < rank; i++) stride *= src.shape[i];
		int block = src.shape[axis] * stride;

		int outPos, i, n, start, end, copySize, outLen;
		int[] newShape;
		float[] outData;

		for (int p = 0; p < partCount; p++) {
			start = indices[p];
			end = indices[p + 1];
			if (start < 0 || end > src.shape[axis] || start >= end)
				throw new IllegalArgumentException("Invalid slice range");

			newShape = src.shape.clone();
			newShape[axis] = end - start;

			outLen = 1;
			for (i = 0; i < rank; i++) outLen *= newShape[i];
			outData = new float[outLen];

			copySize = (end - start) * stride;
			outPos = 0;
			i = 0;
			n = src.data.length;
			while (i < n) {
				System.arraycopy(src.data, i + start * stride, outData, outPos, copySize);
				outPos += copySize;
				i += block;
			}
			results[p] = new Tensor(outData, newShape);
		}
		return results;
	}

	/**
	 * Splits a tensor into equal-sized parts along the specified axis.
	 *
	 * <ul>
	 *   <li>The length of the selected axis must be divisible by {@code parts}.</li>
	 *   <li>Each resulting tensor contains an equal segment of the original tensor.</li>
	 *   <li>This method is equivalent to calling {@link #slice(Tensor, int, int[])} internally.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor t = Tensor.arange(0f, 8f, 1f);
	 * Tensor reshaped = Tensor.reshape(t, 2, 4);
	 * // shape = [2, 4]
	 *
	 * Tensor[] halves = Tensor.slice(reshaped, 1, 2);
	 * // halves[0].shape = [2, 2]
	 * // halves[1].shape = [2, 2]
	 * }</pre>
	 *
	 * @param src   the source tensor to split
	 * @param axis  the axis along which to perform the split
	 * @param parts the number of equal segments to create
	 * @return an array of sub-tensors of equal size
	 * @throws IllegalArgumentException if the axis length is not divisible by {@code parts}
	 */
	public static Tensor[] slice(Tensor src, int axis, int parts) {
		int rank = src.shape.length;
		axis = normalizeAxis(axis, rank);

		int dimSize = src.shape[axis];
		if (dimSize % parts != 0)
			throw new IllegalArgumentException("Dimension " + dimSize + " not divisible by parts=" + parts);

		int step = dimSize / parts;
		int[] indices = new int[parts + 1];
		for (int i = 0; i <= parts; i++) indices[i] = i * step;

		return slice(src, axis, indices);  // reuse main implementation
	}

	/**
	 * Vertically stacks multiple tensors along the second-to-last axis.
	 *
	 * <ul>
	 *   <li>All input tensors must share the same shape except for the stacking axis.</li>
	 *   <li>The specified axis (rank − 2) is expanded to the sum of all corresponding dimensions.</li>
	 *   <li>The resulting tensor merges all data vertically (like row-wise concatenation).</li>
	 *   <li>If the shapes are inconsistent, an {@link IllegalArgumentException} is thrown.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor a = Tensor.ones(2, 3);
	 * Tensor b = Tensor.ones(1, 3);
	 * Tensor v = Tensor.vstack(a, b);
	 * // shape = [3, 3]
	 * }</pre>
	 *
	 * @param tensors the array of tensors to stack vertically
	 * @return a new tensor with combined data along the vertical axis
	 * @throws IllegalArgumentException if tensor shapes are incompatible
	 */
	public static Tensor vstack(Tensor... tensors) {
		int rank = tensors[0].shape.length;
		int axis = rank - 2;

		int sumAxis = 0;
		for (Tensor t : tensors) {
			if (t.shape.length != rank) throw new IllegalArgumentException("Rank mismatch");
			for (int i = 0; i < rank; i++) {
				if (i == axis) continue;
				if (t.shape[i] != tensors[0].shape[i])
					throw new IllegalArgumentException("Shape mismatch for vstack");
			}
			sumAxis += t.shape[axis];
		}

		int[] newShape = tensors[0].shape.clone();
		newShape[axis] = sumAxis;

		int totalLen = 0;
		for (Tensor t : tensors) totalLen += t.data.length;
		float[] outData = new float[totalLen];

		int offset = 0;
		for (Tensor t : tensors) {
			System.arraycopy(t.data, 0, outData, offset, t.data.length);
			offset += t.data.length;
		}

		return new Tensor(outData, newShape);
	}

	/**
	 * Horizontally stacks multiple tensors along the last axis.
	 *
	 * <ul>
	 *   <li>All input tensors must share the same shape except for the last axis.</li>
	 *   <li>The last axis is expanded to the sum of all input tensors’ widths.</li>
	 *   <li>This operation corresponds to column-wise concatenation.</li>
	 *   <li>If the shapes are inconsistent, an {@link IllegalArgumentException} is thrown.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor a = Tensor.ones(2, 2);
	 * Tensor b = Tensor.zeros(2, 1);
	 * Tensor h = Tensor.hstack(a, b);
	 * // shape = [2, 3]
	 * }</pre>
	 *
	 * @param tensors the array of tensors to stack horizontally
	 * @return a new tensor with combined data along the horizontal axis
	 * @throws IllegalArgumentException if tensor shapes are incompatible
	 */
	public static Tensor hstack(Tensor... tensors) {
		int rank = tensors[0].shape.length;
		int axis = rank - 1;

		int sumAxis = 0;
		for (Tensor t : tensors) {
			if (t.shape.length != rank) throw new IllegalArgumentException("Rank mismatch");
			for (int i = 0; i < rank; i++) {
				if (i == axis) continue;
				if (t.shape[i] != tensors[0].shape[i])
					throw new IllegalArgumentException("Shape mismatch for hstack");
			}
			sumAxis += t.shape[axis];
		}

		int[] newShape = tensors[0].shape.clone();
		newShape[axis] = sumAxis;

		int rows = tensors[0].data.length / tensors[0].shape[axis];
		int outCols = newShape[axis];
		float[] outData = new float[rows * outCols];

		int row = 0, oStart, offset, cols;
		while (row < rows) {
			oStart = row * outCols;
			offset = oStart;
			for (Tensor t : tensors) {
				cols = t.shape[axis];
				System.arraycopy(t.data, row * cols, outData, offset, cols);
				offset += cols;
			}
			row++;
		}
		return new Tensor(outData, newShape);
	}

	/**
	 * Stacks a sequence of tensors along a new axis.
	 *
	 * <ul>
	 *   <li>Inserts a new dimension of size {@code arr.length} at the specified {@code axis}.</li>
	 *   <li>All input tensors must have exactly the same shape.</li>
	 *   <li>The resulting tensor will have one additional rank (dimension).</li>
	 *   <li>If any tensor shape differs, an {@link IllegalArgumentException} is thrown.</li>
	 * </ul>
	 *
	 * <h3>Example</h3>
	 * <pre>{@code
	 * Tensor a = Tensor.ones(2, 2);
	 * Tensor b = Tensor.zeros(2, 2);
	 * Tensor stacked = Tensor.stack(new Tensor[]{a, b}, 0);
	 * // shape = [2, 2, 2]
	 * }</pre>
	 *
	 * @param arr  the array of tensors to stack (all must share the same shape)
	 * @param axis the index at which to insert the new axis (0 ≤ axis ≤ rank of input tensor)
	 * @return a new tensor with one additional dimension containing all stacked tensors
	 * @throws IllegalArgumentException if input array is empty or tensors have different shapes
	 */
	public static Tensor stack(Tensor[] arr, int axis) {
		if (arr == null || arr.length == 0) {
			throw new IllegalArgumentException("Input array is empty.");
		}

		int[] baseShape = arr[0].getShape();
		int rank = baseShape.length;
		int N = arr.length;

		for (Tensor t : arr) {
			if (!Arrays.equals(t.getShape(), baseShape)) {
				throw new IllegalArgumentException("All tensors must have the same shape.");
			}
		}

		int[] newShape = new int[rank + 1];
		for (int i = 0; i < axis; i++) newShape[i] = baseShape[i];
		newShape[axis] = N;
		for (int i = axis; i < rank; i++) newShape[i + 1] = baseShape[i];

		int totalOut = 1;
		for (int s : newShape) totalOut *= s;
		float[] outData = new float[totalOut];

		int totalIn = arr[0].getSize();
		int[] coords = new int[rank];
		int[] outCoords = new int[rank + 1];

		for (int n = 0; n < N; n++) {
			float[] inData = arr[n].toArray();

			for (int idx = 0; idx < totalIn; idx++) {
				int tmp = idx;
				for (int d = rank - 1; d >= 0; d--) {
					coords[d] = tmp % baseShape[d];
					tmp /= baseShape[d];
				}

				for (int d = 0, j = 0; d < outCoords.length; d++) {
					if (d == axis) outCoords[d] = n;
					else outCoords[d] = coords[j++];
				}

				int flat = 0, stride = 1;
				for (int d = outCoords.length - 1; d >= 0; d--) {
					flat += outCoords[d] * stride;
					stride *= newShape[d];
				}

				outData[flat] = inData[idx];
			}
		}
		return new Tensor(outData, newShape);
	}

	/**
	 * Loads a tensor from a CSV file into a 2D numerical array.
	 * <p>
	 * Empty fields are replaced with 0.0, and non-numeric values cause
	 * the cell to be skipped with a warning.
	 */
	private void loadTable(String path) {
		ArrayList<float[]> lines = new ArrayList<>();
		int cols = -1;

		try (BufferedReader br = new BufferedReader(new FileReader(path), 8192)) {
			String line;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",", -1);
				if (cols == -1) cols = tokens.length;
				float[] row = new float[cols];
				for (int i = 0; i < cols; i++) {
					if (!tokens[i].isEmpty()) {
						try {
							row[i] = Float.parseFloat(tokens[i]);
						} catch (NumberFormatException e) {
							row[i] = 0.0f; // fallback
						}
					} else row[i] = 0.0f;
				}
				lines.add(row);
			}

			int rows = lines.size();
			this.shape = new int[]{rows, cols};
			this.len = rows * cols;
			this.data = new float[this.len];

			int idx = 0;
			for (float[] row : lines) {
				System.arraycopy(row, 0, this.data, idx, cols);
				idx += cols;
			}
		} catch (IOException e) {
			System.err.println("[Tensor] Failed to load CSV: " + e.getMessage());
			this.data = new float[0];
			this.shape = new int[]{0, 0};
			this.len = 0;
		}
	}

	private void loadWav(String path, int targetRate, int targetBits) {
		File f = new File(path);
		if (!f.exists()) {
			System.err.println("[Tensor] File not found: " + path);
			this.data = new float[0];
			this.shape = new int[]{0};
			this.len = 0;
			return;
		}

		if (f.isDirectory()) {
			loadWavFolder(f, targetRate, targetBits);
		} else {
			float[] wav = loadSingleWav(f, targetRate, targetBits);
			this.data = wav;
			this.shape = new int[]{wav.length};
			this.len = wav.length;
			System.out.printf("[Tensor] Loaded WAV: %s (%d samples @ %d Hz, %d-bit)%n",
					f.getName(), wav.length, targetRate, targetBits);
		}
	}

	private void loadWavFolder(File folder, int targetRate, int targetBits) {
		File[] files = folder.listFiles((d, n) -> n.toLowerCase().endsWith(".wav"));
		if (files == null || files.length == 0) {
			System.err.println("[Tensor] No WAV files found in: " + folder.getPath());
			this.data = new float[0];
			this.shape = new int[]{0, 0};
			this.len = 0;
			return;
		}

		ArrayList<float[]> waves = new ArrayList<>();
		int maxLen = 0;
		for (File f : files) {
			float[] wav = loadSingleWav(f, targetRate, targetBits);
			if (wav.length == 0) continue;
			waves.add(wav);
			maxLen = Math.max(maxLen, wav.length);
		}

		int fileCount = waves.size();
		this.data = new float[fileCount * maxLen];
		this.shape = new int[]{fileCount, maxLen};
		this.len = this.data.length;

		for (int i = 0; i < fileCount; i++) {
			float[] src = waves.get(i);
			System.arraycopy(src, 0, this.data, i * maxLen, src.length);
		}

		System.out.printf("[Tensor] Loaded %d WAVs (→ %d Hz, %d-bit, padded to %d samples)%n",
				fileCount, targetRate, targetBits, maxLen);
	}

	private float[] loadSingleWav(File file, int targetRate, int targetBits) {
		try (AudioInputStream ais = AudioSystem.getAudioInputStream(file)) {
			AudioFormat target = new AudioFormat(
					AudioFormat.Encoding.PCM_SIGNED, targetRate, targetBits,
					1, targetBits / 8, targetRate, false);
			AudioInputStream din = AudioSystem.getAudioInputStream(target, ais);

			ByteArrayOutputStream out = new ByteArrayOutputStream();
			byte[] buf = new byte[8192];
			int n;
			while ((n = din.read(buf)) != -1) out.write(buf, 0, n);

			byte[] bytes = out.toByteArray();
			int samples = bytes.length / (targetBits / 8);
			return decodeSamples(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN),
					targetBits, false, samples);
		} catch (Exception e) {
			return parseRiffManual(file, targetRate);
		}
	}

	private float[] parseRiffManual(File file, int targetRate) {
		try (FileChannel ch = FileChannel.open(file.toPath(), StandardOpenOption.READ)) {
			ByteBuffer buf4 = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(buf4); buf4.flip();
			String riffTag = new String(buf4.array(), "US-ASCII");
			if (!riffTag.equals("RIFF") && !riffTag.equals("RF64"))
				throw new IOException("Unsupported container: " + riffTag);
			ch.position(12);

			int fmtCode = 1, sampleRate = 0, bits = 0, channels = 1;
			int blockAlign = 0, samplesPerBlock = 0;
			boolean isFloat = false;
			long dataOffset = -1, dataSize = 0, fileSize = ch.size();

			while (ch.position() + 8 <= fileSize) {
				String id = readString(ch, 4);
				int chunkSize = readIntLE(ch);
				if (id.equals("fmt ")) {
					fmtCode = readShortLE(ch);
					channels = readShortLE(ch);
					sampleRate = readIntLE(ch);
					ch.position(ch.position() + 4);
					blockAlign = readShortLE(ch);
					bits = readShortLE(ch);
					if (fmtCode == 2 && chunkSize >= 20) {
						int cbSize = readShortLE(ch);
						if (cbSize >= 2 && ch.position() + 2 <= fileSize)
							samplesPerBlock = readShortLE(ch);
					} else {
						long remain = chunkSize - 16;
						if (remain > 0 && ch.position() + remain <= fileSize)
							ch.position(ch.position() + remain);
					}
					isFloat = (fmtCode == 3);
				} else if (id.equals("data")) {
					dataOffset = ch.position() - 4;
					dataSize = chunkSize;
					if (dataSize <= 0 || dataOffset + 8 + dataSize > fileSize)
						dataSize = fileSize - (dataOffset + 8);
					ch.position(ch.position() + dataSize);
				} else {
					if (chunkSize < 0 || ch.position() + chunkSize > fileSize) break;
					ch.position(ch.position() + chunkSize);
				}
			}

			if (dataOffset < 0) {
				dataOffset = Math.max(0, fileSize - 8);
				dataSize = Math.max(0, fileSize - (dataOffset + 8));
			}

			ch.position(dataOffset + 8);
			if (dataSize <= 0) dataSize = ch.size() - (dataOffset + 8);

			ByteBuffer pcm = ByteBuffer.allocate((int) dataSize);
			ch.read(pcm); pcm.flip();

			float[] data;
			if (fmtCode == 1 || fmtCode == 3) {
				ByteBuffer bb = pcm.order(ByteOrder.LITTLE_ENDIAN);
				int samples = (int) dataSize / (bits / 8);
				data = decodeSamples(bb, bits, isFloat, samples);
			} else if (fmtCode == 2 || fmtCode == 17) { // Microsoft ADPCM or IMA ADPCM
				byte[] adpcm = new byte[(int) dataSize];
				pcm.get(adpcm);
				short[] pcm16;
				if (fmtCode == 2)
					pcm16 = decodeADPCMBlock(adpcm, blockAlign, samplesPerBlock, channels);
				else
					pcm16 = decodeIMAADPCMBlock(adpcm, blockAlign, channels);
				data = new float[pcm16.length];
				for (int i = 0; i < pcm16.length; i++) data[i] = pcm16[i] / 32768f;
			} else {
				throw new IOException("Unsupported WAV format (fmt=" + fmtCode + ")");
			}

			if (sampleRate != targetRate)
				data = resampleLinear(data, sampleRate, targetRate);
			return data;
		} catch (Exception e) {
			System.err.printf("[Tensor] Failed to load WAV: %s (%s)%n", file.getName(), e.getMessage());
			return new float[0];
		}
	}

	private short[] decodeIMAADPCMBlock(byte[] data, int blockAlign, int channels) {
		if (channels != 1) throw new UnsupportedOperationException("Only mono IMA ADPCM supported");
		int blocks = data.length / blockAlign;
		short[] out = new short[blocks * (blockAlign - 4) * 2];
		int outPos = 0;

		int[] stepTable = {
				7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
				34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
				157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658,
				724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
				3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
				15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
		};
		int[] indexTable = {-1,-1,-1,-1,2,4,6,8};

		ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < blocks; b++) {
			short predictor = bb.getShort();
			int stepIndex = bb.get() & 0x7F;
			bb.get(); // reserved

			int step = stepTable[stepIndex];
			int valPred = predictor;
			out[outPos++] = (short) valPred;

			for (int i = 4; i < blockAlign; i++) {
				int n = bb.get() & 0xFF;
				for (int nib = 0; nib < 2; nib++) {
					int code = (n >> (nib * 4)) & 0x0F;
					int diff = step >> 3;
					if ((code & 1) != 0) diff += step >> 2;
					if ((code & 2) != 0) diff += step >> 1;
					if ((code & 4) != 0) diff += step;
					if ((code & 8) != 0) diff = -diff;
					valPred += diff;
					valPred = Math.max(-32768, Math.min(32767, valPred));

					stepIndex += indexTable[Math.min(code & 7, 7)];
					stepIndex = Math.max(0, Math.min(88, stepIndex));
					step = stepTable[stepIndex];
					out[outPos++] = (short) valPred;
				}
			}
		}
		return out;
	}

	private int readShortLE(FileChannel ch) throws IOException {
		ByteBuffer b = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
		ch.read(b); b.flip();
		return Short.toUnsignedInt(b.getShort());
	}

	private int readIntLE(FileChannel ch) throws IOException {
		ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
		ch.read(b); b.flip();
		return b.getInt();
	}

	private String readString(FileChannel ch, int len) throws IOException {
		ByteBuffer b = ByteBuffer.allocate(len);
		ch.read(b); b.flip();
		return new String(b.array(), "US-ASCII");
	}

	private float[] decodeSamples(ByteBuffer bb, int bits, boolean isFloat, int samples) {
		float[] out = new float[samples];
		switch (bits) {
			case 8 -> { for (int i = 0; i < samples; i++) out[i] = (bb.get() - 128) / 128f; }
			case 16 -> { for (int i = 0; i < samples; i++) out[i] = bb.getShort() / 32768f; }
			case 24 -> {
				for (int i = 0; i < samples; i++) {
					int b0 = bb.get() & 0xFF, b1 = bb.get() & 0xFF, b2 = bb.get();
					int val = (b2 << 16) | (b1 << 8) | b0;
					if ((b2 & 0x80) != 0) val |= 0xFF000000;
					out[i] = val / 8388608f;
				}
			}
			case 32 -> {
				if (isFloat) for (int i = 0; i < samples; i++) out[i] = bb.getFloat();
				else for (int i = 0; i < samples; i++) out[i] = bb.getInt() / 2147483648f;
			}
		}
		return out;
	}

	private short[] decodeADPCMBlock(byte[] data, int blockAlign, int samplesPerBlock, int channels) {
		if (channels != 1) throw new UnsupportedOperationException("Only mono ADPCM supported");
		int blocks = data.length / blockAlign;
		short[] out = new short[blocks * samplesPerBlock];
		int outPos = 0;
		int[][] coeff = {{256,0},{512,-256},{0,0},{192,64},{240,0},{460,-208},{392,-232}};
		ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
		for (int b = 0; b < blocks; b++) {
			int predictor = bb.get() & 0xFF;
			int delta = bb.getShort();
			int sample1 = bb.getShort();
			int sample2 = bb.getShort();
			out[outPos++] = (short) sample2;
			out[outPos++] = (short) sample1;
			for (int i = 7; i < blockAlign; i++) {
				int val = bb.get() & 0xFF;
				int hi = (val >> 4) & 0x0F, lo = val & 0x0F;
				for (int nib = 0; nib < 2; nib++) {
					int code = (nib == 0 ? lo : hi);
					int pred = (sample1 * coeff[predictor][0] + sample2 * coeff[predictor][1]) / 256;
					int samp = pred + (code < 8 ? code * delta : (code - 16) * delta);
					samp = Math.max(-32768, Math.min(32767, samp));
					sample2 = sample1; sample1 = samp;
					out[outPos++] = (short) samp;
				}
			}
		}
		return out;
	}

	private float[] resampleLinear(float[] in, int srIn, int srOut) {
		if (srIn == srOut) return in;
		double ratio = (double) srOut / srIn;
		int newLen = (int) Math.round(in.length * ratio);
		float[] out = new float[newLen];
		for (int i = 0; i < newLen; i++) {
			double pos = i / ratio;
			int idx = (int) pos;
			double frac = pos - idx;
			out[i] = (idx + 1 < in.length)
					? (float) ((1 - frac) * in[idx] + frac * in[idx + 1])
					: in[in.length - 1];
		}
		return out;
	}

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

	private static int getLenFromShape(int... shape) {
		int size = 1;
		for (int d : shape)
			size *= d;
		return size;
	}

	private static int normalizeAxis(int axis, int rank) {
		if (axis < 0) axis += rank;
		if (axis < 0 || axis >= rank)
			throw new IllegalArgumentException("Axis out of range: " + axis);
		return axis;
	}


}
