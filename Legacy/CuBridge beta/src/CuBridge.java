import java.io.*;

/**
 * CuBridge: A Java-CUDA bridge class for high-performance numerical computation.
 *
 * <p>This class defines tensor-based numerical operations in Java and delegates their execution
 * to either CPU or GPU (via CUDA) depending on system capabilities.
 * Operations are automatically routed to the appropriate memory location (RAM or VRAM),
 * and communication between Java and CUDA kernels is managed through JNI.
 *
 * <h2>Design Principles</h2>
 * <ul>
 *   <li><b>Instruction-Execution Model</b>: Operations are instructed in Java, executed natively via CUDA or CPU.</li>
 *   <li><b>RAM and VRAM Separation</b>: If CUDA is available, operations are executed on the GPU; otherwise, fallback to CPU.</li>
 * </ul>
 *
 * <h2>Data Storage Architecture</h2>
 * <ul>
 *   <li><b>Queue (Execution Queue)</b>: All computations are queued and results are popped sequentially.</li>
 *   <li><b>Map (Persistent Memory)</b>: Tensors stored with specific names have usageCount -1, meaning they persist indefinitely.</li>
 *   <li><b>The queue is a circular buffer</b> where tensors are automatically removed when usageCount reaches zero.</li>
 *   <li><b>All memory is reset when {@code clean()} is called</b>, including both the queue and the map.</li>
 * </ul>
 *
 * <h2>Main Features</h2>
 * <ul>
 *   <li>Unary and binary math operations (e.g., sqrt, log, add, div)</li>
 *   <li>Axis-based statistical functions (sum, mean, std, etc.)</li>
 *   <li>Broadcasting and tensor reshaping (compress, expand, transpose)</li>
 *   <li>Neural network specialized functions (mse, cee, affine, softmax)</li>
 * </ul>
 *
 * <h2>Function Overview</h2>
 * <table border="1">
 *   <caption>Function Overview</caption>
 *   <tr>
 *     <th>Function Type</th>
 *     <th>Representative Functions</th>
 *     <th>Total Functions (Including Overloads)</th>
 *   </tr>
 *   <tr>
 *     <td>Unary Operations</td>
 *     <td>abs, neg, square, sqrt, log</td>
 *     <td>63</td>
 *   </tr>
 *   <tr>
 *     <td>Binary Operations</td>
 *     <td>add, sub, mul, div, dot</td>
 *     <td>64</td>
 *   </tr>
 *   <tr>
 *     <td>Axis-based Operations</td>
 *     <td>sum, mean, var, std, argmax</td>
 *     <td>30</td>
 *   </tr>
 *   <tr>
 *     <td>Neural Network Functions</td>
 *     <td>mse, cee, affine, softmax</td>
 *     <td>13</td>
 *   </tr>
 *   <tr>
 *     <td>Broadcast / Reshape Functions</td>
 *     <td>transpose, compress, expand</td>
 *     <td>12</td>
 *   </tr>
 * </table>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * CuBridge cb = CuBridge.getInstance();
 * cb.put(1.0, "x").put(2.0, "y").add("x", "y", "z");
 * Tensor result = cb.get("z");
 * }</pre>
 *
 * <h2>Notes</h2>
 * <ul>
 *   <li>{@link CuBridgeJNI} is an internal implementation class and should not be accessed directly.
 *       All functions must be invoked through CuBridge.</li>
 *   <li>{@code clean()} will remove all stored tensors and reset both the queue and map to their initial state.</li>
 * </ul>
 *
 * <h2>Related Libraries and Modules</h2>
 * {@link Tensor}        // Multidimensional array structure for data representation
 *
 * @author 배준호, 조선대 3학년
 * @since 1.0
 */
public class CuBridge {
	private static final CuBridge instance = new CuBridge();

//이름이 ""이면 무조건 최선!

	private CuBridge() {
	}
	
	/**
	 * Returns the singleton instance of CuBridge.
	 *
	 * @return the CuBridge instance
	 */
	public static CuBridge getInstance() {
		return instance;
	}
	
	/**
	 * Sets the memory storage mode to RAM (host memory).
	 * Disables auto-detection.
	 */
	public void selectRam() {
		CuBridgeJNI.setAuto(false);
		CuBridgeJNI.setRAM(false);
		CuBridgeJNI.refresh();
	}

	/**
	 * Sets the memory storage mode to VRAM (GPU memory).
	 * Disables auto-detection.
	 */
	public void selectVRam() {
		CuBridgeJNI.setAuto(false);
		CuBridgeJNI.setRAM(true);
		CuBridgeJNI.refresh();
	}
	
	/**
	 * Sets the computation mode to CPU.
	 * Disables auto-detection.
	 */
	public void selectCPU() {
		CuBridgeJNI.setAuto(false);
		CuBridgeJNI.setCAL(false);
		CuBridgeJNI.refresh();
	}
	
	/**
	 * Sets the computation mode to GPU.
	 * Disables auto-detection.
	 */
	void selectGPU() {
		CuBridgeJNI.setAuto(false);
		CuBridgeJNI.setCAL(true);
		CuBridgeJNI.refresh();
	}
	
	/**
	 * Resets the environment to auto-detection mode.
	 * Memory and computation mode will be automatically selected.
	 */
	public void envReset() {
		CuBridgeJNI.setAuto(true);
		CuBridgeJNI.refresh();
	}
	
	/**
	 * Returns the current memory mode.
	 *
	 * @return true if using VRAM; false if using RAM
	 */
	public boolean RamStat() {
		return CuBridgeJNI.getRAM();
	}
	
	/**
	 * Returns the current computation mode.
	 *
	 * @return true if using GPU; false if using CPU
	 */
	public boolean CalStat() {
		return CuBridgeJNI.getCAL();
	}
	
	/**
	 * Returns whether the system is in auto-detection mode.
	 *
	 * @return true if auto mode is enabled
	 */
	public boolean EnvStat() {
		return CuBridgeJNI.getENV();
	}
	
	/**
	 * Prints the current system and CuBridge environment status.
	 *
	 * <p>This includes both general system information and CUDA-related configuration,
	 * and reflects the current CuBridge runtime environment.</p>
	 *
	 * <ul>
	 *   <li>Operating system and physical system RAM</li>
	 *   <li>Detected GPU name and available VRAM (if CUDA is available)</li>
	 *   <li>Installed CUDA Driver and Runtime version (only shown if CUDA is installed)</li>
	 *   <li>CuBridge auto-detection status (for CUDA availability)</li>
	 *   <li>Current compute mode (CPU or GPU)</li>
	 *   <li>Active memory storage location (RAM or VRAM)</li>
	 * </ul>
	 */
	public void getEnvironmentStatus() {
        boolean auto = CuBridgeJNI.getENV();
        boolean gpuCompute = CuBridgeJNI.getCAL();
        boolean gpuMemory = CuBridgeJNI.getRAM();

        StringBuilder sb = new StringBuilder();
        sb.append("[System 환경 상태]\n");
        sb.append(CuBridgeJNI.getSysInfo());
        sb.append("\n[CuBridge 환경 상태]\n");
        sb.append("- 자동 감지 모드: ").append(auto ? "O" : "X").append("\n");
        sb.append("- 연산 방식: ").append(gpuCompute ? "GPU" : "CPU").append("\n");
        sb.append("- 메모리 저장 위치: ").append(gpuMemory ? "VRAM" : "RAM").append("\n");

        System.out.println(sb.toString());
    }

    /**
     * Clears all tensors from the internal queue and memory.
     */
	public void clear() {
		CuBridgeJNI.clear();
		CuBridgeJNI.cachedClean();
		return;
	}

	/**
	 * Returns the current number of tensors in the internal queue.
	 *
	 * @return the size of the queue
	 */
	public int getQueueSize() {
		return CuBridgeJNI.getQueueSize();
	}

	/**
	 * Prints the top tensor from the internal execution queue.
	 *
	 * This tensor is the next to be used in computation.
	 * Equivalent to peeking the front of the queue.
	 */
	public void visualQueue() {
		System.out.println(CuBridgeJNI.visualQueue());
	} // 스택 전체

	/**
	 * Prints the queue stack of a specific tensor name.
	 *
	 * This shows all stacked TensorMeta instances associated with the given name.
	 *
	 * @param name the name of the tensor to inspect
	 */
	public void visualQueue(String name) {
		System.out.println(CuBridgeJNI.visualQueue(name));
	} // 스택 내부 특정 텐서만

	/**
	 * Prints the entire queue contents for all tensor names.
	 *
	 * <ul>
	 *   <li>Includes all names currently managed by the queue system</li>
	 *   <li>Displays the full stack of each tensor</li>
	 * </ul>
	 */
	public void visualQueueAll() {
		System.out.println(CuBridgeJNI.visualQueueAll());
	} // 스택 내부 특정 텐서만

	/**
	 * Returns the total number of tensors currently allocated in memory.
	 *
	 * @return the number of active memory blocks
	 */
	public int getMemSize() {
		return CuBridgeJNI.getMemSize();
	}

	/**
	 * Prints the memory allocation status of a specific tensor.
	 *
	 * @param name the name of the tensor to visualize
	 */
	public void visualMem(String name) {
		System.out.println(CuBridgeJNI.visualMem(name));
	} // 스택 내부 특정 텐서만

	/**
	 * Prints the memory allocation status of all tensors.
	 */
	public void visualMemAll() {
		System.out.println(CuBridgeJNI.visualMemAll());
	} // 스택 내부 특정 텐서만

	/**
	 * <ul>
	 *   <li>Stores an integer scalar as an unnamed tensor.</li>
	 *   <li>This tensor is automatically marked as broadcastable.</li>
	 *   <li>When used in binary operations, it must appear as the second operand.</li>
	 * </ul>
	 *
	 * @param data the integer value to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(int data) {
		return put(new Tensor(data), true);
	}

	/**
	 * <ul>
	 *   <li>Stores an integer scalar as a named tensor.</li>
	 *   <li>This tensor is automatically marked as broadcastable.</li>
	 *   <li>When used in binary operations, it must appear as the second operand.</li>
	 * </ul>
	 *
	 * @param data the integer value to store
	 * @param name the name to assign to the tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(int data, String name) {
		return put(new Tensor(data), name, 1, true);
	}

	/**
	 * <ul>
	 *   <li>Stores an integer scalar with a name and usage count.</li>
	 *   <li>This tensor is automatically marked as broadcastable.</li>
	 *   <li>When used in binary operations, it must appear as the second operand.</li>
	 * </ul>
	 *
	 * @param data the integer value to store
	 * @param name the tensor name
	 * @param usageCount number of times this tensor will be used (must be > 0 or -1)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(int data, String name, int usageCount) {
		return put(new Tensor(data), name, usageCount, true);
	}
	
	/**
	 * <ul>
	 *   <li>Stores a double scalar as an unnamed tensor.</li>
	 *   <li>This tensor is automatically marked as broadcastable.</li>
	 *   <li>When used in binary operations, it must appear as the second operand.</li>
	 * </ul>
	 *
	 * @param data the double value to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(double data) {
		return put(new Tensor(data), true);
	}

	/**
	 * <ul>
	 *   <li>Stores a double scalar as a named tensor.</li>
	 *   <li>This tensor is automatically marked as broadcastable.</li>
	 *   <li>When used in binary operations, it must appear as the second operand.</li>
	 * </ul>
	 *
	 * @param data the double value to store
	 * @param name the tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(double data, String name) {
		return put(new Tensor(data), name, 1, true);
	}

	/**
	 * <ul>
	 *   <li>Stores a double scalar with a name and usage count.</li>
	 *   <li>This tensor is automatically marked as broadcastable.</li>
	 *   <li>When used in binary operations, it must appear as the second operand.</li>
	 * </ul>
	 *
	 * @param data the double value to store
	 * @param name the tensor name
	 * @param usageCount number of times this tensor will be used (must be > 0 or -1)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(double data, String name, int usageCount) {
		return put(new Tensor(data), name, usageCount, true);
	}

	/**
	 * <ul>
	 *   <li>Stores a tensor with an empty name and broadcasting disabled.</li>
	 *   <li>This is typically used for temporary tensors. Name will be "".</li>
	 * </ul>
	 *
	 * @param data the tensor to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data) {
		return put(data, false);
	}
	
	/**
	 * <ul>
	 *   <li>Stores a tensor with an empty name and specified broadcasting flag.</li>
	 * </ul>
	 *
	 * @param data the tensor to store
	 * @param broadcast whether to mark this tensor as broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, boolean broadcast) {
		CuBridgeJNI.put(data.toArray(), data.getShape(), data.getSize(), data.getAxis(), 1, "", broadcast);
		return instance;
	}

	/**
	 * <ul>
	 *   <li>Stores a named tensor with default usage count and no broadcasting.</li>
	 * </ul>
	 *
	 * @param data the tensor to store
	 * @param name the tensor name (must not be empty)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name) {
		return put(data, name, 1, false);
	}

	/**
	 * <ul>
	 *   <li>Stores a named tensor with specified broadcasting flag.</li>
	 * </ul>
	 *
	 * @param data the tensor to store
	 * @param name the tensor name (must not be empty)
	 * @param broadcast whether to mark this tensor as broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, boolean broadcast) {
		return put(data, name, 1, broadcast);
	}

	/**
	 * <ul>
	 *   <li>Stores a named tensor with a specific usage count.</li>
	 * </ul>
	 *
	 * @param data the tensor to store
	 * @param name the tensor name (must not be empty)
	 * @param usageCount number of times this tensor will be used (must be > 0 or -1)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, int usageCount) {
		return put(data, name, usageCount, false);
	}

	/**
	 * <ul>
	 *   <li>Stores a tensor with complete configuration: name, usage count, and broadcasting flag.</li>
	 *   <li>If the name is null or empty, an error is printed.</li>
	 *   <li>If the usage count is 0 or negative (except -1), an error is printed.</li>
	 *   <li>If the name already exists, the tensor will not be stored.</li>
	 * </ul>
	 *
	 * @param data the tensor to store
	 * @param name the tensor name
	 * @param usageCount number of times this tensor will be used (-1 for unlimited)
	 * @param broadcast whether to mark this tensor as broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, int usageCount, boolean broadcast) {
		if (usageCount == 0 || usageCount < -1) {
			System.err.println("Error: Please UsageCount modify.");
			return instance;
		}

		if (name == null || name.isEmpty()) {
			System.err.println("Error: Tensor name must be defined.");
			return instance;
		}

		if (!CuBridgeJNI.put(data.toArray(), data.getShape(), data.getSize(), data.getAxis(), usageCount, name,
				broadcast))
			System.err.println("Error: Tensor name is duplicated. Please choose another name.");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Retrieves and removes the top tensor from the internal queue.</li>
	 *   <li>If the queue is empty, an error message is printed and {@code null} is returned.</li>
	 * </ul>
	 *
	 * @return the retrieved Tensor, or {@code null} if the queue is empty
	 */
	public Tensor get() {
		if (!CuBridgeJNI.pop("")) {
			System.err.println("Error: Queue is empty!");
			return null;
		}

		return getTensor();
	}

	/**
	 * <ul>
	 *   <li>Retrieves and removes the top tensor associated with the given name.</li>
	 *   <li>If no tensor with that name exists in the queue, an error is printed and {@code null} is returned.</li>
	 * </ul>
	 *
	 * @param name the name of the tensor to retrieve
	 * @return the retrieved Tensor, or {@code null} if not found
	 */
	public Tensor get(String name) {
		if (!CuBridgeJNI.pop(name)) {
			System.err.println("Error: The " + name + " is not exist in Queue!");
			return null;
		}

		return getTensor();
	}

	/**
	 * <ul>
	 *   <li>Internal method to extract the data and shape of a tensor from the native backend.</li>
	 *   <li>This method assumes a successful pop operation was already performed.</li>
	 *   <li>It clears cached memory after extracting the data.</li>
	 * </ul>
	 *
	 * @return the Tensor object constructed from native memory
	 */
	private Tensor getTensor() {
		double[] data = CuBridgeJNI.getData();
		int[] shape = CuBridgeJNI.getShape();
		CuBridgeJNI.cachedClean();

		return new Tensor(data, shape);
	}
	
	/**
	 * <ul>
	 *   <li>Increases the usage count of an existing tensor in the execution queue.</li>
	 *   <li>This method searches for a tensor with the given name in the current queue and increments its internal usage count.</li>
	 *   <li>The tensor itself is not duplicated or copied; only its metadata is updated.</li>
	 *   <li><b>Note:</b> This operation is local to the execution queue and does not interact with the map.</li>
	 *   <li>If the usage count is less than 1, or the tensor is not found in the queue, the operation fails.</li>
	 * </ul>
	 *
	 * @param name the name of the tensor in the queue
	 * @param usageCount the number of additional times this tensor should be allowed to be used (must be ≥ 1)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge duple(String name, int usageCount) {
		if(usageCount < 1) {
			System.err.println("Error: Tensor '" + name + "' cannot be duplicated; invalid usage count.");
			return instance;
		}
			
		 if(!CuBridgeJNI.duple(name, usageCount))
			System.err.println("Error: Failed to update usage count for tensor '" + name + "' in the queue.");

		 return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies absolute value (|x|) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge abs() {
		if (!CuBridgeJNI.abs("", ""))
			System.err.println("ABS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies absolute value (|x|) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge abs(String a) {
		if (!CuBridgeJNI.abs(a, ""))
			System.err.println("ABS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies absolute value (|x|) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge abs(String a, String out) {
		if (!CuBridgeJNI.abs(a, out))
			System.err.println("ABS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies negation (-x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge neg() {
		if (!CuBridgeJNI.neg("", ""))
			System.err.println("NEG Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies negation (-x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge neg(String a) {
		if (!CuBridgeJNI.neg(a, ""))
			System.err.println("NEG Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies negation (-x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge neg(String a, String out) {
		if (!CuBridgeJNI.neg(a, out))
			System.err.println("NEG Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies squaring (x²) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge square() {
		if (!CuBridgeJNI.square("", ""))
			System.err.println("SQUARE Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies squaring (x²) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge square(String a) {
		if (!CuBridgeJNI.square(a, ""))
			System.err.println("SQUARE Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies squaring (x²) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge square(String a, String out) {
		if (!CuBridgeJNI.square(a, out))
			System.err.println("SQUARE Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies square root (√x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sqrt() {
		if (!CuBridgeJNI.sqrt("", ""))
			System.err.println("SQRT Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies square root (√x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sqrt(String a) {
		if (!CuBridgeJNI.sqrt(a, ""))
			System.err.println("SQRT Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies square root (√x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sqrt(String a, String out) {
		if (!CuBridgeJNI.sqrt(a, out))
			System.err.println("SQRT Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies base-10 logarithm (log₁₀x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log() {
		if (!CuBridgeJNI.log("", ""))
			System.err.println("LOG Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies base-10 logarithm (log₁₀x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log(String a) {
		if (!CuBridgeJNI.log(a, ""))
			System.err.println("LOG Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies base-10 logarithm (log₁₀x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log(String a, String out) {
		if (!CuBridgeJNI.log(a, out))
			System.err.println("LOG Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies base-2 logarithm (log₂x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log2() {
		if (!CuBridgeJNI.log2("", ""))
			System.err.println("LOG_2 Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies base-2 logarithm (log₂x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log2(String a) {
		if (!CuBridgeJNI.log2(a, ""))
			System.err.println("LOG_2 Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies base-2 logarithm (log₂x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log2(String a, String out) {
		if (!CuBridgeJNI.log2(a, out))
			System.err.println("LOG_2 Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies natural logarithm (ln x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ln() {
		if (!CuBridgeJNI.ln("", ""))
			System.err.println("LOG_e Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies natural logarithm (ln x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ln(String a) {
		if (!CuBridgeJNI.ln(a, ""))
			System.err.println("LOG_e Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies natural logarithm (ln x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ln(String a, String out) {
		if (!CuBridgeJNI.ln(a, out))
			System.err.println("LOG_e Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies reciprocal (1/x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal() {
		if (!CuBridgeJNI.reciprocal("", ""))
			System.err.println("REVERSE Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies reciprocal (1/x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal(String a) {
		if (!CuBridgeJNI.reciprocal(a, ""))
			System.err.println("REVERSE Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies reciprocal (1/x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal(String a, String out) {
		if (!CuBridgeJNI.reciprocal(a, out))
			System.err.println("REVERSE Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	// 이항
	/**
	 * <ul>
	 *   <li>Applies element-wise addition (a + b) to the top two tensors in the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add() {
		if (!CuBridgeJNI.add("", "", ""))
			System.err.println("ADD Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise addition (a + b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add(String a) {
		if (!CuBridgeJNI.add(a, "", ""))
			System.err.println("ADD Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise addition (a + b) using the two specified tensors.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add(String a, String b) {
		if (!CuBridgeJNI.add(a, b, ""))
			System.err.println("ADD Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise addition (a + b) using the two specified tensors and stores the result.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add(String a, String b, String out) {
		if (!CuBridgeJNI.add(a, b, out))
			System.err.println("ADD Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise subtraction (a - b) to the top two tensors in the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub() {
		if (!CuBridgeJNI.sub("", "", ""))
			System.err.println("SUB Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise subtraction (a - b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub(String a) {
		if (!CuBridgeJNI.sub(a, "", ""))
			System.err.println("SUB Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise subtraction (a - b) using the two specified tensors.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub(String a, String b) {
		if (!CuBridgeJNI.sub(a, b, ""))
			System.err.println("SUB Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise subtraction (a - b) using the two specified tensors and stores the result.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub(String a, String b, String out) {
		if (!CuBridgeJNI.sub(a, b, out))
			System.err.println("SUB Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise multiplication (a × b) to the top two tensors in the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul() {
		if (!CuBridgeJNI.mul("", "", ""))
			System.err.println("MUL Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise multiplication (a × b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul(String a) {
		if (!CuBridgeJNI.mul(a, "", ""))
			System.err.println("MUL Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise multiplication (a × b) using the two specified tensors.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul(String a, String b) {
		if (!CuBridgeJNI.mul(a, b, ""))
			System.err.println("MUL Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise multiplication (a × b) using the two specified tensors and stores the result.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul(String a, String b, String out) {
		if (!CuBridgeJNI.mul(a, b, out))
			System.err.println("MUL Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise division (a ÷ b) to the top two tensors in the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div() {
		if (!CuBridgeJNI.div("", "", ""))
			System.err.println("DIV Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise division (a ÷ b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div(String a) {
		if (!CuBridgeJNI.div(a, "", ""))
			System.err.println("DIV Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise division (a ÷ b) using the two specified tensors.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div(String a, String b) {
		if (!CuBridgeJNI.div(a, b, ""))
			System.err.println("DIV Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise division (a ÷ b) using the two specified tensors and stores the result.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div(String a, String b, String out) {
		if (!CuBridgeJNI.div(a, b, out))
			System.err.println("DIV Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise exponentiation (a ^ b) to the top two tensors in the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow() {
		if (!CuBridgeJNI.pow("", "", ""))
			System.err.println("POW Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise exponentiation (a ^ b) using the specified tensor as the base.</li>
	 *   <li>The exponent is taken from the top of the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the base tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow(String a) {
		if (!CuBridgeJNI.pow(a, "", ""))
			System.err.println("POW Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise exponentiation (a ^ b) using the two specified tensors.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the base tensor
	 * @param b the exponent tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow(String a, String b) {
		if (!CuBridgeJNI.pow(a, b, ""))
			System.err.println("POW Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise exponentiation (a ^ b) using the two specified tensors and stores the result.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the base tensor
	 * @param b the exponent tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow(String a, String b, String out) {
		if (!CuBridgeJNI.pow(a, b, out))
			System.err.println("POW Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}
	
	/**
	 * <ul>
	 *   <li>Applies element-wise modulus (a % b) to the top two tensors in the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod() {
		if (!CuBridgeJNI.mod("", "", ""))
			System.err.println("MOD Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise modulus (a % b) using the specified tensor as the dividend.</li>
	 *   <li>The divisor is taken from the top of the queue.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the dividend tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod(String a) {
		if (!CuBridgeJNI.mod(a, "", ""))
			System.err.println("MOD Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise modulus (a % b) using the two specified tensors.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the dividend tensor
	 * @param b the divisor tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod(String a, String b) {
		if (!CuBridgeJNI.mod(a, b, ""))
			System.err.println("MOD Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise modulus (a % b) using the two specified tensors and stores the result.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the dividend tensor
	 * @param b the divisor tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod(String a, String b, String out) {
		if (!CuBridgeJNI.mod(a, b, out))
			System.err.println("MOD Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}
	
	/**
	 * <ul>
	 *   <li>Performs dot product (a ⋅ b) between the last two axes of the top two tensors in the queue.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>All other axes must match.</li>
	 *   <li>Use {@code transpose()} to reshape 1D vectors to (n,1) if needed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot() {
		if (!CuBridgeJNI.dot("", "", ""))
			System.err.println("DOT Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs dot product (a ⋅ b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>All leading axes must match.</li>
	 *   <li>Reshape 1D vectors to (n,1) if necessary.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(String a) {
		if (!CuBridgeJNI.dot(a, "", ""))
			System.err.println("DOT Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs dot product (a ⋅ b) using the two specified tensors.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>All other axes must be equal.</li>
	 *   <li>If either tensor is a 1D vector, use {@code transpose()} to convert it to (n,1).</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(String a, String b) {
		if (!CuBridgeJNI.dot(a, b, ""))
			System.err.println("DOT Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs dot product (a ⋅ b) between the last two axes of the two specified tensors and stores the result.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>All leading axes must have the same shape; otherwise, the operation fails.</li>
	 *   <li>Use {@code transpose()} to reshape 1D vectors to (n, 1) if needed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(String a, String b, String out) {
		if (!CuBridgeJNI.dot(a, b, out))
			System.err.println("DOT Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	// 논리연산
	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than comparison (a &gt; b) to the top two tensors in the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt() {
		if (!CuBridgeJNI.gt("", "", ""))
			System.err.println("GreaterThan Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than comparison (a &gt; b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt(String a) {
		if (!CuBridgeJNI.gt(a, "", ""))
			System.err.println("GreaterThan Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than comparison (a &gt; b) using the two specified tensors.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt(String a, String b) {
		if (!CuBridgeJNI.gt(a, b, ""))
			System.err.println("GreaterThan Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than comparison (a &gt; b) using the two specified tensors and stores the result.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt(String a, String b, String out) {
		if (!CuBridgeJNI.gt(a, b, out))
			System.err.println("GreaterThan Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than comparison (a &lt; b) to the top two tensors in the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return CuBridge instance for chaining
	 */
	public CuBridge lt() {
		if (!CuBridgeJNI.lt("", "", ""))
			System.err.println("LessThan Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than comparison (a &lt; b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge lt(String a) {
		if (!CuBridgeJNI.lt(a, "", ""))
			System.err.println("LessThan Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than comparison (a &lt; b) using the two specified tensors.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge lt(String a, String b) {
		if (!CuBridgeJNI.lt(a, b, ""))
			System.err.println("LessThan Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than comparison (a &lt; b) using the two specified tensors and stores the result.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge lt(String a, String b, String out) {
		if (!CuBridgeJNI.lt(a, b, out))
			System.err.println("LessThan Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than-or-equal comparison (a ≥ b) to the top two tensors in the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ge() {
		if (!CuBridgeJNI.ge("", "", ""))
			System.err.println("GreaterEqual Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than-or-equal comparison (a ≥ b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ge(String a) {
		if (!CuBridgeJNI.ge(a, "", ""))
			System.err.println("GreaterEqual Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than-or-equal comparison (a ≥ b) using the two specified tensors.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ge(String a, String b) {
		if (!CuBridgeJNI.ge(a, b, ""))
			System.err.println("GreaterEqual Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise greater-than-or-equal comparison (a ≥ b) using the two specified tensors and stores the result.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ge(String a, String b, String out) {
		if (!CuBridgeJNI.ge(a, b, out))
			System.err.println("GreaterEqual Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than-or-equal comparison (a ≤ b) to the top two tensors in the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge le() {
		if (!CuBridgeJNI.le("", "", ""))
			System.err.println("LessEqual Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than-or-equal comparison (a ≤ b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge le(String a) {
		if (!CuBridgeJNI.le(a, "", ""))
			System.err.println("LessEqual Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than-or-equal comparison (a ≤ b) using the two specified tensors.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge le(String a, String b) {
		if (!CuBridgeJNI.le(a, b, ""))
			System.err.println("LessEqual Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise less-than-or-equal comparison (a ≤ b) using the two specified tensors and stores the result.</li>
	 *   <li>If the condition is true, the result is 1; otherwise, 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the first input tensor
	 * @param b the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge le(String a, String b, String out) {
		if (!CuBridgeJNI.le(a, b, out))
			System.err.println("LessEqual Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise equality comparison (a == b) to the top two tensors in the queue.</li>
	 *   <li>If the values are equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge eq() {
		if (!CuBridgeJNI.eq("", "", ""))
			System.err.println("Equal Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise equality comparison (a == b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If the values are equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge eq(String a) {
		if (!CuBridgeJNI.eq(a, "", ""))
			System.err.println("Equal Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise equality comparison (a == b) using the two specified tensors.</li>
	 *   <li>If the values are equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge eq(String a, String b) {
		if (!CuBridgeJNI.eq(a, b, ""))
			System.err.println("Equal Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise equality comparison (a == b) using the two specified tensors and stores the result.</li>
	 *   <li>If the values are equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge eq(String a, String b, String out) {
		if (!CuBridgeJNI.eq(a, b, out))
			System.err.println("Equal Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise not-equal comparison (a != b) to the top two tensors in the queue.</li>
	 *   <li>If the values are not equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ne() {
		if (!CuBridgeJNI.ne("", "", ""))
			System.err.println("NotEqual Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise not-equal comparison (a != b) using the specified tensor as the first operand.</li>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>If the values are not equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ne(String a) {
		if (!CuBridgeJNI.ne(a, "", ""))
			System.err.println("NotEqual Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise not-equal comparison (a != b) using the two specified tensors.</li>
	 *   <li>If the values are not equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ne(String a, String b) {
		if (!CuBridgeJNI.ne(a, b, ""))
			System.err.println("NotEqual Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies element-wise not-equal comparison (a != b) using the two specified tensors and stores the result.</li>
	 *   <li>If the values are not equal, returns 1; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ne(String a, String b, String out) {
		if (!CuBridgeJNI.ne(a, b, out))
			System.err.println("NotEqual Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical AND to the top two tensors in the queue.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if both elements are non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge and() {
		if (!CuBridgeJNI.and("", "", ""))
			System.err.println("AND Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical AND using the specified tensor as the first operand.
	 * <ul>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if both elements are non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge and(String a) {
		if (!CuBridgeJNI.and(a, "", ""))
			System.err.println("AND Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical AND using the two specified tensors.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if both elements are non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge and(String a, String b) {
		if (!CuBridgeJNI.and(a, b, ""))
			System.err.println("AND Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical AND using the two specified tensors and stores the result.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if both elements are non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge and(String a, String b, String out) {
		if (!CuBridgeJNI.and(a, b, out))
			System.err.println("AND Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical OR to the top two tensors in the queue.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if either element is non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge or() {
		if (!CuBridgeJNI.or("", "", ""))
			System.err.println("OR Error: Tensor not exist in Queue!");

		return instance;
	}
	
	/**
	 * Applies element-wise logical OR using the specified tensor as the first operand.
	 * <ul>
	 *   <li>The second operand is taken from the top of the queue.</li>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if either element is non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge or(String a) {
		if (!CuBridgeJNI.or(a, "", ""))
			System.err.println("OR Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical OR using the two specified tensors.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if either element is non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge or(String a, String b) {
		if (!CuBridgeJNI.or(a, b, ""))
			System.err.println("OR Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical OR using the two specified tensors and stores the result.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if either element is non-zero; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge or(String a, String b, String out) {
		if (!CuBridgeJNI.or(a, b, out))
			System.err.println("OR Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical NOT to the top tensor in the queue.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if the element is 0; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge not() {
		if (!CuBridgeJNI.not("", ""))
			System.err.println("NOT Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical NOT to the specified tensor.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if the element is 0; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge not(String a) {
		if (!CuBridgeJNI.not(a, ""))
			System.err.println("NOT Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Applies element-wise logical NOT to the specified tensor and stores the result.
	 * <ul>
	 *   <li>Each element is treated as boolean: non-zero as true, zero as false.</li>
	 *   <li>Returns 1 if the element is 0; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge not(String a, String out) {
		if (!CuBridgeJNI.not(a, out))
			System.err.println("NOT Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	// 통계 연산
	// 스칼라
	/**
	 * Computes the sum of all elements in the top tensor in the queue.
	 * <ul>
	 *   <li>This operation always sums across all axes (i.e., global sum).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sum() {
		if (!CuBridgeJNI.sum("", -1, ""))
			System.err.println("SUM Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the sum of all elements in the specified tensor.
	 * <ul>
	 *   <li>This operation always sums across all axes (i.e., global sum).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sum(String a) {
		if (!CuBridgeJNI.sum(a, -1, ""))
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the sum of all elements in the specified tensor and stores the result.
	 * <ul>
	 *   <li>This operation always sums across all axes (i.e., global sum).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sum(String a, String out) {
		if (!CuBridgeJNI.sum(a, -1, out))
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the mean of all elements in the top tensor in the queue.
	 * <ul>
	 *   <li>This operation always computes the global mean over all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mean() {
		if (!CuBridgeJNI.mean("", -1, ""))
			System.err.println("MEAN Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the mean of all elements in the specified tensor.
	 * <ul>
	 *   <li>This operation always computes the global mean over all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mean(String a) {
		if (!CuBridgeJNI.mean(a, -1, ""))
			System.err.println("MEAN Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the mean of all elements in the specified tensor and stores the result.
	 * <ul>
	 *   <li>This operation always computes the global mean over all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mean(String a, String out) {
		if (!CuBridgeJNI.mean(a, -1, out))
			System.err.println("MEAN Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the maximum of all elements in the top tensor in the queue.
	 * <ul>
	 *   <li>This operation always finds the global maximum across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge max() {
		if (!CuBridgeJNI.max("", -1, ""))
			System.err.println("MAX Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the maximum of all elements in the specified tensor.
	 * <ul>
	 *   <li>This operation always finds the global maximum across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge max(String a) {
		if (!CuBridgeJNI.max(a, -1, ""))
			System.err.println("MAX Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the maximum of all elements in the specified tensor and stores the result.
	 * <ul>
	 *   <li>This operation always finds the global maximum across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge max(String a, String out) {
		if (!CuBridgeJNI.max(a, -1, out))
			System.err.println("MAX Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the minimum of all elements in the top tensor in the queue.
	 * <ul>
	 *   <li>This operation always finds the global minimum across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge min() {
		if (!CuBridgeJNI.min("", -1, ""))
			System.err.println("MIN Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the minimum of all elements in the specified tensor.
	 * <ul>
	 *   <li>This operation always finds the global minimum across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge min(String a) {
		if (!CuBridgeJNI.min(a, -1, ""))
			System.err.println("MIN Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the minimum of all elements in the specified tensor and stores the result.
	 * <ul>
	 *   <li>This operation always finds the global minimum across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge min(String a, String out) {
		if (!CuBridgeJNI.min(a, -1, out))
			System.err.println("MIN Error: The " + a + " does not exist in Queue!");

		return instance;
	}
	
	/**
	 * Computes the variance of all elements in the top tensor in the queue.
	 * <ul>
	 *   <li>This operation always computes the global variance across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge var() {
		if (!CuBridgeJNI.var("", -1, ""))
			System.err.println("VAR Error: Tensor not exist in Queue!");

		return instance;
	}
	
	/**
	 * Computes the variance of all elements in the specified tensor.
	 * <ul>
	 *   <li>This operation always computes the global variance across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge var(String a) {
		if (!CuBridgeJNI.var(a, -1, ""))
			System.err.println("VAR Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the variance of all elements in the specified tensor and stores the result.
	 * <ul>
	 *   <li>This operation always computes the global variance across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge var(String a, String out) {
		if (!CuBridgeJNI.var(a, -1, out))
			System.err.println("VAR Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the standard deviation of all elements in the top tensor in the queue.
	 * <ul>
	 *   <li>This operation always computes the global standard deviation across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge std() {
		if (!CuBridgeJNI.std("", -1, ""))
			System.err.println("STD Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the standard deviation of all elements in the specified tensor.
	 * <ul>
	 *   <li>This operation always computes the global standard deviation across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge std(String a) {
		if (!CuBridgeJNI.std(a, -1, ""))
			System.err.println("STD Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the standard deviation of all elements in the specified tensor and stores the result.
	 * <ul>
	 *   <li>This operation always computes the global standard deviation across all axes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge std(String a, String out) {
		if (!CuBridgeJNI.std(a, -1, out))
			System.err.println("STD Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	// 텐서
	/**
	 * Computes the sum over all axes of the top tensor in the queue.
	 * <ul>
	 *   <li>If axis = -1, sums all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, computes the sum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sum(int axis) {
		if (!CuBridgeJNI.sum("", axis, ""))
			System.err.println("SUM Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the sum over all axes of the specified tensor.
	 * <ul>
	 *   <li>If axis = -1, sums all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, computes the sum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sum(String a, int axis) {
		if (!CuBridgeJNI.sum(a, axis, ""))
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the sum over all axes of the specified tensor and stores the result.
	 * <ul>
	 *   <li>If axis = -1, sums all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, computes the sum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sum(String a, int axis, String out) {
		if (!CuBridgeJNI.sum(a, axis, out))
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the mean over all axes of the top tensor in the queue.
	 * <ul>
	 *   <li>If axis = -1, computes the mean of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, computes the mean from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mean(int axis) {
		if (!CuBridgeJNI.mean("", axis, ""))
			System.err.println("MEAN_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the mean over all axes of the specified tensor.
	 * <ul>
	 *   <li>If axis = -1, computes the mean of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, computes the mean from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mean(String a, int axis) {
		if (!CuBridgeJNI.mean(a, axis, ""))
			System.err.println("MEAN_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the mean over all axes of the specified tensor and stores the result.
	 * <ul>
	 *   <li>If axis = -1, computes the mean of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, computes the mean from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mean(String a, int axis, String out) {
		if (!CuBridgeJNI.mean(a, axis, out))
			System.err.println("MEAN_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the maximum value over all axes of the top tensor in the queue.
	 * <ul>
	 * 	 <li>If axis = -1, finds the maximum of all elements in the tensor.</li>
	 *   <li>If axis = -1, finds the maximum of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, finds the maximum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge max(int axis) {
		if (!CuBridgeJNI.max("", axis, ""))
			System.err.println("MAX_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the maximum value over all axes of the specified tensor.
	 * <ul>
	 *   <li>If axis = -1, finds the maximum of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, finds the maximum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge max(String a, int axis) {
		if (!CuBridgeJNI.max(a, axis, ""))
			System.err.println("MAX_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the maximum value over all axes of the specified tensor and stores the result.
	 * <ul>
	 *   <li>If axis = -1, finds the maximum of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, finds the maximum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge max(String a, int axis, String out) {
		if (!CuBridgeJNI.max(a, axis, out))
			System.err.println("MAX_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the minimum value over all axes of the top tensor in the queue.
	 * <ul>
	 *   <li>If axis = -1, finds the minimum of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, finds the minimum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge min(int axis) {
		if (!CuBridgeJNI.min("", axis, ""))
			System.err.println("MIN_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the minimum value over all axes of the specified tensor.
	 * <ul>
	 *   <li>If axis = -1, finds the minimum of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, finds the minimum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge min(String a, int axis) {
		if (!CuBridgeJNI.min(a, axis, ""))
			System.err.println("MIN_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the minimum value over all axes of the specified tensor and stores the result.
	 * <ul>
	 *   <li>If axis = -1, finds the minimum of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, finds the minimum from the innermost axis up to and including the given axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge min(String a, int axis, String out) {
		if (!CuBridgeJNI.min(a, axis, out))
			System.err.println("MIN_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}
	
	/**
	 * Computes the variance over the specified axis of the top tensor in the queue.
	 * <ul>
	 *   <li>If axis = -1, computes the variance of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, aggregates all axes from the innermost to the given axis (inclusive) to compute the variance.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge var(int axis) {
		if (!CuBridgeJNI.var("", axis, ""))
			System.err.println("VAR_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the variance over the specified axis of the given tensor.
	 * <ul>
	 *   <li>If axis = -1, computes the variance of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, aggregates all axes from the innermost to the given axis (inclusive) to compute the variance.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge var(String a, int axis) {
		if (!CuBridgeJNI.var(a, axis, ""))
			System.err.println("VAR_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the variance over the specified axis of the given tensor and stores the result.
	 * <ul>
	 *   <li>If axis = -1, computes the variance of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, aggregates all axes from the innermost to the given axis (inclusive) to compute the variance.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis up to which variance is computed (inclusive); -1 means full reduction
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge var(String a, int axis, String out) {
		if (!CuBridgeJNI.var(a, axis, out))
			System.err.println("VAR_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the standard deviation over the specified axis of the top tensor in the queue.
	 * <ul>
	 *   <li>If axis = -1, computes the standard deviation of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, aggregates all axes from the innermost to the given axis (inclusive) to compute the std.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge std(int axis) {
		if (!CuBridgeJNI.std("", axis, ""))
			System.err.println("STD_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the standard deviation over the specified axis of the given tensor.
	 * <ul>
	 *   <li>If axis = -1, computes the standard deviation of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, aggregates all axes from the innermost to the given axis (inclusive) to compute the std.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the dimension along which to compute the sum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge std(String a, int axis) {
		if (!CuBridgeJNI.std(a, axis, ""))
			System.err.println("STD_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Computes the standard deviation over the specified axis of the given tensor and stores the result.
	 * <ul>
	 *   <li>If axis = -1, computes the standard deviation of all elements in the tensor.</li>
	 *   <li>If axis ≥ 0, aggregates all axes from the innermost to the given axis (inclusive) to compute the std.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis up to which std is computed (inclusive); -1 means full reduction
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge std(String a, int axis, String out) {
		if (!CuBridgeJNI.std(a, axis, out))
			System.err.println("STD_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Finds the index of the maximum value along the specified axis of the top tensor in the queue.
	 * <ul>
	 *   <li>This function does not support axis = -1.</li>
	 *   <li>The maximum is computed along the specified axis only.</li>
	 *   <li>Returns the index tensor with the same shape except for the reduced axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the axis along which to find the index of the maximum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge argmax(int axis) {
		if (!CuBridgeJNI.argmax("", axis, ""))
			System.err.println("ARGMAX Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Finds the index of the maximum value along the specified axis of the given tensor.
	 * <ul>
	 *   <li>This function does not support axis = -1.</li>
	 *   <li>The maximum is computed along the specified axis only.</li>
	 *   <li>Returns the index tensor with the same shape except for the reduced axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the index of the maximum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge argmax(String a, int axis) {
		if (!CuBridgeJNI.argmax(a, axis, ""))
			System.err.println("ARGMAX Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Finds the index of the maximum value along the specified axis of the given tensor and stores the result.
	 * <ul>
	 *   <li>This function does not support axis = -1.</li>
	 *   <li>The maximum is computed along the specified axis only.</li>
	 *   <li>Returns the index tensor with the same shape except for the reduced axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the index of the maximum
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge argmax(String a, int axis, String out) {
		if (!CuBridgeJNI.argmax(a, axis, out))
			System.err.println("ARGMAX Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Finds the index of the minimum value along the specified axis of the top tensor in the queue.
	 * <ul>
	 *   <li>This function does not support axis = -1.</li>
	 *   <li>The minimum is computed along the specified axis only.</li>
	 *   <li>Returns the index tensor with the same shape except for the reduced axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the axis along which to find the index of the minimum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge argmin(int axis) {
		if (!CuBridgeJNI.argmin("", axis, ""))
			System.err.println("ARGMIN Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * Finds the index of the minimum value along the specified axis of the given tensor.
	 * <ul>
	 *   <li>This function does not support axis = -1.</li>
	 *   <li>The minimum is computed along the specified axis only.</li>
	 *   <li>Returns the index tensor with the same shape except for the reduced axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the index of the minimum
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge argmin(String a, int axis) {
		if (!CuBridgeJNI.argmin(a, axis, ""))
			System.err.println("ARGMIN Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * Finds the index of the minimum value along the specified axis of the given tensor and stores the result.
	 * <ul>
	 *   <li>This function does not support axis = -1.</li>
	 *   <li>The minimum is computed along the specified axis only.</li>
	 *   <li>Returns the index tensor with the same shape except for the reduced axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the index of the minimum
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge argmin(String a, int axis, String out) {
		if (!CuBridgeJNI.argmin(a, axis, out))
			System.err.println("ARGMIN Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	// 함수
	/**
	 * <ul>
	 *   <li>Applies sine (sin x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sin() {
		if (!CuBridgeJNI.sin("", ""))
			System.err.println("SIN Error: Tensor not exist in Queue!");

		return instance;
	}
	
	/**
	 * <ul>
	 *   <li>Applies sine (sin x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sin(String name) {
		if (!CuBridgeJNI.sin(name, ""))
			System.err.println("SIN Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies sine (sin x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sin(String name, String out) {
		if (!CuBridgeJNI.sin(name, out))
			System.err.println("SIN Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies cosine (cos x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cos() {
		if (!CuBridgeJNI.cos("", ""))
			System.err.println("COS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies cosine (cos x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cos(String name) {
		if (!CuBridgeJNI.cos(name, ""))
			System.err.println("COS Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies cosine (cos x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cos(String name, String out) {
		if (!CuBridgeJNI.cos(name, out))
			System.err.println("COS Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies tangent (tan x) to the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge tan() {
		if (!CuBridgeJNI.tan("", ""))
			System.err.println("TAN Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies tangent (tan x) to the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge tan(String name) {
		if (!CuBridgeJNI.tan(name, ""))
			System.err.println("TAN Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies tangent (tan x) to the specified tensor and stores the result under a new name.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge tan(String name, String out) {
		if (!CuBridgeJNI.tan(name, out))
			System.err.println("TAN Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies step function to the top tensor in the queue.</li>
	 *   <li>Returns 1 if the input is greater than 0; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge step() {
		if (!CuBridgeJNI.step("", ""))
			System.err.println("STEP Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies step function to the specified tensor.</li>
	 *   <li>Returns 1 if the input is greater than 0; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge step(String name) {
		if (!CuBridgeJNI.step(name, ""))
			System.err.println("STEP Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies step function to the specified tensor and stores the result under a new name.</li>
	 *   <li>Returns 1 if the input is greater than 0; otherwise, returns 0.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge step(String name, String out) {
		if (!CuBridgeJNI.step(name, out))
			System.err.println("STEP Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies sigmoid function to the top tensor in the queue.</li>
	 *   <li>Uses the formula: 1 / (1 + exp(-x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sigmoid() {
		if (!CuBridgeJNI.sigmoid("", ""))
			System.err.println("SIGMOID Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies sigmoid function to the specified tensor.</li>
	 *   <li>Uses the formula: 1 / (1 + exp(-x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sigmoid(String name) {
		if (!CuBridgeJNI.sigmoid(name, ""))
			System.err.println("SIGMOID Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies sigmoid function to the specified tensor and stores the result under a new name.</li>
	 *   <li>Uses the formula: 1 / (1 + exp(-x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge sigmoid(String name, String out) {
		if (!CuBridgeJNI.sigmoid(name, out))
			System.err.println("SIGMOID Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies hyperbolic tangent function (tanh) to the top tensor in the queue.</li>
	 *   <li>Uses the formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge tanh() {
		if (!CuBridgeJNI.tanh("", ""))
			System.err.println("TANH Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies tanh function to the specified tensor.</li>
	 *   <li>Uses the formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge tanh(String name) {
		if (!CuBridgeJNI.tanh(name, ""))
			System.err.println("TANH Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies tanh function to the specified tensor and stores the result under a new name.</li>
	 *   <li>Uses the formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge tanh(String name, String out) {
		if (!CuBridgeJNI.tanh(name, out))
			System.err.println("TANH Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies ReLU (Rectified Linear Unit) to the top tensor in the queue.</li>
	 *   <li>ReLU(x) = max(0, x).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge relu() {
		if (!CuBridgeJNI.reLu("", ""))
			System.err.println("RELU Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies ReLU (Rectified Linear Unit) to the specified tensor.</li>
	 *   <li>ReLU(x) = max(0, x).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge relu(String name) {
		if (!CuBridgeJNI.reLu(name, ""))
			System.err.println("RELU Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies ReLU (Rectified Linear Unit) to the specified tensor and stores the result under a new name.</li>
	 *   <li>ReLU(x) = max(0, x).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge relu(String name, String out) {
		if (!CuBridgeJNI.reLu(name, out))
			System.err.println("RELU Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies Leaky ReLU activation to the top tensor in the queue.</li>
	 *   <li>Returns x if x > 0, otherwise returns αx (α is 0.01).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge leakrelu() {
		if (!CuBridgeJNI.leakReLu("", ""))
			System.err.println("LEAKRELU Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies Leaky ReLU activation to the specified tensor.</li>
	 *   <li>Returns x if x > 0, otherwise returns αx (α is 0.01).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge leakrelu(String name) {
		if (!CuBridgeJNI.leakReLu(name, ""))
			System.err.println("LEAKRELU Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies Leaky ReLU activation to the specified tensor and stores the result under a new name.</li>
	 *   <li>Returns x if x > 0, otherwise returns αx (α is 0.01).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge leakrelu(String name, String out) {
		if (!CuBridgeJNI.leakReLu(name, out))
			System.err.println("LEAKRELU Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies softmax function to the top tensor in the queue along the last axis.</li>
	 *   <li>Transforms input into a probability distribution over classes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge softmax() {
		if (!CuBridgeJNI.softmax("", ""))
			System.err.println("SOFTMAX Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies softmax function to the specified tensor along the last axis.</li>
	 *   <li>Transforms input into a probability distribution over classes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge softmax(String name) {
		if (!CuBridgeJNI.softmax(name, ""))
			System.err.println("SOFTMAX Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies softmax function to the specified tensor along the last axis and stores the result.</li>
	 *   <li>Transforms input into a probability distribution over classes.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge softmax(String name, String out) {
		if (!CuBridgeJNI.softmax(name, out))
			System.err.println("SOFTMAX Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies Softplus activation to the top tensor in the queue.</li>
	 *   <li>Uses the formula: log(1 + exp(x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge softplus() {
		if (!CuBridgeJNI.softplus("", ""))
			System.err.println("SOFTPLUS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies Softplus activation to the specified tensor.</li>
	 *   <li>Uses the formula: log(1 + exp(x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge softplus(String name) {
		if (!CuBridgeJNI.softplus(name, ""))
			System.err.println("SOFTPLUS Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies Softplus activation to the specified tensor and stores the result under a new name.</li>
	 *   <li>Uses the formula: log(1 + exp(x)).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge softplus(String name, String out) {
		if (!CuBridgeJNI.softplus(name, out))
			System.err.println("SOFTPLUS Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies exponential function (exp) to the top tensor in the queue.</li>
	 *   <li>Uses the formula: exp(x) = e^x for each element.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge exp() {
		if (!CuBridgeJNI.exp("", ""))
			System.err.println("EXP Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies exponential function (exp) to the specified tensor.</li>
	 *   <li>Uses the formula: exp(x) = e^x for each element.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge exp(String name) {
		if (!CuBridgeJNI.exp(name, ""))
			System.err.println("EXP Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies exponential function (exp) to the specified tensor and stores the result under a new name.</li>
	 *   <li>Uses the formula: exp(x) = e^x for each element.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge exp(String name, String out) {
		if (!CuBridgeJNI.exp(name, out))
			System.err.println("EXP Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies rounding to the nearest integer for each element of the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge round() {
		if (!CuBridgeJNI.round("", ""))
			System.err.println("ROUND Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies rounding to the nearest integer for each element of the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge round(String name) {
		if (!CuBridgeJNI.round(name, ""))
			System.err.println("ROUND Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies rounding to the nearest integer for each element of the specified tensor and stores the result.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge round(String name, String out) {
		if (!CuBridgeJNI.round(name, out))
			System.err.println("ROUND Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies ceiling operation (rounding up) for each element of the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ceil() {
		if (!CuBridgeJNI.ceil("", ""))
			System.err.println("CEIL Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies ceiling operation (rounding up) for each element of the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ceil(String name) {
		if (!CuBridgeJNI.ceil(name, ""))
			System.err.println("CEIL Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies ceiling operation (rounding up) for each element of the specified tensor and stores the result.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge ceil(String name, String out) {
		if (!CuBridgeJNI.ceil(name, out))
			System.err.println("CEIL Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies floor operation (rounding down) for each element of the top tensor in the queue.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge floor() {
		if (!CuBridgeJNI.floor("", ""))
			System.err.println("FLOOR Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies floor operation (rounding down) for each element of the specified tensor.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge floor(String name) {
		if (!CuBridgeJNI.floor(name, ""))
			System.err.println("FLOOR Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Applies floor operation (rounding down) for each element of the specified tensor and stores the result.</li>
	 *   <li>All operations are performed element-wise on the tensor.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge floor(String name, String out) {
		if (!CuBridgeJNI.floor(name, out))
			System.err.println("FLOOR Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Transposes the last two axes of the top tensor in the queue.</li>
	 *   <li>If the input is a 1D vector, it is automatically reshaped to a column vector (n, 1) before transposition.</li>
	 *   <li>This operation does not affect leading axes (only the two innermost dimensions are swapped).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge transpose() {
		if (!CuBridgeJNI.transpose("", ""))
			System.err.println("TRANSPOSE Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Transposes the last two axes of the specified tensor.</li>
	 *   <li>If the input is a 1D vector, it is automatically reshaped to a column vector (n, 1) before transposition.</li>
	 *   <li>This operation does not affect leading axes (only the two innermost dimensions are swapped).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge transpose(String name) {
		if (!CuBridgeJNI.transpose(name, ""))
			System.err.println("TRANSPOSE Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Transposes the last two axes of the specified tensor and stores the result under a new name.</li>
	 *   <li>If the input is a 1D vector, it is automatically reshaped to a column vector (n, 1) before transposition.</li>
	 *   <li>This operation does not affect leading axes (only the two innermost dimensions are swapped).</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge transpose(String name, String out) {
		if (!CuBridgeJNI.transpose(name, out))
			System.err.println("TRANSPOSE Error: The " + name + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Compresses the specified axis of the top tensor by summing its values.</li>
	 *   <li>Only the specified axis is compressed; all other axes remain unchanged.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the axis to compress
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge compress(int axis) {
		if (!CuBridgeJNI.compress("", axis, false, ""))
			System.err.println("COMPRESS_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Compresses the specified axis of the given tensor by summing its values.</li>
	 *   <li>Only the specified axis is compressed; all other axes remain unchanged.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis to compress
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge compress(String a, int axis) {
		if (!CuBridgeJNI.compress(a, axis, false, ""))
			System.err.println("COMPRESS_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Compresses the specified axis of the given tensor by summing its values and stores the result under a new name.</li>
	 *   <li>Only the specified axis is compressed; all other axes remain unchanged.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis to compress
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge compress(String a, int axis, String out) {
		if (!CuBridgeJNI.compress(a, axis, false, out))
			System.err.println("COMPRESS_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Compresses the specified axis of the top tensor by either summing or averaging its values.</li>
	 *   <li>Set {@code avg} to true to use average, or false to use sum.</li>
	 *   <li>Only the specified axis is compressed; all other axes remain unchanged.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param axis the axis to compress
	 * @param avg whether to average (true) or sum (false)
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge compress(int axis, boolean avg) {
		if (!CuBridgeJNI.compress("", axis, avg, ""))
			System.err.println("COMPRESS_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Compresses the specified axis of the given tensor by either summing or averaging its values.</li>
	 *   <li>Set {@code avg} to true to use average, or false to use sum.</li>
	 *   <li>Only the specified axis is compressed; all other axes remain unchanged.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis to compress
	 * @param avg whether to average (true) or sum (false)
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge compress(String a, int axis, boolean avg) {
		if (!CuBridgeJNI.compress(a, axis, avg, ""))
			System.err.println("COMPRESS_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Compresses the specified axis of the given tensor by either summing or averaging its values and stores the result.</li>
	 *   <li>Set {@code avg} to true to use average, or false to use sum.</li>
	 *   <li>Only the specified axis is compressed; all other axes remain unchanged.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis to compress
	 * @param avg whether to average (true) or sum (false)
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge compress(String a, int axis, boolean avg, String out) {
		if (!CuBridgeJNI.compress(a, axis, avg, out))
			System.err.println("COMPRESS_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Repeats the elements along the specified axis of the top tensor {@code n} times.</li>
	 *   <li>This operation only works if the size of the given axis is exactly 1.</li>
	 *   <li>If the condition is not met or the tensor is not found, an error message is printed and the operation fails.</li>
	 * </ul>
	 * @param axis the axis to repeat
	 * @param n the number of repetitions
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge expand(int axis, int n) {
		if (!CuBridgeJNI.expand("", axis, n, ""))
			System.err.println("REPEAT_AXIS Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Repeats the elements along the specified axis of the given tensor {@code n} times.</li>
	 *   <li>This operation only works if the size of the given axis is exactly 1.</li>
	 *   <li>If the condition is not met or the tensor is not found, an error message is printed and the operation fails.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis to repeat
	 * @param n the number of repetitions
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge expand(String a, int axis, int n) {
		if (!CuBridgeJNI.expand(a, axis, n, ""))
			System.err.println("REPEAT_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Repeats the elements along the specified axis of the given tensor {@code n} times and stores the result under a new name.</li>
	 *   <li>This operation only works if the size of the given axis is exactly 1.</li>
	 *   <li>If the condition is not met or the tensor is not found, an error message is printed and the operation fails.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param axis the axis to repeat
	 * @param n the number of repetitions
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge expand(String a, int axis, int n, String out) {
		if (!CuBridgeJNI.expand(a, axis, n, out))
			System.err.println("REPEAT_AXIS Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	// 신경망 특화
	/**
	 * <ul>
	 *   <li>Computes the Mean Squared Error (MSE) loss between the top two tensors in the queue.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mse() {
		if (!CuBridgeJNI.mse("", "", ""))
			System.err.println("MSE Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the MSE loss between the specified prediction tensor and the top target tensor.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param yh the name of the predicted output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mse(String yh) {
		if (!CuBridgeJNI.mse(yh, "", ""))
			System.err.println("MSE Error: The " + yh + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the MSE loss between the specified prediction and target tensors.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mse(String yh, String y) {
		if (!CuBridgeJNI.mse(yh, y, ""))
			System.err.println("MSE Error: The " + yh + " or " + y + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the MSE loss between the specified prediction and target tensors and stores the result.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss in the specified output tensor.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge mse(String yh, String y, String out) {
		if (!CuBridgeJNI.mse(yh, y, out))
			System.err.println("MSE Error: The " + yh + " or " + y + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the Cross-Entropy Error (CEE) loss between the top two tensors in the queue.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cee() {
		if (!CuBridgeJNI.cee("", "", ""))
			System.err.println("CEE Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the CEE loss between the specified prediction tensor and the top target tensor.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param yh the name of the predicted output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cee(String yh) {
		if (!CuBridgeJNI.cee(yh, "", ""))
			System.err.println("CEE Error: The " + yh + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the CEE loss between the specified prediction and target tensors.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cee(String yh, String y) {
		if (!CuBridgeJNI.cee(yh, y, ""))
			System.err.println("CEE Error: The " + yh + " or " + y + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Computes the CEE loss between the specified prediction and target tensors and stores the result.</li>
	 *   <li>Returns a scalar (1x1 tensor) representing the total loss in the specified output tensor.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge cee(String yh, String y, String out) {
		if (!CuBridgeJNI.cee(yh, y, out))
			System.err.println("CEE Error: The " + yh + " or " + y + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs affine transformation: a · b + c.</li>
	 *   <li>Ensure the last axis of tensor a matches the second-to-last axis of tensor b for valid dot product.</li>
	 *   <li>Tensor c must have the same size as the number of columns in the resulting matrix (i.e., last axis of b).</li>
	 *   <li>Broadcasting is allowed for tensor c; its broadcasting setting does not affect correctness.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge affine() {
		if (!CuBridgeJNI.affine("", "", "", ""))
			System.err.println("AFFINE Error: Tensor not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs affine transformation: a · b + c using the specified input tensor a.</li>
	 *   <li>The dot product a · b requires the last axis of a to match the second-to-last axis of b.</li>
	 *   <li>Tensor c must match the output column size (last axis of b), and may be broadcasted.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge affine(String a) {
		if (!CuBridgeJNI.affine(a, "", "", ""))
			System.err.println("AFFINE Error: The " + a + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs affine transformation: a · b + c using the specified tensors a and b.</li>
	 *   <li>Ensure the last axis of a equals the second-to-last axis of b for valid matrix multiplication.</li>
	 *   <li>Tensor c must match the number of columns in b and may be broadcasted if needed.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param b the name of the weight tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge affine(String a, String b) {// 자동으로 c는 0으로 계산하도록?
		if (!CuBridgeJNI.affine(a, b, "", ""))
			System.err.println("AFFINE Error: The " + a + " or " + b + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs affine transformation: a · b + c using the specified tensors a, b, and c.</li>
	 *   <li>Valid dot product requires last axis of a = second-to-last axis of b.</li>
	 *   <li>c must have the same size as the output column dimension (last axis of b), and broadcasting is allowed.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param b the name of the weight tensor
	 * @param c the name of the bias tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge affine(String a, String b, String c) {
		if (!CuBridgeJNI.affine(a, b, c, ""))
			System.err.println("AFFINE Error: The " + a + " or " + b + " or " + c + " does not exist in Queue!");

		return instance;
	}

	/**
	 * <ul>
	 *   <li>Performs affine transformation: a · b + c and stores the result in the specified output tensor.</li>
	 *   <li>Valid dot product requires last axis of a = second-to-last axis of b.</li>
	 *   <li>c must match the output's column count and may be broadcasted regardless of its setting.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * @param a the name of the input tensor
	 * @param b the name of the weight tensor
	 * @param c the name of the bias tensor
	 * @param out the name for the output tensor
	 * @return public CuBridge instance for chaining
	 */
	public CuBridge affine(String a, String b, String c, String out) {
		if (!CuBridgeJNI.affine(a, b, c, out))
			System.err.println("AFFINE Error: The " + a + " or " + b + " or " + c + " does not exist in Queue!");

		return instance;
	}

	// 사용자 정의 함수
	// 이제 여기에 오버로딩 함수들 섞어서 다단계 함수들 만들기 가능

}
