package CuBridge;

import java.util.UUID;

public class CuBridge {
	private static final CuBridge instance = new CuBridge();

	private CuBridge() {
		loadConst();
	}
	
	private void loadConst() {
	    put(1.0, "_ONE", -1);
	    put(2.0, "_TWO", -1);
	    put(3.0, "_THREE", -1);
	    put(4.0, "_FOUR", -1);
	    put(5.0, "_FIVE", -1);
	    put(6.0, "_SIX", -1);
	    put(7.0, "_SEVEN", -1);
	    put(8.0, "_EIGHT", -1);
	    put(9.0, "_NINE", -1);
	    put(0.0, "_ZERO", -1);
	    put(0.5, "_HALF", -1);
	    put(100.0, "_HUNDRED", -1);
	    put(255.0, "_MAXPIXEL", -1);
	    put(-1.0, "_NEG", -1);
	    put(1e-6, "_EPSILON", -1);
	    put(0.001, "_RATE", -1);
	    put(3.14159265359, "_PI", -1);
	    put(2.718281, "_E", -1);
	}

	private String genRandomName() {
	    return "TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
	}
		
	/**
	 * Returns the singleton instance of CuBridge.
	 *
	 * @return the global CuBridge instance (singleton)
	 */
	public static CuBridge getInstance() {
		return instance;
	}

	/**
	 * Forces computation mode to CPU.
	 * <p>
	 * Disables auto-detection and executes all operations on the CPU.
	 * Tensor memory is always stored in RAM.
	 */
	public void selectCPU() {
		CuBridgeJNI.setCAL(false);
		CuBridgeJNI.refresh();
	}

	/**
	 * Forces computation mode to GPU.
	 * <p>
	 * Disables auto-detection and executes all operations using CUDA (if available).
	 * Note: All tensor data is stored in RAM. During GPU execution,
	 * necessary data is temporarily transferred to VRAM for computation and
	 * results are copied back to RAM automatically.
	 */
	public void selectGPU() {
		CuBridgeJNI.setCAL(true);
		CuBridgeJNI.refresh();
	}

	/**
	 * Resets CuBridge to auto-detection mode.
	 * <p>
	 * CuBridge will automatically detect CUDA availability
	 * and select between CPU and GPU computation accordingly.
	 * All tensor memory remains in RAM regardless of the compute mode.
	 */
	public void envReset() {
		CuBridgeJNI.setAuto();
		CuBridgeJNI.refresh();
	}

	/**
	 * Prints the current system and CuBridge environment status.
	 * <p>
	 * This includes system information and CuBridge runtime configuration,
	 * such as:
	 * </p>
	 * <ul>
	 *   <li>Operating system and physical RAM</li>
	 *   <li>Detected GPU name and available VRAM (if CUDA is available)</li>
	 *   <li>Installed CUDA Driver and Runtime versions</li>
	 *   <li>CuBridge detection mode (auto/manual)</li>
	 *   <li>Current compute device (CPU or GPU)</li>
	 * </ul>
	 * <p>
	 * Note: All tensor memory is managed in RAM regardless of compute device.
	 * </p>
	 */
	public void getEnvironmentStatus() {
		boolean auto = CuBridgeJNI.getENV();
		boolean gpuCompute = CuBridgeJNI.getCAL();

		StringBuilder sb = new StringBuilder();
		sb.append("[System 환경 상태]\n");
		sb.append(CuBridgeJNI.getSysInfo());
		sb.append("\n[CuBridge 환경 상태]\n");
		sb.append("- 자동 감지 모드: ").append(auto ? "O" : "X").append("\n");
		sb.append("- 연산 방식: ").append(gpuCompute ? "GPU" : "CPU").append("\n");

		System.out.println(sb.toString());
	}

	/**
	 * Clears all tensors from the internal queue.
	 */
	public void clear() {
		CuBridgeJNI.clear();
		return;
	}

	/**
	 * Prints all tensors currently stored in the queue (excluding constants).
	 * Displays only non-constant tensors with their auto-generated or user-defined names.
	 */
	public void visualQueue() {
		System.out.println(CuBridgeJNI.visualQueue());
	}

	/**
	 * Prints all tensors currently stored in the queue (including constants).
	 * Displays every tensor with its associated name in the order they were added.
	 */
	public void visualQueueAll() {
		System.out.println(CuBridgeJNI.visualQueueAll());
	}

	/**
	 * Prints all tensors currently stored in the buffer (excluding constants).
	 * Displays only non-constant tensors with their auto-generated or user-defined names.
	 */
	public void visualBuffer() {
		System.out.println(CuBridgeJNI.visualBuffer());
	}

	/**
	 * Prints all tensors currently stored in the buffer (including constants).
	 * Displays every tensor with its associated name in the order they were added.
	 */
	public void visualBufferAll() {
		System.out.println(CuBridgeJNI.visualBufferAll());
	}

	// put에서, 만일 ""일 경우 난수를 넣어야 한다

	/**
	 * Stores an integer scalar tensor.
	 * <p>
	 * Full parameter: {@code put(int data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * <li>name is auto-generated</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the integer value to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(int data) {
		return put(new Tensor(data), true);
	}

	/**
	 * Stores an integer scalar tensor.
	 * <p>
	 * Full parameter: {@code put(int data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the integer value to store
	 * @param name the tensor name (must be unique and non-empty)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(int data, String name) {
		return put(new Tensor(data), name, 1, true);
	}

	/**
	 * Stores an integer scalar tensor.
	 * <p>
	 * Full parameter: {@code put(int data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * </ul>
	 * <br>
	 * Constants must satisfy both of the following:
	 * <ul>
	 * <li>usageCount == -1</li>
	 * <li>name must start with an underscore ("_")</li>
	 * </ul>
	 * </p>
	 *
	 * @param data        the integer value to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1 for constants)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(int data, String name, int usageCount) {
		return put(new Tensor(data), name, usageCount, true);
	}

	/**
	 * Stores a double scalar tensor.
	 * <p>
	 * Full parameter: {@code put(double data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * <li>name is auto-generated</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the double value to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(double data) {
		return put(new Tensor(data), true);
	}

	/**
	 * Stores a double scalar tensor.
	 * <p>
	 * Full parameter: {@code put(double data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the double value to store
	 * @param name the tensor name (must be unique and non-empty)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(double data, String name) {
		return put(new Tensor(data), name, 1, true);
	}

	/**
	 * Stores a double scalar tensor.
	 * <p>
	 * Full parameter: {@code put(double data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * </ul>
	 * <br>
	 * Constants must satisfy both of the following:
	 * <ul>
	 * <li>usageCount == -1</li>
	 * <li>name must start with an underscore ("_")</li>
	 * </ul>
	 * </p>
	 *
	 * @param data        the double value to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1 for constants)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(double data, String name, int usageCount) {
		return put(new Tensor(data), name, usageCount, true);
	}

	/**
	 * Stores a tensor with default configuration.
	 * <p>
	 * Full parameter: {@code put(Tensor data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = false (not broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * <li>name is auto-generated</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the tensor to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data) {
		return put(data, false);
	}

	/**
	 * Stores a tensor with broadcast option and a temporary name.
	 * <p>
	 * Full parameter: {@code put(Tensor data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>usageCount = 1 (default)</li>
	 * <li>name is auto-generated</li>
	 * </ul>
	 * </p>
	 *
	 * @param data      the tensor to store
	 * @param broadcast whether the tensor is broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, boolean broadcast) {
		CuBridgeJNI.put(data.toArray(), data.getShape(), data.getSize(), data.getAxis(), 1, genRandomName(), broadcast);
		return instance;
	}

	/**
	 * Stores a named tensor with default usage and no broadcasting.
	 * <p>
	 * Full parameter: {@code put(Tensor data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>usageCount = 1 (default)</li>
	 * <li>broadcast = false (not broadcastable)</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the tensor to store
	 * @param name the tensor name (must be unique and non-empty)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name) {
		return put(data, name, 1, false);
	}

	/**
	 * Stores a named tensor with default usage count.
	 * <p>
	 * Full parameter: {@code put(Tensor data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>usageCount = 1 (default)</li>
	 * </ul>
	 * </p>
	 *
	 * @param data      the tensor to store
	 * @param name      the tensor name (must be unique and non-empty)
	 * @param broadcast whether the tensor is broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, boolean broadcast) {
		return put(data, name, 1, broadcast);
	}

	/**
	 * Stores a named tensor with a specific usage count.
	 * <p>
	 * Full parameter: {@code put(Tensor data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = false (not broadcastable)</li>
	 * </ul>
	 * <br>
	 * Constants must satisfy both of the following:
	 * <ul>
	 * <li>usageCount == -1</li>
	 * <li>name must start with an underscore ("_")</li>
	 * </ul>
	 * </p>
	 *
	 * @param data        the tensor to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1 for constants)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, int usageCount) {
		return put(data, name, usageCount, false);
	}

	/**
	 * Stores a tensor with complete configuration.
	 *
	 * This method registers a tensor in the internal queue with its data, shape,
	 * name, usage count, and broadcast flag.
	 *
	 * Constants must satisfy both of the following:
	 * - usageCount must be -1
	 * - name must start with an underscore ("_")
	 *
	 * @param data        the tensor to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0, or -1 for constants)
	 * @param broadcast   whether the tensor is broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, int usageCount, boolean broadcast) {
		if (usageCount == 0) {
			System.err.println("Error: Please UsageCount modify.");
			return instance;
		}
		
		if((usageCount < 0) && !name.startsWith("_")) {
	        System.err.println("[Error] Constant tensor must start with '_'. Given name: " + name);
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
	 * Retrieves and removes the best available tensor from the internal queue.
	 * <p>
	 * Full parameter: {@code get(String name)}<br>
	 * This version:
	 * <ul>
	 * <li>If multiple tensors exist, the most suitable one is automatically selected.</li>
	 * <li>If the queue is empty, an error message is printed and {@code null} is returned.</li>
	 * </ul>
	 * </p>
	 *
	 * @return the retrieved tensor, or {@code null} if the queue is empty
	 */
	public Tensor get() {
		if (!CuBridgeJNI.pop("")) {
			System.err.println("Error: Queue is empty!");
			return null;
		}
		return getTensor("");
	}

	/**
	 * Retrieves and removes the top tensor with the specified name from the queue.
	 * <p>
	 * Full parameter: {@code get(String name)}<br>
	 * This version:
	 * <ul>
	 * <li>If no tensor with that name exists in the queue, an error is printed and {@code null} is returned.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the tensor to retrieve
	 * @return the retrieved tensor, or {@code null} if not found
	 */
	public Tensor get(String name) {
		if (!CuBridgeJNI.pop(name)) {
			System.err.println("Error: The " + name + " is not exist in Queue!");
			return null;
		}

		return getTensor(name);
	}

	private Tensor getTensor(String name) {
		double[] data = CuBridgeJNI.getData(name);
		int[] shape = CuBridgeJNI.getShape(name);
		CuBridgeJNI.bufferClean();

		return new Tensor(data, shape);
	}

	/**
	 * Updates the usage count of a tensor in the execution queue.
	 * <p>
	 * Full parameter: {@code duple(String name, int usageCount)}<br>
	 * This operation:
	 * <ul>
	 * <li>Changes how many times the specified tensor can be used.</li>
	 * <li>Only the internal usage metadata is updated; no duplication occurs.</li>
	 * <li>If the usage count is less than 1, or the tensor does not exist, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name        the name of the tensor in the queue
	 * @param usageCount  number of additional times the tensor can be used (≥ 1)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge duple(String name, int usageCount) {
		if (usageCount < 1) {
			System.err.println("Error: Tensor '" + name + "' cannot be duplicated; invalid usage count.");
			return instance;
		}

		if (!CuBridgeJNI.duple(name, usageCount))
			System.err.println("Error: Failed to update usage count for tensor '" + name + "' in the queue.");

		return instance;
	}

	/**
	 * Updates the broadcastable flag of a tensor in the execution queue.
	 * <p>
	 * Full parameter: {@code broad(String name, boolean broad)}<br>
	 * This operation:
	 * <ul>
	 * <li>Changes whether the specified tensor is treated as broadcastable in binary operations.</li>
	 * <li>Only the broadcast flag is updated; the tensor itself remains unchanged.</li>
	 * <li>If the tensor does not exist in the queue, the operation fails silently.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the tensor in the queue
	 * @param broad  whether to mark the tensor as broadcastable
	 * @return CuBridge instance for chaining
	 */
	public CuBridge broad(String name, boolean broad) {
		if(!CuBridgeJNI.broad(name, broad))
			System.err.println("Error: Failed to update broad for tensor '" + name + "' in the queue.");
		return instance;
	}
	
	/**
	 * Reshapes the shape and size (slen) of the specified tensor.
	 * <p>
	 * Full parameter: {@code reshape(String name, int[] shape)}<br>
	 * This operation:
	 * <ul>
	 *   <li>Changes the internal shape and size (slen) metadata of the specified tensor.</li>
	 *   <li>The underlying data remains unchanged; only how it is interpreted changes.</li>
	 *   <li>If the tensor does not exist in the queue, the operation fails silently.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the tensor in the queue
	 * @param shape  the new shape to apply
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge reshape(String name, int[] shape) {
		if(!CuBridgeJNI.reshape(name, shape, shape.length))
			System.err.println("Error: Failed to update shape for tensor '" + name + "' in the queue.");
		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code abs(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge abs() {
		if (!CuBridgeJNI.abs("", genRandomName()))
			System.err.println("[ERROR][ABS][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the specified tensor.
	 * <p>
	 * Full parameter: {@code abs(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge abs(String a) {
		if (!CuBridgeJNI.abs(a, genRandomName()))
			System.err.println("[ERROR][ABS][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code abs(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge abs(String a, String out) {
		if (!CuBridgeJNI.abs(a, out))
			System.err.println("[ERROR][ABS][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code neg(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge neg() {
		if (!CuBridgeJNI.neg("", genRandomName()))
			System.err.println("[ERROR][NEG][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code neg(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge neg(String a) {
		if (!CuBridgeJNI.neg(a, genRandomName()))
			System.err.println("[ERROR][NEG][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code neg(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge neg(String a, String out) {
		if (!CuBridgeJNI.neg(a, out))
			System.err.println("[ERROR][NEG][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies squaring (x²) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge square() {
		if (!CuBridgeJNI.square("", genRandomName()))
			System.err.println("[ERROR][SQUARE][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies squaring (x²) to the specified tensor.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge square(String a) {
		if (!CuBridgeJNI.square(a, genRandomName()))
			System.err.println("[ERROR][SQUARE][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies squaring (x²) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge square(String a, String out) {
		if (!CuBridgeJNI.square(a, out))
			System.err.println("[ERROR][SQUARE][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code sqrt(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sqrt() {
		if (!CuBridgeJNI.sqrt("", genRandomName()))
			System.err.println("[ERROR][SQRT][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code sqrt(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sqrt(String a) {
		if (!CuBridgeJNI.sqrt(a, genRandomName()))
			System.err.println("[ERROR][SQRT][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code sqrt(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sqrt(String a, String out) {
		if (!CuBridgeJNI.sqrt(a, out))
			System.err.println("[ERROR][SQRT][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies base-10 logarithm (log₁₀x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log() {
		if (!CuBridgeJNI.log("", genRandomName()))
			System.err.println("[ERROR][LOG][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies base-10 logarithm (log₁₀x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log(String a) {
		if (!CuBridgeJNI.log(a, genRandomName()))
			System.err.println("[ERROR][LOG][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies base-10 logarithm (log₁₀x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log(String a, String out) {
		if (!CuBridgeJNI.log(a, out))
			System.err.println("[ERROR][LOG][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies base-2 logarithm (log₂x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log2() {
		if (!CuBridgeJNI.log2("", genRandomName()))
			System.err.println("[ERROR][LOG2][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies base-2 logarithm (log₂x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log2(String a) {
		if (!CuBridgeJNI.log2(a, genRandomName()))
			System.err.println("[ERROR][LOG2][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies base-2 logarithm (log₂x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge log2(String a, String out) {
		if (!CuBridgeJNI.log2(a, out))
			System.err.println("[ERROR][LOG2][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}
	
	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code ln(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ln() {
		if (!CuBridgeJNI.ln("", genRandomName()))
			System.err.println("[ERROR][LN][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code ln(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ln(String a) {
		if (!CuBridgeJNI.ln(a, genRandomName()))
			System.err.println("[ERROR][LN][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code ln(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ln(String a, String out) {
		if (!CuBridgeJNI.ln(a, out))
			System.err.println("[ERROR][LN][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the top tensor in the queue.
	 * <p>
	 * Full parameter: {@code reciprocal(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input tensor is retrieved from the top of the queue.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal() {
		if (!CuBridgeJNI.reciprocal("", genRandomName()))
			System.err.println("[ERROR][RECIPROCAL][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code reciprocal(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal(String a) {
		if (!CuBridgeJNI.reciprocal(a, genRandomName()))
			System.err.println("[ERROR][RECIPROCAL][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the specified tensor and stores the result under a new name.
	 * <ul>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 *
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal(String a, String out) {
		if (!CuBridgeJNI.reciprocal(a, out))
			System.err.println("[ERROR][RECIPROCAL][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	// 함수
	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the top tensor in the queue.
	 * <p>
	 * Full parameter: {@code sin(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input tensor is retrieved from the top of the queue.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sin() {
		if (!CuBridgeJNI.sin("", genRandomName()))
			System.err.println("[ERROR][SIN][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code sin(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sin(String a) {
		if (!CuBridgeJNI.sin(a, genRandomName()))
			System.err.println("[ERROR][SIN][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the specified tensor and stores the result under a new name.
	 * <ul>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sin(String a, String out) {
		if (!CuBridgeJNI.sin(a, out))
			System.err.println("[ERROR][SIN][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the top tensor in the queue.
	 * <p>
	 * Full parameter: {@code cos(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input tensor is retrieved from the top of the queue.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cos() {
		if (!CuBridgeJNI.cos("", genRandomName()))
			System.err.println("[ERROR][COS][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code cos(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cos(String a) {
		if (!CuBridgeJNI.cos(a, genRandomName()))
			System.err.println("[ERROR][COS][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the specified tensor and stores the result under a new name.
	 * <ul>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cos(String a, String out) {
		if (!CuBridgeJNI.cos(a, out))
			System.err.println("[ERROR][COS][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the top tensor in the queue.
	 * <p>
	 * Full parameter: {@code tan(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input tensor is retrieved from the top of the queue.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tan() {
		if (!CuBridgeJNI.tan("", genRandomName()))
			System.err.println("[ERROR][TAN][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code tan(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tan(String a) {
		if (!CuBridgeJNI.tan(a, genRandomName()))
			System.err.println("[ERROR][TAN][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the specified tensor and stores the result under a new name.
	 * <ul>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tan(String a, String out) {
		if (!CuBridgeJNI.tan(a, out))
			System.err.println("[ERROR][TAN][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies the step function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Returns 1 if the input is greater than 0; otherwise, returns 0.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge step() {
		if (!CuBridgeJNI.step("", genRandomName()))
			System.err.println("[ERROR][STEP][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies the step function to the specified tensor.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Returns 1 if the input is greater than 0; otherwise, returns 0.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge step(String a) {
		if (!CuBridgeJNI.step(a, genRandomName()))
			System.err.println("[ERROR][STEP][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies the step function to the specified tensor and stores the result under a new name.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Returns 1 if the input is greater than 0; otherwise, returns 0.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge step(String a, String out) {
		if (!CuBridgeJNI.step(a, out))
			System.err.println("[ERROR][STEP][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code sigmoid(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: 1 / (1 + exp(-x)).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sigmoid() {
		if (!CuBridgeJNI.sigmoid("", genRandomName()))
			System.err.println("[ERROR][SIGMOID][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the specified tensor.
	 * <p>
	 * Full parameter: {@code sigmoid(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: 1 / (1 + exp(-x)).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sigmoid(String a) {
		if (!CuBridgeJNI.sigmoid(a, genRandomName()))
			System.err.println("[ERROR][SIGMOID][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code sigmoid(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Uses the formula: 1 / (1 + exp(-x)).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sigmoid(String a, String out) {
		if (!CuBridgeJNI.sigmoid(a, out))
			System.err.println("[ERROR][SIGMOID][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies hyperbolic tangent (tanh) function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tanh() {
		if (!CuBridgeJNI.tanh("", genRandomName()))
			System.err.println("[ERROR][TANH][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tanh function to the specified tensor.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tanh(String a) {
		if (!CuBridgeJNI.tanh(a, genRandomName()))
			System.err.println("[ERROR][TANH][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tanh function to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Uses the formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function can be used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tanh(String a, String out) {
		if (!CuBridgeJNI.tanh(a, out))
			System.err.println("[ERROR][TANH][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU (Rectified Linear Unit) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: ReLU(x) = max(0, x).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function is widely used as an activation function in neural networks.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge relu() {
		if (!CuBridgeJNI.ReLu("", genRandomName()))
			System.err.println("[ERROR][RELU][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU (Rectified Linear Unit) to the specified tensor.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: ReLU(x) = max(0, x).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function is widely used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge relu(String a) {
		if (!CuBridgeJNI.ReLu(a, genRandomName()))
			System.err.println("[ERROR][RELU][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU (Rectified Linear Unit) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Uses the formula: ReLU(x) = max(0, x).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function is widely used as an activation function in neural networks.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge relu(String a, String out) {
		if (!CuBridgeJNI.ReLu(a, out))
			System.err.println("[ERROR][RELU][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code leakrelu(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: x if x > 0, else αx (α = 0.01).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function is used as an alternative activation to avoid dead neurons in ReLU.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge leakrelu() {
		if (!CuBridgeJNI.leakReLu("", genRandomName()))
			System.err.println("[ERROR][LEAKRELU][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the specified tensor.
	 * <p>
	 * Full parameter: {@code leakrelu(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: x if x > 0, else αx (α = 0.01).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function is used as an alternative activation to avoid dead neurons in ReLU.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge leakrelu(String a) {
		if (!CuBridgeJNI.leakReLu(a, genRandomName()))
			System.err.println("[ERROR][LEAKRELU][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code leakrelu(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Uses the formula: x if x > 0, else αx (α = 0.01).</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>This function is used as an alternative activation to avoid dead neurons in ReLU.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge leakrelu(String a, String out) {
		if (!CuBridgeJNI.leakReLu(a, out))
			System.err.println("[ERROR][LEAKRELU][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Softplus activation to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: log(1 + exp(x)).</li>
	 * <li>This is a smooth approximation of ReLU, used in neural networks for better gradient flow.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softplus() {
		if (!CuBridgeJNI.softplus("", genRandomName()))
			System.err.println("[ERROR][SOFTPLUS][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Softplus activation to the specified tensor.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: log(1 + exp(x)).</li>
	 * <li>This is a smooth approximation of ReLU, used in neural networks for better gradient flow.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softplus(String a) {
		if (!CuBridgeJNI.softplus(a, genRandomName()))
			System.err.println("[ERROR][SOFTPLUS][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Softplus activation to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Uses the formula: log(1 + exp(x)).</li>
	 * <li>This is a smooth approximation of ReLU, used in neural networks for better gradient flow.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softplus(String a, String out) {
		if (!CuBridgeJNI.softplus(a, out))
			System.err.println("[ERROR][SOFTPLUS][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: exp(x) = e<sup>x</sup> for each element.</li>
	 * <li>This operation is fundamental in many mathematical and statistical computations.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge exp() {
		if (!CuBridgeJNI.exp("", genRandomName()))
			System.err.println("[ERROR][EXP][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential function to the specified tensor.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Uses the formula: exp(x) = e<sup>x</sup> for each element.</li>
	 * <li>This operation is fundamental in many mathematical and statistical computations.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge exp(String a) {
		if (!CuBridgeJNI.exp(a, genRandomName()))
			System.err.println("[ERROR][EXP][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential function to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Uses the formula: exp(x) = e<sup>x</sup> for each element.</li>
	 * <li>This operation is fundamental in many mathematical and statistical computations.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge exp(String a, String out) {
		if (!CuBridgeJNI.exp(a, out))
			System.err.println("[ERROR][EXP][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}
	
	/**
	 * Unary Operation (Angle Conversion)
	 *
	 * Converts the top tensor in the queue from degrees to radians.
	 * <p>
	 * Full parameter: {@code deg2rad(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Each element is converted using the formula: rad = deg × π / 180.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge deg2rad() {
		if (!CuBridgeJNI.deg2rad("", genRandomName()))
			System.err.println("[ERROR][DEG2RAD][Cannot Execute][Tensor -, -]");

		return instance;
	}
	
	/**
	 * Unary Operation (Angle Conversion)
	 *
	 * Converts the specified tensor from degrees to radians.
	 * <p>
	 * Full parameter: {@code deg2rad(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Each element is converted using the formula: rad = deg × π / 180.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor in degrees
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge deg2rad(String a) {
		if (!CuBridgeJNI.deg2rad(a, genRandomName()))
			System.err.println("[ERROR][DEG2RAD][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}
	
	/**
	 * Unary Operation (Angle Conversion)
	 *
	 * Converts the specified tensor from degrees to radians and stores the result.
	 * <p>
	 * Full parameter: {@code deg2rad(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified output name.</li>
	 * <li>Each element is converted using the formula: rad = deg × π / 180.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor in degrees
	 * @param out the name for the output tensor in radians
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge deg2rad(String a, String out) {
		if (!CuBridgeJNI.deg2rad(a, out))
			System.err.println("[ERROR][DEG2RAD][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Angle Conversion)
	 *
	 * Converts the top tensor in the queue from radians to degrees.
	 * <p>
	 * Full parameter: {@code rad2deg(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Each element is converted using the formula: deg = rad × 180 / π.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge rad2deg() {
		if (!CuBridgeJNI.rad2deg("", genRandomName()))
			System.err.println("[ERROR][RAD2DEG][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Angle Conversion)
	 *
	 * Converts the specified tensor from radians to degrees.
	 * <p>
	 * Full parameter: {@code rad2deg(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Each element is converted using the formula: deg = rad × 180 / π.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor in radians
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge rad2deg(String a) {
		if (!CuBridgeJNI.rad2deg(a, genRandomName()))
			System.err.println("[ERROR][RAD2DEG][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}
	
	/**
	 * Unary Operation (Angle Conversion)
	 *
	 * Converts the specified tensor from radians to degrees and stores the result.
	 * <p>
	 * Full parameter: {@code rad2deg(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified output name.</li>
	 * <li>Each element is converted using the formula: deg = rad × 180 / π.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor in radians
	 * @param out the name for the output tensor in degrees
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge rad2deg(String a, String out) {
		if (!CuBridgeJNI.rad2deg(a, out))
			System.err.println("[ERROR][RAD2DEG][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code round(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Rounds each element to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge round() {
		if (!CuBridgeJNI.round("", genRandomName()))
			System.err.println("[ERROR][ROUND][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the specified tensor.
	 * <p>
	 * Full parameter: {@code round(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Rounds each element to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge round(String a) {
		if (!CuBridgeJNI.round(a, genRandomName()))
			System.err.println("[ERROR][ROUND][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code round(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Rounds each element to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge round(String a, String out) {
		if (!CuBridgeJNI.round(a, out))
			System.err.println("[ERROR][ROUND][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling operation (rounding up) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code ceil(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Rounds each element up to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ceil() {
		if (!CuBridgeJNI.ceil("", genRandomName()))
			System.err.println("[ERROR][CEIL][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling operation (rounding up) to the specified tensor.
	 * <p>
	 * Full parameter: {@code ceil(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Rounds each element up to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ceil(String a) {
		if (!CuBridgeJNI.ceil(a, genRandomName()))
			System.err.println("[ERROR][CEIL][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling operation (rounding up) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code ceil(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Rounds each element up to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out  the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ceil(String a, String out) {
		if (!CuBridgeJNI.ceil(a, out))
			System.err.println("[ERROR][CEIL][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor operation (rounding down) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code floor(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Rounds each element down to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge floor() {
		if (!CuBridgeJNI.floor("", genRandomName()))
			System.err.println("[ERROR][FLOOR][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor operation (rounding down) to the specified tensor.
	 * <p>
	 * Full parameter: {@code floor(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Rounds each element down to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge floor(String a) {
		if (!CuBridgeJNI.floor(a, genRandomName()))
			System.err.println("[ERROR][FLOOR][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor operation (rounding down) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code floor(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Rounds each element down to the nearest integer.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out  the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge floor(String a, String out) {
		if (!CuBridgeJNI.floor(a, out))
			System.err.println("[ERROR][FLOOR][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code not(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Each element is treated as boolean: 0 as false, non-zero as true.</li>
	 * <li>Returns 1 if the element is 0; otherwise, returns 0.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge not() {
		if (!CuBridgeJNI.not("", genRandomName()))
			System.err.println("[ERROR][NOT][Cannot Execute][Tensor -, -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the specified tensor.
	 * <p>
	 * Full parameter: {@code not(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Each element is treated as boolean: 0 as false, non-zero as true.</li>
	 * <li>Returns 1 if the element is 0; otherwise, returns 0.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge not(String a) {
		if (!CuBridgeJNI.not(a, genRandomName()))
			System.err.println("[ERROR][NOT][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code not(String name, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The result is stored under the specified name.</li>
	 * <li>Each element is treated as boolean: 0 as false, non-zero as true.</li>
	 * <li>Returns 1 if the element is 0; otherwise, returns 0.</li>
	 * <li>Operation is applied element-wise.</li>
	 * <li>If the input tensor is missing, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @param out  the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge not(String a, String out) {
		if (!CuBridgeJNI.not(a, out))
			System.err.println("[ERROR][NOT][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	// 이항
	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code add(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add() {
		if (!CuBridgeJNI.add("", "", genRandomName()))
			System.err.println("[ERROR][ADD][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 *
	 * <p>
	 * Full parameter: {@code add(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add(String a) {
		if (!CuBridgeJNI.add(a, "", genRandomName()))
			System.err.println("[ERROR][ADD][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b) using the two specified input tensors.
	 *
	 * <p>
	 * Full parameter: {@code add(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add(String a, String b) {
		if (!CuBridgeJNI.add(a, b, genRandomName()))
			System.err.println("[ERROR][ADD][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b) using the two specified input tensors and stores the result under the specified name.
	 *
	 * <p>
	 * Full parameter: {@code add(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge add(String a, String b, String out) {
		if (!CuBridgeJNI.add(a, b, out))
			System.err.println("[ERROR][ADD][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subtraction (a - b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code sub(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub() {
		if (!CuBridgeJNI.sub("", "", genRandomName()))
			System.err.println("[ERROR][SUB][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subtraction (a - b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 *
	 * <p>
	 * Full parameter: {@code sub(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub(String a) {
		if (!CuBridgeJNI.sub(a, "", genRandomName()))
			System.err.println("[ERROR][SUB][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subtraction (a - b) using the two specified input tensors.
	 *
	 * <p>
	 * Full parameter: {@code sub(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub(String a, String b) {
		if (!CuBridgeJNI.sub(a, b, genRandomName()))
			System.err.println("[ERROR][SUB][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subtraction (a - b) using the two specified input tensors and stores the result under the specified name.
	 *
	 * <p>
	 * Full parameter: {@code sub(String a, String b, String out)}<br>
	 * This version:
	 * <ul> 
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sub(String a, String b, String out) {
		if (!CuBridgeJNI.sub(a, b, out))
			System.err.println("[ERROR][SUB][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise multiplication (a × b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code mul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul() {
		if (!CuBridgeJNI.mul("", "", genRandomName()))
			System.err.println("[ERROR][MUL][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise multiplication (a × b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 *
	 * <p>
	 * Full parameter: {@code mul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul(String a) {
		if (!CuBridgeJNI.mul(a, "", genRandomName()))
			System.err.println("[ERROR][MUL][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise multiplication (a × b) using the two specified input tensors.
	 *
	 * <p>
	 * Full parameter: {@code mul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul(String a, String b) {
		if (!CuBridgeJNI.mul(a, b, genRandomName()))
			System.err.println("[ERROR][MUL][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise multiplication (a × b) using the two specified input tensors and stores the result under the specified name.
	 *
	 * <p>
	 * Full parameter: {@code mul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mul(String a, String b, String out) {
		if (!CuBridgeJNI.mul(a, b, out))
			System.err.println("[ERROR][MUL][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise division (a ÷ b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code div(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div() {
		if (!CuBridgeJNI.div("", "", genRandomName()))
			System.err.println("[ERROR][DIV][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise division (a ÷ b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 *
	 * <p>
	 * Full parameter: {@code div(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div(String a) {
		if (!CuBridgeJNI.div(a, "", genRandomName()))
			System.err.println("[ERROR][DIV][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise division (a ÷ b) using the two specified input tensors.
	 *
	 * <p>
	 * Full parameter: {@code div(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div(String a, String b) {
		if (!CuBridgeJNI.div(a, b, genRandomName()))
			System.err.println("[ERROR][DIV][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise division (a ÷ b) using the two specified input tensors and stores the result under the specified name.
	 *
	 * <p>
	 * Full parameter: {@code div(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge div(String a, String b, String out) {
		if (!CuBridgeJNI.div(a, b, out))
			System.err.println("[ERROR][DIV][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise exponentiation (a ^ b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code pow(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow() {
		if (!CuBridgeJNI.pow("", "", genRandomName()))
			System.err.println("[ERROR][POW][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise exponentiation (a ^ b), using the specified tensor as the base and the top tensor from the queue as the exponent.
	 *
	 * <p>
	 * Full parameter: {@code pow(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The base tensor is specified by name, and the exponent is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified base tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the base tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow(String a) {
		if (!CuBridgeJNI.pow(a, "", genRandomName()))
			System.err.println("[ERROR][POW][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise exponentiation (a ^ b) using the two specified input tensors.
	 *
	 * <p>
	 * Full parameter: {@code pow(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the base tensor
	 * @param b the name of the exponent tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow(String a, String b) {
		if (!CuBridgeJNI.pow(a, b, genRandomName()))
			System.err.println("[ERROR][POW][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise exponentiation (a ^ b) using the two specified input tensors and stores the result under the specified name.
	 *
	 * <p>
	 * Full parameter: {@code pow(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the base tensor
	 * @param b   the name of the exponent tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge pow(String a, String b, String out) {
		if (!CuBridgeJNI.pow(a, b, out))
			System.err.println("[ERROR][POW][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modulus (a % b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code mod(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod() {
		if (!CuBridgeJNI.mod("", "", genRandomName()))
			System.err.println("[ERROR][MOD][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modulus (a % b), using the specified tensor as the dividend and the top tensor from the queue as the divisor.
	 *
	 * <p>
	 * Full parameter: {@code mod(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The dividend tensor is specified by name, and the divisor is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified dividend tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the dividend tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod(String a) {
		if (!CuBridgeJNI.mod(a, "", genRandomName()))
			System.err.println("[ERROR][MOD][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modulus (a % b) using the two specified input tensors.
	 *
	 * <p>
	 * Full parameter: {@code mod(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the dividend tensor
	 * @param b the name of the divisor tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod(String a, String b) {
		if (!CuBridgeJNI.mod(a, b, genRandomName()))
			System.err.println("[ERROR][MOD][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modulus (a % b) using the two specified input tensors and stores the result under the specified name.
	 *
	 * <p>
	 * Full parameter: {@code mod(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the dividend tensor
	 * @param b   the name of the divisor tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mod(String a, String b, String out) {
		if (!CuBridgeJNI.mod(a, b, out))
			System.err.println("[ERROR][MOD][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a > b) between the top two tensors in the queue.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code gt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt() {
		if (!CuBridgeJNI.gt("", "", genRandomName()))
			System.err.println("[ERROR][GT][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a > b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code gt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt(String a) {
		if (!CuBridgeJNI.gt(a, "", genRandomName()))
			System.err.println("[ERROR][GT][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a > b) using the two specified input tensors.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code gt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt(String a, String b) {
		if (!CuBridgeJNI.gt(a, b, genRandomName()))
			System.err.println("[ERROR][GT][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a > b) using the two specified input tensors and stores the result under the specified name.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code gt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge gt(String a, String b, String out) {
		if (!CuBridgeJNI.gt(a, b, out))
			System.err.println("[ERROR][GT][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a < b) between the top two tensors in the queue.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code lt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge lt() {
		if (!CuBridgeJNI.lt("", "", genRandomName()))
			System.err.println("[ERROR][LT][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a < b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code lt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge lt(String a) {
		if (!CuBridgeJNI.lt(a, "", genRandomName()))
			System.err.println("[ERROR][LT][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a < b) using the two specified input tensors.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code lt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge lt(String a, String b) {
		if (!CuBridgeJNI.lt(a, b, genRandomName()))
			System.err.println("[ERROR][LT][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a < b) using the two specified input tensors and stores the result under the specified name.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code lt(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge lt(String a, String b, String out) {
		if (!CuBridgeJNI.lt(a, b, out))
			System.err.println("[ERROR][LT][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≥ b) between the top two tensors in the queue.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ge(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ge() {
		if (!CuBridgeJNI.ge("", "", genRandomName()))
			System.err.println("[ERROR][GE][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≥ b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ge(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ge(String a) {
		if (!CuBridgeJNI.ge(a, "", genRandomName()))
			System.err.println("[ERROR][GE][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≥ b) using the two specified input tensors.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ge(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ge(String a, String b) {
		if (!CuBridgeJNI.ge(a, b, genRandomName()))
			System.err.println("[ERROR][GE][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≥ b) using the two specified input tensors and stores the result under the specified name.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ge(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ge(String a, String b, String out) {
		if (!CuBridgeJNI.ge(a, b, out))
			System.err.println("[ERROR][GE][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≤ b) between the top two tensors in the queue.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code le(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge le() {
		if (!CuBridgeJNI.le("", "", genRandomName()))
			System.err.println("[ERROR][LE][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≤ b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code le(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge le(String a) {
		if (!CuBridgeJNI.le(a, "", genRandomName()))
			System.err.println("[ERROR][LE][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≤ b) using the two specified input tensors.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code le(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge le(String a, String b) {
		if (!CuBridgeJNI.le(a, b, genRandomName()))
			System.err.println("[ERROR][LE][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise comparison (a ≤ b) using the two specified input tensors and stores the result under the specified name.
	 * Returns 1 where the condition is true, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code le(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge le(String a, String b, String out) {
		if (!CuBridgeJNI.le(a, b, out))
			System.err.println("[ERROR][LE][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise equality comparison (a == b) between the top two tensors in the queue.
	 * Returns 1 where the values are equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code eq(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge eq() {
		if (!CuBridgeJNI.eq("", "", genRandomName()))
			System.err.println("[ERROR][EQ][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise equality comparison (a == b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Returns 1 where the values are equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code eq(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge eq(String a) {
		if (!CuBridgeJNI.eq(a, "", genRandomName()))
			System.err.println("[ERROR][EQ][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise equality comparison (a == b) using the two specified input tensors.
	 * Returns 1 where the values are equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code eq(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge eq(String a, String b) {
		if (!CuBridgeJNI.eq(a, b, genRandomName()))
			System.err.println("[ERROR][EQ][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise equality comparison (a == b) using the two specified input tensors and stores the result under the specified name.
	 * Returns 1 where the values are equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code eq(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge eq(String a, String b, String out) {
		if (!CuBridgeJNI.eq(a, b, out))
			System.err.println("[ERROR][EQ][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise not-equal comparison (a != b) between the top two tensors in the queue.
	 * Returns 1 where the values are not equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ne(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ne() {
		if (!CuBridgeJNI.ne("", "", genRandomName()))
			System.err.println("[ERROR][NE][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise not-equal comparison (a != b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Returns 1 where the values are not equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ne(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ne(String a) {
		if (!CuBridgeJNI.ne(a, "", genRandomName()))
			System.err.println("[ERROR][NE][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise not-equal comparison (a != b) using the two specified input tensors.
	 * Returns 1 where the values are not equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ne(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ne(String a, String b) {
		if (!CuBridgeJNI.ne(a, b, genRandomName()))
			System.err.println("[ERROR][NE][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise not-equal comparison (a != b) using the two specified input tensors and stores the result under the specified name.
	 * Returns 1 where the values are not equal, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code ne(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ne(String a, String b, String out) {
		if (!CuBridgeJNI.ne(a, b, out))
			System.err.println("[ERROR][NE][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical AND (a && b) between the top two tensors in the queue.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where both elements are non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code and(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge and() {
		if (!CuBridgeJNI.and("", "", genRandomName()))
			System.err.println("[ERROR][AND][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical AND (a && b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where both elements are non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code and(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge and(String a) {
		if (!CuBridgeJNI.and(a, "", genRandomName()))
			System.err.println("[ERROR][AND][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical AND (a && b) using the two specified input tensors.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where both elements are non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code and(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge and(String a, String b) {
		if (!CuBridgeJNI.and(a, b, genRandomName()))
			System.err.println("[ERROR][AND][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical AND (a && b) using the two specified input tensors and stores the result under the specified name.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where both elements are non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code and(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge and(String a, String b, String out) {
		if (!CuBridgeJNI.and(a, b, out))
			System.err.println("[ERROR][AND][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical OR (a || b) between the top two tensors in the queue.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where at least one element is non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code or(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The top two tensors in the queue are used as inputs.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge or() {
		if (!CuBridgeJNI.or("", "", genRandomName()))
			System.err.println("[ERROR][OR][Cannot Execute][Tensor -, -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical OR (a || b), using the specified tensor as the first operand and the top tensor from the queue as the second.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where at least one element is non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code or(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first tensor is specified by name, and the second is taken from the top of the queue.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If the specified tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge or(String a) {
		if (!CuBridgeJNI.or(a, "", genRandomName()))
			System.err.println("[ERROR][OR][Cannot Execute][Tensor " + a + ", -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical OR (a || b) using the two specified input tensors.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where at least one element is non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code or(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensors are specified by name.</li>
	 *   <li>The result tensor is stored with an auto-generated name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge or(String a, String b) {
		if (!CuBridgeJNI.or(a, b, genRandomName()))
			System.err.println("[ERROR][OR][Cannot Execute][Tensor " + a + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs element-wise logical OR (a || b) using the two specified input tensors and stores the result under the specified name.
	 * Each element is interpreted as boolean: non-zero as true, zero as false.
	 * Returns 1 where at least one element is non-zero, 0 otherwise.
	 *
	 * <p>
	 * Full parameter: {@code or(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensor names are all specified.</li>
	 *   <li>The result tensor is stored with the specified name.</li>
	 *   <li>If either input tensor is missing, an error message is printed and the operation fails.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the first input tensor
	 * @param b   the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge or(String a, String b, String out) {
		if (!CuBridgeJNI.or(a, b, out))
			System.err.println("[ERROR][OR][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing all elements across all axes.
	 *
	 * <p>
	 * Full parameter: {@code sum(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum() {
		if (!CuBridgeJNI.sum("", genRandomName(), -1))
			System.err.println("[ERROR][SUM][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing elements along the specified axis and all subsequent axes.
	 *
	 * <p>
	 * Full parameter: {@code sum(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the starting axis for reduction; -1 reduces across all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(int axis) {
		if (!CuBridgeJNI.sum("", genRandomName(), axis))
			System.err.println("[ERROR][SUM][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing all elements across all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code sum(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(String a) {
		if (!CuBridgeJNI.sum(a, genRandomName(), -1))
			System.err.println("[ERROR][SUM][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing elements along the specified axis and all subsequent axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code sum(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the starting axis for reduction; -1 reduces across all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(String a, int axis) {
		if (!CuBridgeJNI.sum(a, genRandomName(), axis))
			System.err.println("[ERROR][SUM][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing all elements across all axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code sum(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name to assign to the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(String a, String out) {
		if (!CuBridgeJNI.sum(a, out, -1))
			System.err.println("[ERROR][SUM][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing elements along the specified axis and all subsequent axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code sum(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name to assign to the output tensor
	 * @param axis the starting axis for reduction; -1 reduces across all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(String a, String out, int axis) {
		if (!CuBridgeJNI.sum(a, out, axis))
			System.err.println("[ERROR][SUM][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the mean value over all axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code mean(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs mean reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean() {
		if (!CuBridgeJNI.mean("", genRandomName(), -1))
			System.err.println("[ERROR][MEAN][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the mean starting from the specified axis down to the last axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code mean(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs mean reduction from the specified axis down through all subsequent axes.</li>
	 *   <li>If axis = -1, computes the mean over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the starting axis for mean reduction; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(int axis) {
		if (!CuBridgeJNI.mean("", genRandomName(), axis))
			System.err.println("[ERROR][MEAN][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the mean value over all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code mean(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor is specified by name.</li>
	 *   <li>Performs mean reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(String a) {
		if (!CuBridgeJNI.mean(a, genRandomName(), -1))
			System.err.println("[ERROR][MEAN][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the mean starting from the specified axis down to the last axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code mean(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor is specified by name.</li>
	 *   <li>Performs mean reduction from the specified axis down through all subsequent axes.</li>
	 *   <li>If axis = -1, computes the mean over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the starting axis for mean reduction; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(String a, int axis) {
		if (!CuBridgeJNI.mean(a, genRandomName(), axis))
			System.err.println("[ERROR][MEAN][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the mean value over all axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code mean(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensors are specified by name.</li>
	 *   <li>Performs mean reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(String a, String out) {
		if (!CuBridgeJNI.mean(a, out, -1))
			System.err.println("[ERROR][MEAN][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the mean starting from the specified axis down to the last axis of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code mean(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input and output tensors are specified by name.</li>
	 *   <li>Performs mean reduction from the specified axis down through all subsequent axes.</li>
	 *   <li>If axis = -1, computes the mean over all axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the starting axis for mean reduction; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(String a, String out, int axis) {
		if (!CuBridgeJNI.mean(a, out, axis))
			System.err.println("[ERROR][MEAN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes variance over all axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code var(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes variance across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var() {
		if (!CuBridgeJNI.var("", genRandomName(), -1))
			System.err.println("[ERROR][VAR][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes variance starting from the specified axis down to the last axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code var(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes variance from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes variance over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the starting axis for variance computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(int axis) {
		if (!CuBridgeJNI.var("", genRandomName(), axis))
			System.err.println("[ERROR][VAR][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes variance over all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code var(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes variance across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(String a) {
		if (!CuBridgeJNI.var(a, genRandomName(), -1))
			System.err.println("[ERROR][VAR][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes variance starting from the specified axis down to the last axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code var(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes variance from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes variance over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the starting axis for variance computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(String a, int axis) {
		if (!CuBridgeJNI.var(a, genRandomName(), axis))
			System.err.println("[ERROR][VAR][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes variance over all axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code var(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes variance across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(String a, String out) {
		if (!CuBridgeJNI.var(a, out, -1))
			System.err.println("[ERROR][VAR][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes variance starting from the specified axis down to the last axis of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code var(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes variance from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes variance over all axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the starting axis for variance computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(String a, String out, int axis) {
		if (!CuBridgeJNI.var(a, out, axis))
			System.err.println("[ERROR][VAR][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes standard deviation over all axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code std(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes standard deviation across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std() {
		if (!CuBridgeJNI.std("", genRandomName(), -1))
			System.err.println("[ERROR][STD][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes standard deviation starting from the specified axis down to the last axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code std(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes standard deviation from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes standard deviation over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the starting axis for std computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(int axis) {
		if (!CuBridgeJNI.std("", genRandomName(), axis))
			System.err.println("[ERROR][STD][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes standard deviation over all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code std(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes standard deviation across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(String a) {
		if (!CuBridgeJNI.std(a, genRandomName(), -1))
			System.err.println("[ERROR][STD][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes standard deviation starting from the specified axis down to the last axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code std(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes standard deviation from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes standard deviation over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the starting axis for std computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(String a, int axis) {
		if (!CuBridgeJNI.std(a, genRandomName(), axis))
			System.err.println("[ERROR][STD][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes standard deviation over all axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code std(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes standard deviation across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(String a, String out) {
		if (!CuBridgeJNI.std(a, out, -1))
			System.err.println("[ERROR][STD][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes standard deviation starting from the specified axis down to the last axis of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code std(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes standard deviation from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes standard deviation over all axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the starting axis for std computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(String a, String out, int axis) {
		if (!CuBridgeJNI.std(a, out, axis))
			System.err.println("[ERROR][STD][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the maximum value over all axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code max(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes maximum across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max() {
		if (!CuBridgeJNI.max("", genRandomName(), -1))
			System.err.println("[ERROR][MAX][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the maximum value starting from the specified axis down to the last axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code max(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes maximum from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes maximum over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the starting axis for maximum computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(int axis) {
		if (!CuBridgeJNI.max("", genRandomName(), axis))
			System.err.println("[ERROR][MAX][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the maximum value over all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code max(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes maximum across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(String a) {
		if (!CuBridgeJNI.max(a, genRandomName(), -1))
			System.err.println("[ERROR][MAX][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the maximum value starting from the specified axis down to the last axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code max(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes maximum from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes maximum over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the starting axis for maximum computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(String a, int axis) {
		if (!CuBridgeJNI.max(a, genRandomName(), axis))
			System.err.println("[ERROR][MAX][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the maximum value over all axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code max(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes maximum across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(String a, String out) {
		if (!CuBridgeJNI.max(a, out, -1))
			System.err.println("[ERROR][MAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the maximum value starting from the specified axis down to the last axis of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code max(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes maximum from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes maximum over all axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the starting axis for maximum computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(String a, String out, int axis) {
		if (!CuBridgeJNI.max(a, out, axis))
			System.err.println("[ERROR][MAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the minimum value over all axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code min(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes minimum across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min() {
		if (!CuBridgeJNI.min("", genRandomName(), -1))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the minimum value starting from the specified axis down to the last axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code min(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Computes minimum from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes minimum over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the starting axis for minimum computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(int axis) {
		if (!CuBridgeJNI.min("", genRandomName(), axis))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the minimum value over all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code min(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes minimum across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(String a) {
		if (!CuBridgeJNI.min(a, genRandomName(), -1))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the minimum value starting from the specified axis down to the last axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter: {@code min(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Computes minimum from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes minimum over all axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the starting axis for minimum computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(String a, int axis) {
		if (!CuBridgeJNI.min(a, genRandomName(), axis))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the minimum value over all axes of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code min(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes minimum across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(String a, String out) {
		if (!CuBridgeJNI.min(a, out, -1))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Computes the minimum value starting from the specified axis down to the last axis of the specified tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code min(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Computes minimum from the specified axis through all subsequent axes.</li>
	 *   <li>If axis = -1, computes minimum over all axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the starting axis for minimum computation; -1 means all axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(String a, String out, int axis) {
		if (!CuBridgeJNI.min(a, out, axis))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code accumulate(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to performing accumulation along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to perform accumulation; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate() {
		if (!CuBridgeJNI.accumulate("", genRandomName(), -1))
			System.err.println("[ERROR][ACCUMULATE][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code accumulate(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to performing accumulation along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to perform accumulation; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(int axis) {
		if (!CuBridgeJNI.accumulate("", genRandomName(), axis))
			System.err.println("[ERROR][ACCUMULATE][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code accumulate(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to performing accumulation along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to perform accumulation; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(String a) {
		if (!CuBridgeJNI.accumulate(a, genRandomName(), -1))
			System.err.println("[ERROR][ACCUMULATE][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code accumulate(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to performing accumulation along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to perform accumulation; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(String a, int axis) {
		if (!CuBridgeJNI.accumulate(a, genRandomName(), axis))
			System.err.println("[ERROR][ACCUMULATE][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code accumulate(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to performing accumulation along the first axis (index 0).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to perform accumulation; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(String a, String out) {
		if (!CuBridgeJNI.accumulate(a, out, -1))
			System.err.println("[ERROR][ACCUMULATE][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code accumulate(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to performing accumulation along the first axis (index 0).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to perform accumulation; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(String a, String out, int axis) {
		if (!CuBridgeJNI.accumulate(a, out, axis))
			System.err.println("[ERROR][ACCUMULATE][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Computes the average (mean) along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code compress(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to averaging along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to compute the average; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress() {
		if (!CuBridgeJNI.compress("", genRandomName(), -1))
			System.err.println("[ERROR][COMPRESS][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Computes the average (mean) along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code compress(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to averaging along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to compute the average; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(int axis) {
		if (!CuBridgeJNI.compress("", genRandomName(), axis))
			System.err.println("[ERROR][COMPRESS][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Computes the average (mean) along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code compress(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to averaging along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to compute the average; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(String a) {
		if (!CuBridgeJNI.compress(a, genRandomName(), -1))
			System.err.println("[ERROR][COMPRESS][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Computes the average (mean) along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code compress(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to averaging along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to compute the average; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(String a, int axis) {
		if (!CuBridgeJNI.compress(a, genRandomName(), axis))
			System.err.println("[ERROR][COMPRESS][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Computes the average (mean) along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code compress(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to averaging along the first axis (index 0).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to compute the average; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(String a, String out) {
		if (!CuBridgeJNI.compress(a, out, -1))
			System.err.println("[ERROR][COMPRESS][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Computes the average (mean) along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code compress(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to averaging along the first axis (index 0).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to compute the average; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(String a, String out, int axis) {
		if (!CuBridgeJNI.compress(a, out, axis))
			System.err.println("[ERROR][COMPRESS][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Expands the specified axis by copying it N times.
	 *
	 * <p>
	 * Full parameter: {@code expand(String a, String out, int axis, int N)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>Requires that N is divisible by the size of the specified axis; otherwise, the operation is invalid.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis (must be divisible by original size)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge expand(int axis, int N) {
		if (!CuBridgeJNI.expand("", genRandomName(), axis, N))
			System.err.println("[ERROR][EXPAND][Cannot Execute][Tensor -, -, axis=" + axis + ", N=" + N + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Expands the specified axis by copying it N times.
	 *
	 * <p>
	 * Full parameter: {@code expand(String a, String out, int axis, int N)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>Requires that N is divisible by the size of the specified axis; otherwise, the operation is invalid.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis (must be divisible by original size)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge expand(String a, int axis, int N) {
		if (!CuBridgeJNI.expand(a, genRandomName(), axis, N))
			System.err.println("[ERROR][EXPAND][Cannot Execute][Tensor " + a + ", -, axis=" + axis + ", N=" + N + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Expands the specified axis by copying it N times.
	 *
	 * <p>
	 * Full parameter: {@code expand(String a, String out, int axis, int N)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>Requires that N is divisible by the size of the specified axis; otherwise, the operation is invalid.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis (must be divisible by original size)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge expand(String a, String out, int axis, int N) {
		if (!CuBridgeJNI.expand(a, out, axis, N))
			System.err.println("[ERROR][EXPAND][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + ", N=" + N + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code argMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the max indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMax() {
		if (!CuBridgeJNI.argMax("", genRandomName(), -1))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code argMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the max indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMax(int axis) {
		if (!CuBridgeJNI.argMax("", genRandomName(), axis))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code argMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the max indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMax(String a) {
		if (!CuBridgeJNI.argMax(a, genRandomName(), -1))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code argMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the max indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMax(String a, int axis) {
		if (!CuBridgeJNI.argMax(a, genRandomName(), axis))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code argMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the maximum values along the specified axis.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to find the max indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMax(String a, String out, int axis) {
		if (!CuBridgeJNI.argMax(a, out, axis))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code argMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the min indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMin() {
		if (!CuBridgeJNI.argMin("", genRandomName(), -1))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code argMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the min indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMin(int axis) {
		if (!CuBridgeJNI.argMin("", genRandomName(), axis))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code argMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the min indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMin(String a) {
		if (!CuBridgeJNI.argMin(a, genRandomName(), -1))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code argMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the min indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMin(String a, int axis) {
		if (!CuBridgeJNI.argMin(a, genRandomName(), axis))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code argMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the indices of the minimum values along the specified axis.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to find the min indices; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argMin(String a, String out, int axis) {
		if (!CuBridgeJNI.argMin(a, out, axis))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum values along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code axisMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the maximum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax() {
		if (!CuBridgeJNI.axisMax("", genRandomName(), -1))
			System.err.println("[ERROR][AXISMAX][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum values along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code axisMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the maximum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(int axis) {
		if (!CuBridgeJNI.axisMax("", genRandomName(), axis))
			System.err.println("[ERROR][AXISMAX][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum values along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code axisMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the maximum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(String a) {
		if (!CuBridgeJNI.axisMax(a, genRandomName(), -1))
			System.err.println("[ERROR][AXISMAX][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum values along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code axisMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the maximum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the maximum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(String a, int axis) {
		if (!CuBridgeJNI.axisMax(a, genRandomName(), axis))
			System.err.println("[ERROR][AXISMAX][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum values along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code axisMax(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the maximum values along the specified axis.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to find the maximum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(String a, String out, int axis) {
		if (!CuBridgeJNI.axisMax(a, out, axis))
			System.err.println("[ERROR][AXISMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum values along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code axisMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the minimum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin() {
		if (!CuBridgeJNI.axisMin("", genRandomName(), -1))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum values along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code axisMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to find the minimum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(int axis) {
		if (!CuBridgeJNI.axisMin("", genRandomName(), axis))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum values along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code axisMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the minimum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(String a) {
		if (!CuBridgeJNI.axisMin(a, genRandomName(), -1))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum values along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code axisMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the minimum values along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to find the minimum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(String a, int axis) {
		if (!CuBridgeJNI.axisMin(a, genRandomName(), axis))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum values along the specified axis of the given tensor and stores the result in the output tensor.
	 *
	 * <p>
	 * Full parameter: {@code axisMin(String a, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, defaults to searching along the first axis (index 0).</li>
	 *   <li>Returns the minimum values along the specified axis.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis along which to find the minimum values; -1 means use the first axis (index 0) by default
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(String a, String out, int axis) {
		if (!CuBridgeJNI.axisMin(a, out, axis))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the last two axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * This version:
	 * <ul>
	 *   <li>Operates on the top tensor in the queue.</li>
	 *   <li>Swaps the last two axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose() {
		if (!CuBridgeJNI.transpose("", genRandomName(), 0, -1))
			System.err.println("[ERROR][TRANSPOSE][Cannot Execute][Tensor -, -, axis0=0, axis1=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the last two axes of the specified input tensor.
	 *
	 * <p>
	 * Full parameter: {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Swaps the last two axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(String a) {
		if (!CuBridgeJNI.transpose(a, genRandomName(), 0, -1))
			System.err.println("[ERROR][TRANSPOSE][Cannot Execute][Tensor " + a + ", -, axis0=0, axis1=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the last two axes of the specified input tensor and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensors are specified by name.</li>
	 *   <li>Swaps the last two axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(String a, String out) {
		if (!CuBridgeJNI.transpose(a, out, 0, -1))
			System.err.println("[ERROR][TRANSPOSE][Cannot Execute][Tensor " + a + ", " + out + ", axis0=0, axis1=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the two specified axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * This version:
	 * <ul>
	 *   <li>Operates on the top tensor in the queue.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis1 the first axis to swap; -1 means swap the last two axes
	 * @param axis2 the second axis to swap; -1 means swap the last two axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(int axis1, int axis2) {
		if (!CuBridgeJNI.transpose("", genRandomName(), axis1, axis2))
			System.err.println("[ERROR][TRANSPOSE][Cannot Execute][Tensor -, -, axis0=" + axis1 + ", axis1=" + axis2 + "]");
		return instance;
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the two specified axes of the given tensor.
	 *
	 * <p>
	 * Full parameter: {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis1 the first axis to swap; -1 means swap the last two axes
	 * @param axis2 the second axis to swap; -1 means swap the last two axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(String a, int axis1, int axis2) {
		if (!CuBridgeJNI.transpose(a, genRandomName(), axis1, axis2))
			System.err.println("[ERROR][TRANSPOSE][Cannot Execute][Tensor " + a + ", -, axis0=" + axis1 + ", axis1=" + axis2 + "]");
		return instance;
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the two specified axes of the given tensor and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensors are specified by name.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a    the name of the input tensor
	 * @param out  the name of the output tensor
	 * @param axis1 the first axis to swap; -1 means swap the last two axes
	 * @param axis2 the second axis to swap; -1 means swap the last two axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(String a, String out, int axis1, int axis2) {
		if (!CuBridgeJNI.transpose(a, out, axis1, axis2))
			System.err.println("[ERROR][TRANSPOSE][Cannot Execute][Tensor " + a + ", " + out + ", axis0=" + axis1 + ", axis1=" + axis2 + "]");
		return instance;
	}

	/**
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product (a ⋅ b) of the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code dot(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top two tensors in the queue as inputs.</li>
	 *   <li>Only supports 1D or 2D tensors. If both inputs are higher than 2D, automatically calls {@code matmul()}.</li>
	 *   <li>If either input is 1D and the other is 2D, automatically reshapes the 1D tensor to perform matrix multiplication.</li>
	 *   <li>The last axis of {@code a} must match the first axis of {@code b}.</li>
	 *   <li>All other (leading) axes must match.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot() {
		if (!CuBridgeJNI.dot("", "", genRandomName()))
			System.err.println("[ERROR][DOT][Cannot Execute][Tensor -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product (a ⋅ b) between the specified tensor and the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code dot(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first input tensor is specified by name; the second is taken from the queue.</li>
	 *   <li>Only supports 1D or 2D tensors. If both inputs are higher than 2D, automatically calls {@code matmul()}.</li>
	 *   <li>If either input is 1D and the other is 2D, automatically reshapes the 1D tensor to perform matrix multiplication.</li>
	 *   <li>The last axis of {@code a} must match the first axis of {@code b}.</li>
	 *   <li>All other (leading) axes must match.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(String a) {
		if (!CuBridgeJNI.dot(a, "", genRandomName()))
			System.err.println("[ERROR][DOT][Cannot Execute][Tensor " + a + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product (a ⋅ b) between two specified tensors.
	 *
	 * <p>
	 * Full parameter: {@code dot(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both input tensors are specified by name.</li>
	 *   <li>Only supports 1D or 2D tensors. If both inputs are higher than 2D, automatically calls {@code matmul()}.</li>
	 *   <li>If either input is 1D and the other is 2D, automatically reshapes the 1D tensor to perform matrix multiplication.</li>
	 *   <li>The last axis of {@code a} must match the first axis of {@code b}.</li>
	 *   <li>All other (leading) axes must match.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(String a, String b) {
		if (!CuBridgeJNI.dot(a, b, genRandomName()))
			System.err.println("[ERROR][DOT][Cannot Execute][Tensor " + a + ", " + b + "]");
		return instance;
	}

	/**
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product (a ⋅ b) between two specified tensors and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code dot(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both input and output tensors are specified by name.</li>
	 *   <li>Only supports 1D or 2D tensors. If both inputs are higher than 2D, automatically calls {@code matmul()}.</li>
	 *   <li>If either input is 1D and the other is 2D, automatically reshapes the 1D tensor to perform matrix multiplication.</li>
	 *   <li>The last axis of {@code a} must match the first axis of {@code b}.</li>
	 *   <li>All other (leading) axes must match.</li>
	 *   <li>The result is stored under the specified name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(String a, String b, String out) {
		if (!CuBridgeJNI.dot(a, b, out))
			System.err.println("[ERROR][DOT][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs matrix multiplication between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code matmul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top two tensors in the queue as inputs.</li>
	 *   <li>Both tensors must be at least 3D. If both tensors are 1D or 2D, calls {@code dot()} instead.</li>
	 *   <li>Broadcasts all leading axes except the last two.</li>
	 *   <li>{@code a.shape[-1]} must match {@code b.shape[-2]}.</li>
	 *   <li>Result shape: broadcasted leading dims + [a.shape[-2], b.shape[-1]].</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge matmul() {
		if (!CuBridgeJNI.matmul("", "", genRandomName()))
			System.err.println("[ERROR][MATMUL][Cannot Execute][Tensor -, -]");
		return instance;
	}

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs matrix multiplication using the specified tensor as the first input.
	 *
	 * <p>
	 * Full parameter: {@code matmul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The first input tensor is specified by name; the second is taken from the queue.</li>
	 *   <li>Both tensors must be at least 3D. If both tensors are 1D or 2D, calls {@code dot()} instead.</li>
	 *   <li>Broadcasts all leading axes except the last two.</li>
	 *   <li>{@code a.shape[-1]} must match {@code b.shape[-2]}.</li>
	 *   <li>Result shape: broadcasted leading dims + [a.shape[-2], b.shape[-1]].</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge matmul(String a) {
		if (!CuBridgeJNI.matmul(a, "", genRandomName()))
			System.err.println("[ERROR][MATMUL][Cannot Execute][Tensor " + a + ", -]");
		return instance;
	}

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs matrix multiplication between two specified tensors.
	 *
	 * <p>
	 * Full parameter: {@code matmul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both input tensors are specified by name.</li>
	 *   <li>Both tensors must be at least 3D. If both tensors are 1D or 2D, calls {@code dot()} instead.</li>
	 *   <li>Broadcasts all leading axes except the last two.</li>
	 *   <li>{@code a.shape[-1]} must match {@code b.shape[-2]}.</li>
	 *   <li>Result shape: broadcasted leading dims + [a.shape[-2], b.shape[-1]].</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge matmul(String a, String b) {
		if (!CuBridgeJNI.matmul(a, b, genRandomName()))
			System.err.println("[ERROR][MATMUL][Cannot Execute][Tensor " + a + ", " + b + "]");
		return instance;
	}

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs matrix multiplication between two specified tensors and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code matmul(String a, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both input and output tensors are specified by name.</li>
	 *   <li>Both tensors must be at least 3D. If both tensors are 1D or 2D, calls {@code dot()} instead.</li>
	 *   <li>Broadcasts all leading axes except the last two.</li>
	 *   <li>{@code a.shape[-1]} must match {@code b.shape[-2]}.</li>
	 *   <li>Result shape: broadcasted leading dims + [a.shape[-2], b.shape[-1]].</li>
	 *   <li>The result is stored under the specified name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge matmul(String a, String b, String out) {
		if (!CuBridgeJNI.matmul(a, b, out))
			System.err.println("[ERROR][MATMUL][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	// 신경망 특화
	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the Mean Squared Error (MSE) loss between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code mse(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top two tensors in the queue as inputs.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse() {
		if (!CuBridgeJNI.mse("", "", genRandomName()))
			System.err.println("[ERROR][MSE][Cannot Execute][Tensor -, -]");
		return instance;
	}

	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the MSE loss between the specified prediction tensor and the top target tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code mse(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The prediction tensor is specified by name; the target is taken from the queue.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param yh the name of the predicted output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse(String yh) {
		if (!CuBridgeJNI.mse(yh, "", genRandomName()))
			System.err.println("[ERROR][MSE][Cannot Execute][Tensor " + yh + ", -]");
		return instance;
	}

	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the MSE loss between the specified prediction and target tensors.
	 *
	 * <p>
	 * Full parameter: {@code mse(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both prediction and target tensors are specified by name.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse(String yh, String y) {
		if (!CuBridgeJNI.mse(yh, y, genRandomName()))
			System.err.println("[ERROR][MSE][Cannot Execute][Tensor " + yh + ", " + y + "]");
		return instance;
	}

	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the MSE loss between the specified prediction and target tensors and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code mse(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Prediction, target, and output tensors are all specified by name.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>The result is a scalar (1x1 tensor) containing the total loss, stored under the given name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse(String yh, String y, String out) {
		if (!CuBridgeJNI.mse(yh, y, out))
			System.err.println("[ERROR][MSE][Cannot Execute][Tensor " + yh + ", " + y + ", out=" + out + "]");
		return instance;
	}

	/**
	 * Loss Operation (Cross-Entropy Error)
	 *
	 * Computes the Cross-Entropy Error (CEE) loss between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code cee(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top two tensors in the queue as inputs.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee() {
		if (!CuBridgeJNI.cee("", "", genRandomName()))
			System.err.println("[ERROR][CEE][Cannot Execute][Tensor -, -]");
		return instance;
	}

	/**
	 * Loss Operation (Cross-Entropy Error)
	 *
	 * Computes the CEE loss between the specified prediction tensor and the top target tensor in the queue.
	 *
	 * <p>
	 * Full parameter: {@code cee(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>The prediction tensor is specified by name; the target is taken from the queue.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param yh the name of the predicted output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee(String yh) {
		if (!CuBridgeJNI.cee(yh, "", genRandomName()))
			System.err.println("[ERROR][CEE][Cannot Execute][Tensor " + yh + ", -]");
		return instance;
	}

	/**
	 * Loss Operation (Cross-Entropy Error)
	 *
	 * Computes the CEE loss between the specified prediction and target tensors.
	 *
	 * <p>
	 * Full parameter: {@code cee(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both prediction and target tensors are specified by name.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee(String yh, String y) {
		if (!CuBridgeJNI.cee(yh, y, genRandomName()))
			System.err.println("[ERROR][CEE][Cannot Execute][Tensor " + yh + ", " + y + "]");
		return instance;
	}

	/**
	 * Loss Operation (Cross-Entropy Error)
	 *
	 * Computes the CEE loss between the specified prediction and target tensors and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code cee(String yh, String y, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Prediction, target, and output tensors are all specified by name.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>If any input contains negative values, the result will be NaN.</li>
	 *   <li>The result is a scalar (1x1 tensor) containing the total loss, stored under the given name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param yh the name of the predicted output tensor
	 * @param y the name of the target tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee(String yh, String y, String out) {
		if (!CuBridgeJNI.cee(yh, y, out))
			System.err.println("[ERROR][CEE][Cannot Execute][Tensor " + yh + ", " + y + ", out=" + out + "]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation: x · w + b.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top three tensors in the queue as inputs (input, weight, bias).</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w} for valid dot product.</li>
	 *   <li>{@code w} (weight tensor) supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} (bias tensor) must match the output's column count (last axis of {@code w}), and may be broadcasted regardless of its setting.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine() {
		if (!CuBridgeJNI.affine("", "", "", genRandomName()))
			System.err.println("[ERROR][AFFINE][Cannot Execute][Tensor -, -, -, -]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation: x · w + b using the specified input tensor x.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>{@code x} is specified by name; {@code w} and {@code b} are taken from the queue.</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w} for valid dot product.</li>
	 *   <li>{@code w} (weight tensor) supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} (bias tensor) must match the output's column count (last axis of {@code w}), and may be broadcasted regardless of its setting.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine(String x) {
		if (!CuBridgeJNI.affine(x, "", "", genRandomName()))
			System.err.println("[ERROR][AFFINE][Cannot Execute][Tensor " + x + ", -, -, -]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation: x · w + b using the specified input and weight tensors.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>{@code x} and {@code w} are specified by name; {@code b} is taken from the queue.</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w} for valid dot product.</li>
	 *   <li>{@code w} (weight tensor) supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} (bias tensor) must match the output's column count (last axis of {@code w}), and may be broadcasted regardless of its setting.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x the name of the input tensor
	 * @param w the name of the weight tensor (broadcast allowed on all axes except last two)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine(String x, String w) {
		if (!CuBridgeJNI.affine(x, w, "", genRandomName()))
			System.err.println("[ERROR][AFFINE][Cannot Execute][Tensor " + x + ", " + w + ", -, -]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation: x · w + b using the specified input, weight, and bias tensors.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>All input tensors are specified by name.</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w} for valid dot product.</li>
	 *   <li>{@code w} (weight tensor) supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} (bias tensor) must match the output's column count (last axis of {@code w}), and may be broadcasted regardless of its setting.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x the name of the input tensor
	 * @param w the name of the weight tensor (broadcast allowed on all axes except last two)
	 * @param b the name of the bias tensor (broadcast allowed to output column count)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine(String x, String w, String b) {
		if (!CuBridgeJNI.affine(x, w, b, genRandomName()))
			System.err.println("[ERROR][AFFINE][Cannot Execute][Tensor " + x + ", " + w + ", " + b + ", -]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation: x · w + b and stores the result in the specified output tensor.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>All input and output tensors are specified by name.</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w} for valid dot product.</li>
	 *   <li>{@code w} (weight tensor) supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} (bias tensor) must match the output's column count (last axis of {@code w}), and may be broadcasted regardless of its setting.</li>
	 *   <li>If no suitable tensors are found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x the name of the input tensor
	 * @param w the name of the weight tensor (broadcast allowed on all axes except last two)
	 * @param b the name of the bias tensor (broadcast allowed to output column count)
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine(String x, String w, String b, String out) {
		if (!CuBridgeJNI.affine(x, w, b, out))
			System.err.println("[ERROR][AFFINE][Cannot Execute][Tensor " + x + ", " + w + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Applies the softmax function to the top tensor in the queue along axis 1.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Transforms input into a probability distribution along axis 1.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax() {
		if (!CuBridgeJNI.softmax("", genRandomName(), 1))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor -, -, axis=1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Applies the softmax function to the specified tensor along axis 1.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Transforms input into a probability distribution along axis 1.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String name) {
		if (!CuBridgeJNI.softmax(name, genRandomName(), 1))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + name + ", -, axis=1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Applies the softmax function to the specified tensor along axis 1 and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Transforms input into a probability distribution along axis 1.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String name, String out) {
		if (!CuBridgeJNI.softmax(name, out, 1))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + name + ", " + out + ", axis=1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Applies the softmax function to the top tensor in the queue along the specified axis.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Applies softmax along the specified axis.</li>
	 *   <li>Transforms input into a probability distribution along the specified axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(int axis) {
		if (!CuBridgeJNI.softmax("", genRandomName(), axis))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Applies the softmax function to the specified tensor along the given axis.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Applies softmax along the specified axis.</li>
	 *   <li>Transforms input into a probability distribution along the specified axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @param axis the axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String name, int axis) {
		if (!CuBridgeJNI.softmax(name, genRandomName(), axis))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + name + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Applies the softmax function to the specified tensor along the given axis and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensors are specified by name.</li>
	 *   <li>Applies softmax along the specified axis.</li>
	 *   <li>Transforms input into a probability distribution along the specified axis.</li>
	 *   <li>If no suitable tensor is found, the operation fails and an error message is printed.</li>
	 * </ul>
	 * </p>
	 *
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @param axis the axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String name, String out, int axis) {
		if (!CuBridgeJNI.softmax(name, out, axis))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + name + ", " + out + ", axis=" + axis + "]");
		return instance;
	}
	
	/**
	 * Transformation Operation (im2col 1D)
	 *
	 * Converts a 1D input tensor into column format using a 1D kernel tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col1D(String input, String kernel, String out, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>Applies zero padding and unit stride (pad = 0, stride = 1).</li>
	 *   <li>Both input and kernel tensors must exist in the queue.</li>
	 *   <li>The result is stored under the specified output name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col1D(String input, String kernel, String out) {
		if(!CuBridgeJNI.im2col1D(input, kernel, out, 0, 1))
			System.err.println("[ERROR][IM2COL][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (im2col 1D)
	 *
	 * Converts a 1D input tensor into column format using a kernel and specified padding.
	 *
	 * <p>
	 * Full parameter: {@code im2col1D(String input, String kernel, String out, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>Uses the given padding and default stride = 1.</li>
	 *   <li>Padding is applied symmetrically to both sides.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @param pad    padding applied on both sides
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col1D(String input, String kernel, String out, int pad) {
		if(!CuBridgeJNI.im2col1D(input, kernel, out, pad, 1))
			System.err.println("[ERROR][IM2COL][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}

	/**
	 * Transformation Operation (im2col 1D)
	 *
	 * Converts a 1D input tensor into column format using kernel, padding, and stride.
	 *
	 * <p>
	 * Full parameter: {@code im2col1D(String input, String kernel, String out, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>All parameters explicitly provided: kernel, padding, stride.</li>
	 *   <li>Supports arbitrary stride and padding values.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @param pad    padding applied to both sides
	 * @param stride stride value between kernel applications
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col1D(String input, String kernel, String out, int pad, int stride) {
		if(!CuBridgeJNI.im2col1D(input, kernel, out, pad, stride))
			System.err.println("[ERROR][IM2COL][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Reconstructs the original 1D input tensor from a column matrix.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>Uses default output length = -1, padding = 0, stride = 1.</li>
	 *   <li>Output shape is inferred when oL = -1.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the reconstructed output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im1D(String input, String kernel, String out) {
		if(!CuBridgeJNI.col2im1D(input, kernel, out, -1, 0, 1))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Reconstructs a 1D input tensor from column format using specified padding.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>Uses default output length = -1 and stride = 1.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @param pad    padding applied during im2col1D
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im1D(String input, String kernel, String out, int pad) {
		if(!CuBridgeJNI.col2im1D(input, kernel, out, -1, pad, 1))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Reconstructs a 1D input tensor using padding and stride information.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>Output length is still inferred (oL = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @param pad    padding used during im2col
	 * @param stride stride used during im2col
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im1D(String input, String kernel, String out, int pad, int stride) {
		if(!CuBridgeJNI.col2im1D(input, kernel, out, -1, pad, stride))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Fully reconstructs a 1D input tensor from column format with all parameters specified.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 2D or 3D (shape: [N, W] or [N, C, W]).</li>	
	 *   <li>Explicitly defines output length, padding, and stride.</li>
	 *   <li>Ensures accurate inverse transformation of im2col1D.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the reconstructed output tensor
	 * @param oL     original input length
	 * @param pad    padding used during im2col
	 * @param stride stride used during im2col
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im1D(String input, String kernel, String out, int oL, int pad, int stride) {
		if(!CuBridgeJNI.col2im1D(input, kernel, out, oL, pad, stride))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format using a 2D kernel tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Applies symmetric padding = 0 and stride = 1.</li>
	 *   <li>The result is stored under the specified output name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input   the name of the 2D input tensor
	 * @param kernel  the name of the 2D kernel tensor
	 * @param out     the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col2D(String input, String kernel, String out) {
		if(!CuBridgeJNI.im2col2D(input, kernel, out, 0, 0, 1, 1))
			System.err.println("[ERROR][IM2COL2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format using symmetric padding.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Applies the same padding to height and width (padH = padW = pad).</li>
	 *   <li>Uses default stride = 1 for both dimensions.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input   the name of the 2D input tensor
	 * @param kernel  the name of the 2D kernel tensor
	 * @param out     the name for the output tensor
	 * @param pad     symmetric padding for height and width
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col2D(String input, String kernel, String out, int pad) {
		if(!CuBridgeJNI.im2col2D(input, kernel, out, pad, pad, 1, 1))
			System.err.println("[ERROR][IM2COL2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format using symmetric padding and stride.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Applies the same value for padding and stride on both dimensions.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input   the name of the 2D input tensor
	 * @param kernel  the name of the 2D kernel tensor
	 * @param out     the name for the output tensor
	 * @param pad     padding value for both height and width
	 * @param stride  stride value for both height and width
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col2D(String input, String kernel, String out, int pad, int stride) {
		if(!CuBridgeJNI.im2col2D(input, kernel, out, pad, pad, stride, stride))
			System.err.println("[ERROR][IM2COL2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format using full control over padding and stride.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Supports asymmetric padding and non-uniform stride across height and width.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the name of the 2D input tensor
	 * @param kernel   the name of the 2D kernel tensor
	 * @param out      the name for the output tensor
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW) {
		if(!CuBridgeJNI.im2col2D(input, kernel, out, padH, padW, strideH, strideW))
			System.err.println("[ERROR][IM2COL2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Reconstructs the original 2D input tensor from a column matrix.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Uses default output size = -1 and symmetric padding = 0, stride = 1.</li>
	 *   <li>If output size is -1, it will be inferred from kernel and input shapes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input   the name of the column input tensor
	 * @param kernel  the name of the kernel tensor
	 * @param out     the name for the reconstructed output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im2D(String input, String kernel, String out) {
		if(!CuBridgeJNI.col2im2D(input, kernel, out, -1, -1, 0, 0, 1, 1))
			System.err.println("[ERROR][COL2IM2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Reconstructs a 2D input tensor using symmetric padding, with output size inferred.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Uses oH = oW = -1 (auto-infer), stride = 1.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input   the name of the column input tensor
	 * @param kernel  the name of the kernel tensor
	 * @param out     the name for the reconstructed output tensor
	 * @param pad     symmetric padding used during im2col
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im2D(String input, String kernel, String out, int pad) {
		if(!CuBridgeJNI.col2im2D(input, kernel, out, -1, -1, pad, pad, 1, 1))
			System.err.println("[ERROR][COL2IM2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
	
	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Reconstructs a 2D input tensor using symmetric padding and stride.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Uses oH = oW = -1 (auto-infer).</li>
	 * </ul>
	 * </p>
	 *
	 * @param input   the name of the column input tensor
	 * @param kernel  the name of the kernel tensor
	 * @param out     the name for the reconstructed output tensor
	 * @param pad     symmetric padding used during im2col
	 * @param stride  symmetric stride used during im2col
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im2D(String input, String kernel, String out, int pad, int stride) {
		if(!CuBridgeJNI.col2im2D(input, kernel, out, -1, -1, pad, pad, stride, stride))
			System.err.println("[ERROR][COL2IM2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}

	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Fully reconstructs a 2D input tensor using all output and kernel parameters.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version:
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>All parameters are explicitly specified, including target output height and width.</li>
	 *   <li>Should match the configuration used in the corresponding im2col2D operation.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the name of the column input tensor
	 * @param kernel   the name of the kernel tensor
	 * @param out      the name for the reconstructed output tensor
	 * @param oH       original input height
	 * @param oW       original input width
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW) {
		if(!CuBridgeJNI.col2im2D(input, kernel, out, oH, oW, padH, padW, strideH, strideW))
			System.err.println("[ERROR][COL2IM2D][Cannot Execute][Tensor " + input + "] Please verify tensor existence and parameters.");
		
		return instance;
	}
}
