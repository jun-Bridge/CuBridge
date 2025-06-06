package CuBridge;

import java.util.UUID;

public class CuBridge {
	private static final CuBridge instance = new CuBridge();

	private CuBridge() {
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
	 * Prints the tensor queue for the specified name.
	 * No duplicate names exist. Tensors without user-defined names
	 * are shown with temporary auto-generated names.
	 *
	 * @param name the tensor name to inspect
	 */
	public void visualQueue(String name) {
		System.out.println(CuBridgeJNI.visualQueue(name));
	} // 스택 내부 특정 텐서만

	/**
	 * Prints all tensors currently stored in the queue.
	 * Shows every tensor under each name in order.
	 * Unnamed tensors use temporary auto-generated names.
	 */
	public void visualQueue() {
		System.out.println(CuBridgeJNI.visualQueueAll());
	} // 스택 내부 특정 텐서만

	/**
	 * Prints the tensor buffer for the specified name.
	 * No duplicate names exist. Tensors without user-defined names
	 * are shown with temporary auto-generated names.
	 *
	 * @param name the tensor name to inspect
	 */
	public void visualBuffer(String name) {
		System.out.println(CuBridgeJNI.visualBuffer(name));
	} // 스택 내부 특정 텐서만

	/**
	 * Prints all tensors currently stored in the buffer.
	 * Shows every tensor under each name in order.
	 * Unnamed tensors use temporary auto-generated names.
	 */
	public void visualBuffer() {
		System.out.println(CuBridgeJNI.visualBufferAll());
	} // 스택 내부 특정 텐서만

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
	 * </p>
	 *
	 * @param data        the integer value to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1)
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
	 * </p>
	 *
	 * @param data        the double value to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1)
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
	 * </p>
	 *
	 * @param data        the tensor to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(Tensor data, String name, int usageCount) {
		return put(data, name, usageCount, false);
	}

	/**
	 * Stores a tensor with complete configuration.
	 *
	 * @param data        the tensor to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1 for unlimited)
	 * @param broadcast   whether the tensor is broadcastable
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
		if (!CuBridgeJNI.abs("", ""))
			System.err.println("ABS Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.abs(a, ""))
			System.err.println("ABS Error: The " + a + " does not exist in Queue!");

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
			System.err.println("ABS Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.neg("", ""))
			System.err.println("NEG Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.neg(a, ""))
			System.err.println("NEG Error: The " + a + " does not exist in Queue!");

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
			System.err.println("NEG Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.square("", ""))
			System.err.println("SQUARE Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.square(a, ""))
			System.err.println("SQUARE Error: The " + a + " does not exist in Queue!");

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
			System.err.println("SQUARE Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.sqrt("", ""))
			System.err.println("SQRT Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.sqrt(a, ""))
			System.err.println("SQRT Error: The " + a + " does not exist in Queue!");

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
			System.err.println("SQRT Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.log("", ""))
			System.err.println("LOG Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.log(a, ""))
			System.err.println("LOG Error: The " + a + " does not exist in Queue!");

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
			System.err.println("LOG Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.log2("", ""))
			System.err.println("LOG_2 Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.log2(a, ""))
			System.err.println("LOG_2 Error: The " + a + " does not exist in Queue!");

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
			System.err.println("LOG_2 Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.ln("", ""))
			System.err.println("LOG_e Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.ln(a, ""))
			System.err.println("LOG_e Error: The " + a + " does not exist in Queue!");

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
			System.err.println("LOG_e Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.reciprocal("", ""))
			System.err.println("REVERSE Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.reciprocal(a, ""))
			System.err.println("REVERSE Error: The " + a + " does not exist in Queue!");

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
			System.err.println("REVERSE Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.sin("", ""))
			System.err.println("SIN Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sin(String name) {
		if (!CuBridgeJNI.sin(name, ""))
			System.err.println("SIN Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sin(String name, String out) {
		if (!CuBridgeJNI.sin(name, out))
			System.err.println("SIN Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.cos("", ""))
			System.err.println("COS Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cos(String name) {
		if (!CuBridgeJNI.cos(name, ""))
			System.err.println("COS Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cos(String name, String out) {
		if (!CuBridgeJNI.cos(name, out))
			System.err.println("COS Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.tan("", ""))
			System.err.println("TAN Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tan(String name) {
		if (!CuBridgeJNI.tan(name, ""))
			System.err.println("TAN Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tan(String name, String out) {
		if (!CuBridgeJNI.tan(name, out))
			System.err.println("TAN Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.step("", ""))
			System.err.println("STEP Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge step(String name) {
		if (!CuBridgeJNI.step(name, ""))
			System.err.println("STEP Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge step(String name, String out) {
		if (!CuBridgeJNI.step(name, out))
			System.err.println("STEP Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.sigmoid("", ""))
			System.err.println("SIGMOID Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sigmoid(String name) {
		if (!CuBridgeJNI.sigmoid(name, ""))
			System.err.println("SIGMOID Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sigmoid(String name, String out) {
		if (!CuBridgeJNI.sigmoid(name, out))
			System.err.println("SIGMOID Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.tanh("", ""))
			System.err.println("TANH Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tanh(String name) {
		if (!CuBridgeJNI.tanh(name, ""))
			System.err.println("TANH Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge tanh(String name, String out) {
		if (!CuBridgeJNI.tanh(name, out))
			System.err.println("TANH Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.ReLu("", ""))
			System.err.println("RELU Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */

	public CuBridge relu(String name) {
		if (!CuBridgeJNI.ReLu(name, ""))
			System.err.println("RELU Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge relu(String name, String out) {
		if (!CuBridgeJNI.ReLu(name, out))
			System.err.println("RELU Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.leakReLu("", ""))
			System.err.println("LEAKRELU Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge leakrelu(String name) {
		if (!CuBridgeJNI.leakReLu(name, ""))
			System.err.println("LEAKRELU Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge leakrelu(String name, String out) {
		if (!CuBridgeJNI.leakReLu(name, out))
			System.err.println("LEAKRELU Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.softplus("", ""))
			System.err.println("SOFTPLUS Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softplus(String name) {
		if (!CuBridgeJNI.softplus(name, ""))
			System.err.println("SOFTPLUS Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softplus(String name, String out) {
		if (!CuBridgeJNI.softplus(name, out))
			System.err.println("SOFTPLUS Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.exp("", ""))
			System.err.println("EXP Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge exp(String name) {
		if (!CuBridgeJNI.exp(name, ""))
			System.err.println("EXP Error: The " + name + " does not exist in Queue!");

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
	 * @param name   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge exp(String name, String out) {
		if (!CuBridgeJNI.exp(name, out))
			System.err.println("EXP Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.round("", ""))
			System.err.println("ROUND Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge round(String name) {
		if (!CuBridgeJNI.round(name, ""))
			System.err.println("ROUND Error: The " + name + " does not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge round(String name, String out) {
		if (!CuBridgeJNI.round(name, out))
			System.err.println("ROUND Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.ceil("", ""))
			System.err.println("CEIL Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ceil(String name) {
		if (!CuBridgeJNI.ceil(name, ""))
			System.err.println("CEIL Error: The " + name + " does not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @param out  the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge ceil(String name, String out) {
		if (!CuBridgeJNI.ceil(name, out))
			System.err.println("CEIL Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.floor("", ""))
			System.err.println("FLOOR Error: Tensor not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge floor(String name) {
		if (!CuBridgeJNI.floor(name, ""))
			System.err.println("FLOOR Error: The " + name + " does not exist in Queue!");

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
	 * @param name the name of the input tensor
	 * @param out  the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge floor(String name, String out) {
		if (!CuBridgeJNI.floor(name, out))
			System.err.println("FLOOR Error: The " + name + " does not exist in Queue!");

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
		if (!CuBridgeJNI.not("", ""))
			System.err.println("NOT Error: Tensor not exist in Queue!");

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
		if (!CuBridgeJNI.not(a, ""))
			System.err.println("NOT Error: The " + a + " does not exist in Queue!");

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
			System.err.println("NOT Error: The " + a + " does not exist in Queue!");

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
		if (!CuBridgeJNI.add("", "", ""))
			System.err.println("ADD Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.add(a, "", ""))
			System.err.println("ADD Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.add(a, b, ""))
			System.err.println("ADD Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("ADD Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.sub("", "", ""))
			System.err.println("SUB Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.sub(a, "", ""))
			System.err.println("SUB Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.sub(a, b, ""))
			System.err.println("SUB Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("SUB Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mul("", "", ""))
			System.err.println("MUL Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.mul(a, "", ""))
			System.err.println("MUL Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mul(a, b, ""))
			System.err.println("MUL Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("MUL Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.div("", "", ""))
			System.err.println("DIV Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.div(a, "", ""))
			System.err.println("DIV Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.div(a, b, ""))
			System.err.println("DIV Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("DIV Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.pow("", "", ""))
			System.err.println("POW Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.pow(a, "", ""))
			System.err.println("POW Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.pow(a, b, ""))
			System.err.println("POW Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("POW Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mod("", "", ""))
			System.err.println("MOD Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.mod(a, "", ""))
			System.err.println("MOD Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mod(a, b, ""))
			System.err.println("MOD Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("MOD Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.gt("", "", ""))
			System.err.println("GreaterThan Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.gt(a, "", ""))
			System.err.println("GreaterThan Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.gt(a, b, ""))
			System.err.println("GreaterThan Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("GreaterThan Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.lt("", "", ""))
			System.err.println("LessThan Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.lt(a, "", ""))
			System.err.println("LessThan Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.lt(a, b, ""))
			System.err.println("LessThan Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("LessThan Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.ge("", "", ""))
			System.err.println("GreaterEqual Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.ge(a, "", ""))
			System.err.println("GreaterEqual Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.ge(a, b, ""))
			System.err.println("GreaterEqual Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("GreaterEqual Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.le("", "", ""))
			System.err.println("LessEqual Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.le(a, "", ""))
			System.err.println("LessEqual Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.le(a, b, ""))
			System.err.println("LessEqual Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("LessEqual Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.eq("", "", ""))
			System.err.println("Equal Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.eq(a, "", ""))
			System.err.println("Equal Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.eq(a, b, ""))
			System.err.println("Equal Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("Equal Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.ne("", "", ""))
			System.err.println("NotEqual Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.ne(a, "", ""))
			System.err.println("NotEqual Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.ne(a, b, ""))
			System.err.println("NotEqual Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("NotEqual Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.and("", "", ""))
			System.err.println("AND Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.and(a, "", ""))
			System.err.println("AND Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.and(a, b, ""))
			System.err.println("AND Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("AND Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.or("", "", ""))
			System.err.println("OR Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.or(a, "", ""))
			System.err.println("OR Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.or(a, b, ""))
			System.err.println("OR Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("OR Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.sum("", "", -1))
			System.err.println("SUM Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.sum("", "", axis))
			System.err.println("SUM Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.sum(a, "", -1))
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.sum(a, "", axis))
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");
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
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");
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
			System.err.println("SUM Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mean("", "", -1))
			System.err.println("MEAN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.mean("", "", axis))
			System.err.println("MEAN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.mean(a, "", -1))
			System.err.println("MEAN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mean(a, "", axis))
			System.err.println("MEAN Error: The " + a + " does not exist in Queue!");
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
			System.err.println("MEAN Error: The " + a + " does not exist in Queue!");
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
			System.err.println("MEAN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.var("", "", -1))
			System.err.println("VAR Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.var("", "", axis))
			System.err.println("VAR Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.var(a, "", -1))
			System.err.println("VAR Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.var(a, "", axis))
			System.err.println("VAR Error: The " + a + " does not exist in Queue!");
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
			System.err.println("VAR Error: The " + a + " does not exist in Queue!");
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
			System.err.println("VAR Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.std("", "", -1))
			System.err.println("STD Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.std("", "", axis))
			System.err.println("STD Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.std(a, "", -1))
			System.err.println("STD Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.std(a, "", axis))
			System.err.println("STD Error: The " + a + " does not exist in Queue!");
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
			System.err.println("STD Error: The " + a + " does not exist in Queue!");
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
			System.err.println("STD Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.max("", "", -1))
			System.err.println("MAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.max("", "", axis))
			System.err.println("MAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.max(a, "", -1))
			System.err.println("MAX Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.max(a, "", axis))
			System.err.println("MAX Error: The " + a + " does not exist in Queue!");
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
			System.err.println("MAX Error: The " + a + " does not exist in Queue!");
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
			System.err.println("MAX Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.min("", "", -1))
			System.err.println("MIN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.min("", "", axis))
			System.err.println("MIN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.min(a, "", -1))
			System.err.println("MIN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.min(a, "", axis))
			System.err.println("MIN Error: The " + a + " does not exist in Queue!");
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
			System.err.println("MIN Error: The " + a + " does not exist in Queue!");
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
			System.err.println("MIN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.accumulate("", "", -1))
			System.err.println("ACCUMULATE Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.accumulate("", "", axis))
			System.err.println("ACCUMULATE Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.accumulate(a, "", -1))
			System.err.println("ACCUMULATE Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.accumulate(a, "", axis))
			System.err.println("ACCUMULATE Error: The " + a + " does not exist in Queue!");
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
			System.err.println("ACCUMULATE Error: The " + a + " does not exist in Queue!");
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
			System.err.println("ACCUMULATE Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.compress("", "", -1))
			System.err.println("COMPRESS Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.compress("", "", axis))
			System.err.println("COMPRESS Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.compress(a, "", -1))
			System.err.println("COMPRESS Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.compress(a, "", axis))
			System.err.println("COMPRESS Error: The " + a + " does not exist in Queue!");
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
			System.err.println("COMPRESS Error: The " + a + " does not exist in Queue!");
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
			System.err.println("COMPRESS Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.expand("", "", axis, N))
			System.err.println("EXPAND Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.expand(a, "", axis, N))
			System.err.println("EXPAND Error: The " + a + " does not exist in Queue!");
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
			System.err.println("EXPAND Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.argMax("", "", -1))
			System.err.println("ARGMAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.argMax("", "", axis))
			System.err.println("ARGMAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.argMax(a, "", -1))
			System.err.println("ARGMAX Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.argMax(a, "", axis))
			System.err.println("ARGMAX Error: The " + a + " does not exist in Queue!");
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
			System.err.println("ARGMAX Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.argMin("", "", -1))
			System.err.println("ARGMIN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.argMin("", "", axis))
			System.err.println("ARGMIN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.argMin(a, "", -1))
			System.err.println("ARGMIN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.argMin(a, "", axis))
			System.err.println("ARGMIN Error: The " + a + " does not exist in Queue!");
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
			System.err.println("ARGMIN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.axisMax("", "", -1))
			System.err.println("AXISMAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.axisMax("", "", axis))
			System.err.println("AXISMAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.axisMax(a, "", -1))
			System.err.println("AXISMAX Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.axisMax(a, "", axis))
			System.err.println("AXISMAX Error: The " + a + " does not exist in Queue!");
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
			System.err.println("AXISMAX Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.axisMin("", "", -1))
			System.err.println("AXISMIN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.axisMin("", "", axis))
			System.err.println("AXISMIN Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.axisMin(a, "", -1))
			System.err.println("AXISMIN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.axisMin(a, "", axis))
			System.err.println("AXISMIN Error: The " + a + " does not exist in Queue!");
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
			System.err.println("AXISMIN Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.transpose("", "", 0, -1))
			System.err.println("TRANSPOSE Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.transpose(a, "", 0, -1))
			System.err.println("TRANSPOSE Error: The " + a + " does not exist in Queue!");
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
			System.err.println("TRANSPOSE Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.transpose("", "", axis1, axis2))
			System.err.println("TRANSPOSE Error: Tensor not exist OR axis too big in Queue!");
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
		if (!CuBridgeJNI.transpose(a, "", axis1, axis2))
			System.err.println("TRANSPOSE Error: The " + a + " does not exist OR axis too big in Queue!");
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
			System.err.println("TRANSPOSE Error: The " + a + " does not exist OR axis too big in Queue!");
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
		if (!CuBridgeJNI.dot("", "", ""))
			System.err.println("DOT Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.dot(a, "", ""))
			System.err.println("DOT Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.dot(a, b, ""))
			System.err.println("DOT Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("DOT Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.matmul("", "", ""))
			System.err.println("MATMUL Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.matmul(a, "", ""))
			System.err.println("MATMUL Error: The " + a + " does not exist in Queue!");
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
		if (!CuBridgeJNI.matmul(a, b, ""))
			System.err.println("MATMUL Error: The " + a + " or " + b + " does not exist in Queue!");
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
			System.err.println("MATMUL Error: The " + a + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mse("", "", ""))
			System.err.println("MSE Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.mse(yh, "", ""))
			System.err.println("MSE Error: The " + yh + " does not exist in Queue!");
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
		if (!CuBridgeJNI.mse(yh, y, ""))
			System.err.println("MSE Error: The " + yh + " or " + y + " does not exist in Queue!");
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
			System.err.println("MSE Error: The " + yh + " or " + y + " does not exist in Queue!");
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
		if (!CuBridgeJNI.cee("", "", ""))
			System.err.println("CEE Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.cee(yh, "", ""))
			System.err.println("CEE Error: The " + yh + " does not exist in Queue!");
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
		if (!CuBridgeJNI.cee(yh, y, ""))
			System.err.println("CEE Error: The " + yh + " or " + y + " does not exist in Queue!");
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
			System.err.println("CEE Error: The " + yh + " or " + y + " does not exist in Queue!");
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
		if (!CuBridgeJNI.affine("", "", "", ""))
			System.err.println("AFFINE Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.affine(x, "", "", ""))
			System.err.println("AFFINE Error: The " + x + " does not exist in Queue!");
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
		if (!CuBridgeJNI.affine(x, w, "", ""))
			System.err.println("AFFINE Error: The " + x + " or " + w + " does not exist in Queue!");
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
		if (!CuBridgeJNI.affine(x, w, b, ""))
			System.err.println("AFFINE Error: The " + x + " or " + w + " or " + b + " does not exist in Queue!");
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
			System.err.println("AFFINE Error: The " + x + " or " + w + " or " + b + " does not exist in Queue!");
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
		if (!CuBridgeJNI.softmax("", "", 1))
			System.err.println("SOFTMAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.softmax(name, "", 1))
			System.err.println("SOFTMAX Error: The " + name + " does not exist in Queue!");
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
			System.err.println("SOFTMAX Error: The " + name + " does not exist in Queue!");
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
		if (!CuBridgeJNI.softmax("", "", axis))
			System.err.println("SOFTMAX Error: Tensor not exist in Queue!");
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
		if (!CuBridgeJNI.softmax(name, "", axis))
			System.err.println("SOFTMAX Error: The " + name + " does not exist in Queue!");
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
			System.err.println("SOFTMAX Error: The " + name + " does not exist in Queue!");
		return instance;
	}
}
