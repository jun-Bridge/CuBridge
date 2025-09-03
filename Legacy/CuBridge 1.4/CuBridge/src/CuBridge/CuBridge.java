package CuBridge;

import java.util.UUID;

public class CuBridge {
	private static final CuBridge instance = new CuBridge();

	private CuBridge() {
		loadConst();
	}
	
	private void loadConst() {
	    put(1.0f, "_ONE", -1);
	    put(2.0f, "_TWO", -1);
	    put(3.0f, "_THREE", -1);
	    put(4.0f, "_FOUR", -1);
	    put(5.0f, "_FIVE", -1);
	    put(6.0f, "_SIX", -1);
	    put(7.0f, "_SEVEN", -1);
	    put(8.0f, "_EIGHT", -1);
	    put(9.0f, "_NINE", -1);
	    put(0.0f, "_ZERO", -1);
	    put(0.5f, "_HALF", -1);
	    put(100.0f, "_HUNDRED", -1);
	    put(255.0f, "_MAXPIXEL", -1);
	    put(-1.0f, "_NEG", -1);
	    put(1e-6f, "_EPSILON", -1);
	    put(0.001f, "_RATE", -1);
	    put(3.14159265359f, "_PI", -1);
	    put(2.718281f, "_E", -1);
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
	 * Stores a float scalar tensor.
	 * <p>
	 * Full parameter: {@code put(float data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * <li>name is auto-generated</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the float value to store
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(float data) {
		return put(new Tensor(data), true);
	}

	/**
	 * Stores a float scalar tensor.
	 * <p>
	 * Full parameter: {@code put(float data, String name, int usageCount, boolean broadcast)}<br>
	 * This version:
	 * <ul>
	 * <li>broadcast = true (automatically marked as broadcastable)</li>
	 * <li>usageCount = 1 (default)</li>
	 * </ul>
	 * </p>
	 *
	 * @param data the float value to store
	 * @param name the tensor name (must be unique and non-empty)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(float data, String name) {
		return put(new Tensor(data), name, 1, true);
	}

	/**
	 * Stores a float scalar tensor.
	 * <p>
	 * Full parameter: {@code put(float data, String name, int usageCount, boolean broadcast)}<br>
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
	 * @param data        the float value to store
	 * @param name        the tensor name (must be unique and non-empty)
	 * @param usageCount  number of times this tensor will be used (>0 or -1 for constants)
	 * @return CuBridge instance for chaining
	 */
	public CuBridge put(float data, String name, int usageCount) {
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
		if(data == null){
			System.err.println("Error: Input Tensor is NULL.");
			return instance;
		}

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
		float[] data = CuBridgeJNI.getData(name);
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
	 * Full parameter: {@code abs(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code abs(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Full parameter: {@code abs(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies absolute value (|x|) to the given Tensor.
	 * <p>
	 * Full parameter: {@code abs(String a, String out)}<br>
	 * Full parameter: {@code abs(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge abs(Tensor a) {
		if (a == null) System.err.println("[ERROR][ABS][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).abs(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code abs(String a, String out)}<br>
	 * Full parameter: {@code abs(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge abs(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][ABS][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).abs(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code absI(String a)}<br>
	 * Full parameter: {@code absI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 absolute value (|x|)
	 */
	public Tensor absI() {
		String oName = "Imm_" + genRandomName();

		return abs("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code absI(String a)}<br>
	 * Full parameter: {@code absI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 absolute value (|x|)
	 */
	public Tensor absI(String a) {
		String oName = "Imm_" + genRandomName();

		return abs(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies absolute value (|x|) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code absI(String a)}<br>
	 * Full parameter: {@code absI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 absolute value (|x|)
	 */
	public Tensor absI(Tensor a) {
		if (a == null) System.err.println("[ERROR][ABS][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).abs(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code neg(String a, String out)}<br>
	 * Full parameter: {@code neg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code neg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Full parameter: {@code neg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies negation (-x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code neg(String a, String out)}<br>
	 * Full parameter: {@code neg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge neg(Tensor a) {
		if (a == null) System.err.println("[ERROR][NEG][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).neg(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code neg(String a, String out)}<br>
	 * Full parameter: {@code neg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge neg(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][NEG][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).neg(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code negI(String a)}<br>
	 * Full parameter: {@code negI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 negation (-x)
	 */
	public Tensor negI() {
		String oName = "Imm_" + genRandomName();

		return neg("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code negI(String a)}<br>
	 * Full parameter: {@code negI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 negation (-x)
	 */
	public Tensor negI(String a) {
		String oName = "Imm_" + genRandomName();

		return neg(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies negation (-x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code negI(String a)}<br>
	 * Full parameter: {@code negI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 negation (-x)
	 */
	public Tensor negI(Tensor a) {
		if (a == null) System.err.println("[ERROR][NEG][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).neg(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square (x²) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * Full parameter: {@code square(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies square (x²) to the specified tensor.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * Full parameter: {@code square(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Applies square (x²) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * Full parameter: {@code square(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies square (x²) to the given Tensor.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * Full parameter: {@code square(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge square(Tensor a) {
		if (a == null) System.err.println("[ERROR][SQUARE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).square(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square (x²) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code square(String a, String out)}<br>
	 * Full parameter: {@code square(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge square(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][SQUARE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).square(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square (x²) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code squareI(String a)}<br>
	 * Full parameter: {@code squareI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 square (x²)
	 */
	public Tensor squareI() {
		String oName = "Imm_" + genRandomName();

		return square("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square (x²) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code squareI(String a)}<br>
	 * Full parameter: {@code squareI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 square (x²)
	 */
	public Tensor squareI(String a) {
		String oName = "Imm_" + genRandomName();

		return square(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square (x²) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code squareI(String a)}<br>
	 * Full parameter: {@code squareI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 square (x²)
	 */
	public Tensor squareI(Tensor a) {
		if (a == null) System.err.println("[ERROR][SQUARE][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).square(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code sqrt(String a, String out)}<br>
	 * Full parameter: {@code sqrt(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code sqrt(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Full parameter: {@code sqrt(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies square root (√x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code sqrt(String a, String out)}<br>
	 * Full parameter: {@code sqrt(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sqrt(Tensor a) {
		if (a == null) System.err.println("[ERROR][SQRT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sqrt(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code sqrt(String a, String out)}<br>
	 * Full parameter: {@code sqrt(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sqrt(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][SQRT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sqrt(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code sqrtI(String a)}<br>
	 * Full parameter: {@code sqrtI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 square root (√x)
	 */
	public Tensor sqrtI() {
		String oName = "Imm_" + genRandomName();

		return sqrt("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code sqrtI(String a)}<br>
	 * Full parameter: {@code sqrtI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 square root (√x)
	 */
	public Tensor sqrtI(String a) {
		String oName = "Imm_" + genRandomName();

		return sqrt(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies square root (√x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code sqrtI(String a)}<br>
	 * Full parameter: {@code sqrtI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 square root (√x)
	 */
	public Tensor sqrtI(Tensor a) {
		if (a == null) System.err.println("[ERROR][SQRT][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).sqrt(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm (log₁₀x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * Full parameter: {@code log(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies logarithm (log₁₀x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * Full parameter: {@code log(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Applies logarithm (log₁₀x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * Full parameter: {@code log(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies logarithm (log₁₀x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * Full parameter: {@code log(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge log(Tensor a) {
		if (a == null) System.err.println("[ERROR][LOG][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).log(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm (log₁₀x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code log(String a, String out)}<br>
	 * Full parameter: {@code log(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge log(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][LOG][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).log(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm (log₁₀x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code logI(String a)}<br>
	 * Full parameter: {@code logI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 logarithm (log₁₀x)
	 */
	public Tensor logI() {
		String oName = "Imm_" + genRandomName();

		return log("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm (log₁₀x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code logI(String a)}<br>
	 * Full parameter: {@code logI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 logarithm (log₁₀x)
	 */
	public Tensor logI(String a) {
		String oName = "Imm_" + genRandomName();

		return log(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm (log₁₀x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code logI(String a)}<br>
	 * Full parameter: {@code logI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 logarithm (log₁₀x)
	 */
	public Tensor logI(Tensor a) {
		if (a == null) System.err.println("[ERROR][LOG][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).log(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm base 2 (log₂x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * Full parameter: {@code log2(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies logarithm base 2 (log₂x) to the specified tensor.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * Full parameter: {@code log2(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Applies logarithm base 2 (log₂x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * Full parameter: {@code log2(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies logarithm base 2 (log₂x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * Full parameter: {@code log2(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge log2(Tensor a) {
		if (a == null) System.err.println("[ERROR][LOG2][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).log2(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm base 2 (log₂x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code log2(String a, String out)}<br>
	 * Full parameter: {@code log2(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge log2(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][LOG2][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).log2(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm base 2 (log₂x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code log2I(String a)}<br>
	 * Full parameter: {@code log2I(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 logarithm base 2 (log₂x)
	 */
	public Tensor log2I() {
		String oName = "Imm_" + genRandomName();

		return log2("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm base 2 (log₂x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code log2I(String a)}<br>
	 * Full parameter: {@code log2I(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 logarithm base 2 (log₂x)
	 */
	public Tensor log2I(String a) {
		String oName = "Imm_" + genRandomName();

		return log2(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logarithm base 2 (log₂x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code log2I(String a)}<br>
	 * Full parameter: {@code log2I(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 logarithm base 2 (log₂x)
	 */
	public Tensor log2I(Tensor a) {
		if (a == null) System.err.println("[ERROR][LOG2][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).log2(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code ln(String a, String out)}<br>
	 * Full parameter: {@code ln(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code ln(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Full parameter: {@code ln(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies natural logarithm (ln x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code ln(String a, String out)}<br>
	 * Full parameter: {@code ln(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ln(Tensor a) {
		if (a == null) System.err.println("[ERROR][LN][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ln(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code ln(String a, String out)}<br>
	 * Full parameter: {@code ln(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ln(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][LN][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ln(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code lnI(String a)}<br>
	 * Full parameter: {@code lnI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 natural logarithm (ln x)
	 */
	public Tensor lnI() {
		String oName = "Imm_" + genRandomName();

		return ln("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code lnI(String a)}<br>
	 * Full parameter: {@code lnI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 natural logarithm (ln x)
	 */
	public Tensor lnI(String a) {
		String oName = "Imm_" + genRandomName();

		return ln(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies natural logarithm (ln x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code lnI(String a)}<br>
	 * Full parameter: {@code lnI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 natural logarithm (ln x)
	 */
	public Tensor lnI(Tensor a) {
		if (a == null) System.err.println("[ERROR][LN][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).ln(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code reciprocal(String a, String out)}<br>
	 * Full parameter: {@code reciprocal(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code reciprocal(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Applies reciprocal (1/x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code reciprocal(String a, String out)}<br>
	 * Full parameter: {@code reciprocal(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge reciprocal(String a, String out) {
		if (!CuBridgeJNI.reciprocal(a, out))
			System.err.println("[ERROR][RECIPROCAL][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code reciprocal(String a, String out)}<br>
	 * Full parameter: {@code reciprocal(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge reciprocal(Tensor a) {
		if (a == null) System.err.println("[ERROR][RECIPROCAL][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).reciprocal(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code reciprocal(String a, String out)}<br>
	 * Full parameter: {@code reciprocal(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge reciprocal(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][RECIPROCAL][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).reciprocal(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code reciprocalI(String a)}<br>
	 * Full parameter: {@code reciprocalI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 reciprocal (1/x)
	 */
	public Tensor reciprocalI() {
		String oName = "Imm_" + genRandomName();

		return reciprocal("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code reciprocalI(String a)}<br>
	 * Full parameter: {@code reciprocalI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 reciprocal (1/x)
	 */
	public Tensor reciprocalI(String a) {
		String oName = "Imm_" + genRandomName();

		return reciprocal(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies reciprocal (1/x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code reciprocalI(String a)}<br>
	 * Full parameter: {@code reciprocalI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 reciprocal (1/x)
	 */
	public Tensor reciprocalI(Tensor a) {
		if (a == null) System.err.println("[ERROR][RECIPROCAL][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).reciprocal(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code sin(String a, String out)}<br>
	 * Full parameter: {@code sin(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code sin(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies sine (sin x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code sin(String a, String out)}<br>
	 * Full parameter: {@code sin(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies sine (sin x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code sin(String a, String out)}<br>
	 * Full parameter: {@code sin(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sin(Tensor a) {
		if (a == null) System.err.println("[ERROR][SIN][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sin(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code sin(String a, String out)}<br>
	 * Full parameter: {@code sin(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sin(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][SIN][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sin(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code sinI(String a)}<br>
	 * Full parameter: {@code sinI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 sine (sin x)
	 */
	public Tensor sinI() {
		String oName = "Imm_" + genRandomName();

		return sin("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code sinI(String a)}<br>
	 * Full parameter: {@code sinI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 sine (sin x)
	 */
	public Tensor sinI(String a) {
		String oName = "Imm_" + genRandomName();

		return sin(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sine (sin x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code sinI(String a)}<br>
	 * Full parameter: {@code sinI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 sine (sin x)
	 */
	public Tensor sinI(Tensor a) {
		if (a == null) System.err.println("[ERROR][SIN][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).sin(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code cos(String a, String out)}<br>
	 * Full parameter: {@code cos(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code cos(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies cosine (cos x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code cos(String a, String out)}<br>
	 * Full parameter: {@code cos(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies cosine (cos x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code cos(String a, String out)}<br>
	 * Full parameter: {@code cos(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge cos(Tensor a) {
		if (a == null) System.err.println("[ERROR][COS][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).cos(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code cos(String a, String out)}<br>
	 * Full parameter: {@code cos(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge cos(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][COS][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).cos(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code cosI(String a)}<br>
	 * Full parameter: {@code cosI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 cosine (cos x)
	 */
	public Tensor cosI() {
		String oName = "Imm_" + genRandomName();

		return cos("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code cosI(String a)}<br>
	 * Full parameter: {@code cosI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 cosine (cos x)
	 */
	public Tensor cosI(String a) {
		String oName = "Imm_" + genRandomName();

		return cos(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies cosine (cos x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code cosI(String a)}<br>
	 * Full parameter: {@code cosI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 cosine (cos x)
	 */
	public Tensor cosI(Tensor a) {
		if (a == null) System.err.println("[ERROR][COS][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).cos(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code tan(String a, String out)}<br>
	 * Full parameter: {@code tan(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code tan(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies tangent (tan x) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code tan(String a, String out)}<br>
	 * Full parameter: {@code tan(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies tangent (tan x) to the given Tensor.
	 * <p>
	 * Full parameter: {@code tan(String a, String out)}<br>
	 * Full parameter: {@code tan(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge tan(Tensor a) {
		if (a == null) System.err.println("[ERROR][TAN][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).tan(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code tan(String a, String out)}<br>
	 * Full parameter: {@code tan(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge tan(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][TAN][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).tan(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code tanI(String a)}<br>
	 * Full parameter: {@code tanI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 tangent (tan x)
	 */
	public Tensor tanI() {
		String oName = "Imm_" + genRandomName();

		return tan("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code tanI(String a)}<br>
	 * Full parameter: {@code tanI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 tangent (tan x)
	 */
	public Tensor tanI(String a) {
		String oName = "Imm_" + genRandomName();

		return tan(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies tangent (tan x) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code tanI(String a)}<br>
	 * Full parameter: {@code tanI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 tangent (tan x)
	 */
	public Tensor tanI(Tensor a) {
		if (a == null) System.err.println("[ERROR][TAN][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).tan(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies step function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * Full parameter: {@code step(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies step function to the specified tensor.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * Full parameter: {@code step(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies step function to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * Full parameter: {@code step(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies step function to the given Tensor.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * Full parameter: {@code step(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge step(Tensor a) {
		if (a == null) System.err.println("[ERROR][STEP][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).step(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies step function to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code step(String a, String out)}<br>
	 * Full parameter: {@code step(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge step(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][STEP][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).step(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies step function to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code stepI(String a)}<br>
	 * Full parameter: {@code stepI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 step function
	 */
	public Tensor stepI() {
		String oName = "Imm_" + genRandomName();

		return step("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies step function to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code stepI(String a)}<br>
	 * Full parameter: {@code stepI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 step function
	 */
	public Tensor stepI(String a) {
		String oName = "Imm_" + genRandomName();

		return step(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies step function to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code stepI(String a)}<br>
	 * Full parameter: {@code stepI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 step function
	 */
	public Tensor stepI(Tensor a) {
		if (a == null) System.err.println("[ERROR][STEP][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).step(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code sigmoid(String a, String out)}<br>
	 * Full parameter: {@code sigmoid(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code sigmoid(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Full parameter: {@code sigmoid(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies sigmoid function to the given Tensor.
	 * <p>
	 * Full parameter: {@code sigmoid(String a, String out)}<br>
	 * Full parameter: {@code sigmoid(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sigmoid(Tensor a) {
		if (a == null) System.err.println("[ERROR][SIGMOID][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sigmoid(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code sigmoid(String a, String out)}<br>
	 * Full parameter: {@code sigmoid(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sigmoid(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][SIGMOID][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sigmoid(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code sigmoidI(String a)}<br>
	 * Full parameter: {@code sigmoidI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 sigmoid function
	 */
	public Tensor sigmoidI() {
		String oName = "Imm_" + genRandomName();

		return sigmoid("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code sigmoidI(String a)}<br>
	 * Full parameter: {@code sigmoidI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 sigmoid function
	 */
	public Tensor sigmoidI(String a) {
		String oName = "Imm_" + genRandomName();

		return sigmoid(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies sigmoid function to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code sigmoidI(String a)}<br>
	 * Full parameter: {@code sigmoidI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 sigmoid function
	 */
	public Tensor sigmoidI(Tensor a) {
		if (a == null) System.err.println("[ERROR][SIGMOID][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).sigmoid(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies hyperbolic tangent to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * Full parameter: {@code tanh(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies hyperbolic tangent to the specified tensor.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * Full parameter: {@code tanh(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies hyperbolic tangent to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * Full parameter: {@code tanh(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies hyperbolic tangent to the given Tensor.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * Full parameter: {@code tanh(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge tanh(Tensor a) {
		if (a == null) System.err.println("[ERROR][TANH][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).tanh(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies hyperbolic tangent to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code tanh(String a, String out)}<br>
	 * Full parameter: {@code tanh(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge tanh(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][TANH][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).tanh(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies hyperbolic tangent to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code tanhI(String a)}<br>
	 * Full parameter: {@code tanhI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 hyperbolic tangent
	 */
	public Tensor tanhI() {
		String oName = "Imm_" + genRandomName();

		return tanh("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies hyperbolic tangent to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code tanhI(String a)}<br>
	 * Full parameter: {@code tanhI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 hyperbolic tangent
	 */
	public Tensor tanhI(String a) {
		String oName = "Imm_" + genRandomName();

		return tanh(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies hyperbolic tangent to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code tanhI(String a)}<br>
	 * Full parameter: {@code tanhI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 hyperbolic tangent
	 */
	public Tensor tanhI(Tensor a) {
		if (a == null) System.err.println("[ERROR][TANH][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).tanh(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU activation to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * Full parameter: {@code relu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies ReLU activation to the specified tensor.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * Full parameter: {@code relu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies ReLU activation to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * Full parameter: {@code relu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies ReLU activation to the given Tensor.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * Full parameter: {@code relu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge relu(Tensor a) {
		if (a == null) System.err.println("[ERROR][RELU][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).relu(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU activation to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code relu(String a, String out)}<br>
	 * Full parameter: {@code relu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge relu(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][RELU][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).relu(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU activation to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code reluI(String a)}<br>
	 * Full parameter: {@code reluI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 ReLU activation
	 */
	public Tensor reluI() {
		String oName = "Imm_" + genRandomName();

		return relu("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU activation to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code reluI(String a)}<br>
	 * Full parameter: {@code reluI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 ReLU activation
	 */
	public Tensor reluI(String a) {
		String oName = "Imm_" + genRandomName();

		return relu(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ReLU activation to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code reluI(String a)}<br>
	 * Full parameter: {@code reluI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 ReLU activation
	 */
	public Tensor reluI(Tensor a) {
		if (a == null) System.err.println("[ERROR][RELU][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).relu(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code leakrelu(String a, String out)}<br>
	 * Full parameter: {@code leakrelu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code leakrelu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Full parameter: {@code leakrelu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies Leaky ReLU activation to the given Tensor.
	 * <p>
	 * Full parameter: {@code leakrelu(String a, String out)}<br>
	 * Full parameter: {@code leakrelu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge leakrelu(Tensor a) {
		if (a == null) System.err.println("[ERROR][LEAKRELU][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).leakrelu(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code leakrelu(String a, String out)}<br>
	 * Full parameter: {@code leakrelu(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge leakrelu(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][LEAKRELU][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).leakrelu(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code leakreluI(String a)}<br>
	 * Full parameter: {@code leakreluI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 Leaky ReLU activation
	 */
	public Tensor leakreluI() {
		String oName = "Imm_" + genRandomName();

		return leakrelu("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code leakreluI(String a)}<br>
	 * Full parameter: {@code leakreluI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 Leaky ReLU activation
	 */
	public Tensor leakreluI(String a) {
		String oName = "Imm_" + genRandomName();

		return leakrelu(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies Leaky ReLU activation to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code leakreluI(String a)}<br>
	 * Full parameter: {@code leakreluI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 Leaky ReLU activation
	 */
	public Tensor leakreluI(Tensor a) {
		if (a == null) System.err.println("[ERROR][LEAKRELU][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).leakrelu(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies softplus function to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * Full parameter: {@code softplus(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies softplus function to the specified tensor.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * Full parameter: {@code softplus(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies softplus function to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * Full parameter: {@code softplus(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies softplus function to the given Tensor.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * Full parameter: {@code softplus(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge softplus(Tensor a) {
		if (a == null) System.err.println("[ERROR][SOFTPLUS][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).softplus(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies softplus function to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code softplus(String a, String out)}<br>
	 * Full parameter: {@code softplus(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge softplus(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][SOFTPLUS][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).softplus(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies softplus function to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code softplusI(String a)}<br>
	 * Full parameter: {@code softplusI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 softplus function
	 */
	public Tensor softplusI() {
		String oName = "Imm_" + genRandomName();

		return softplus("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies softplus function to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code softplusI(String a)}<br>
	 * Full parameter: {@code softplusI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 softplus function
	 */
	public Tensor softplusI(String a) {
		String oName = "Imm_" + genRandomName();

		return softplus(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies softplus function to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code softplusI(String a)}<br>
	 * Full parameter: {@code softplusI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 softplus function
	 */
	public Tensor softplusI(Tensor a) {
		if (a == null) System.err.println("[ERROR][SOFTPLUS][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).softplus(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential (eˣ) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * Full parameter: {@code exp(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies exponential (eˣ) to the specified tensor.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * Full parameter: {@code exp(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies exponential (eˣ) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * Full parameter: {@code exp(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge exp(String a, String out) {
		if (!CuBridgeJNI.exp(a, out))
			System.err.println("[ERROR][EXP][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential (eˣ) to the given Tensor.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * Full parameter: {@code exp(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge exp(Tensor a) {
		if (a == null) System.err.println("[ERROR][EXP][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).exp(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential (eˣ) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code exp(String a, String out)}<br>
	 * Full parameter: {@code exp(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge exp(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][EXP][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).exp(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential (eˣ) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code expI(String a)}<br>
	 * Full parameter: {@code expI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 exponential (eˣ)
	 */
	public Tensor expI() {
		String oName = "Imm_" + genRandomName();

		return exp("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential (eˣ) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code expI(String a)}<br>
	 * Full parameter: {@code expI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 exponential (eˣ)
	 */
	public Tensor expI(String a) {
		String oName = "Imm_" + genRandomName();

		return exp(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies exponential (eˣ) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code expI(String a)}<br>
	 * Full parameter: {@code expI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 exponential (eˣ)
	 */
	public Tensor expI(Tensor a) {
		if (a == null) System.err.println("[ERROR][EXP][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).exp(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code deg2rad(String a, String out)}<br>
	 * Full parameter: {@code deg2rad(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the specified tensor.
	 * <p>
	 * Full parameter: {@code deg2rad(String a, String out)}<br>
	 * Full parameter: {@code deg2rad(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge deg2rad(String a) {
		if (!CuBridgeJNI.deg2rad(a, genRandomName()))
			System.err.println("[ERROR][DEG2RAD][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code deg2rad(String a, String out)}<br>
	 * Full parameter: {@code deg2rad(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge deg2rad(String a, String out) {
		if (!CuBridgeJNI.deg2rad(a, out))
			System.err.println("[ERROR][DEG2RAD][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the given Tensor.
	 * <p>
	 * Full parameter: {@code deg2rad(String a, String out)}<br>
	 * Full parameter: {@code deg2rad(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge deg2rad(Tensor a) {
		if (a == null) System.err.println("[ERROR][DEG2RAD][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).deg2rad(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code deg2rad(String a, String out)}<br>
	 * Full parameter: {@code deg2rad(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge deg2rad(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][DEG2RAD][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).deg2rad(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code deg2radI(String a)}<br>
	 * Full parameter: {@code deg2radI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 degree to radian conversion
	 */
	public Tensor deg2radI() {
		String oName = "Imm_" + genRandomName();

		return deg2rad("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code deg2radI(String a)}<br>
	 * Full parameter: {@code deg2radI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 degree to radian conversion
	 */
	public Tensor deg2radI(String a) {
		String oName = "Imm_" + genRandomName();

		return deg2rad(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies degree to radian conversion to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code deg2radI(String a)}<br>
	 * Full parameter: {@code deg2radI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 degree to radian conversion
	 */
	public Tensor deg2radI(Tensor a) {
		if (a == null) System.err.println("[ERROR][DEG2RAD][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).deg2rad(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code rad2deg(String a, String out)}<br>
	 * Full parameter: {@code rad2deg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the specified tensor.
	 * <p>
	 * Full parameter: {@code rad2deg(String a, String out)}<br>
	 * Full parameter: {@code rad2deg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge rad2deg(String a) {
		if (!CuBridgeJNI.rad2deg(a, genRandomName()))
			System.err.println("[ERROR][RAD2DEG][Cannot Execute][Tensor " + a + ", -]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code rad2deg(String a, String out)}<br>
	 * Full parameter: {@code rad2deg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
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
	 * Applies radian to degree conversion to the given Tensor.
	 * <p>
	 * Full parameter: {@code rad2deg(String a, String out)}<br>
	 * Full parameter: {@code rad2deg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge rad2deg(Tensor a) {
		if (a == null) System.err.println("[ERROR][RAD2DEG][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).rad2deg(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code rad2deg(String a, String out)}<br>
	 * Full parameter: {@code rad2deg(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge rad2deg(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][RAD2DEG][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).rad2deg(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code rad2degI(String a)}<br>
	 * Full parameter: {@code rad2degI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 radian to degree conversion
	 */
	public Tensor rad2degI() {
		String oName = "Imm_" + genRandomName();

		return rad2deg("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code rad2degI(String a)}<br>
	 * Full parameter: {@code rad2degI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 radian to degree conversion
	 */
	public Tensor rad2degI(String a) {
		String oName = "Imm_" + genRandomName();

		return rad2deg(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies radian to degree conversion to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code rad2degI(String a)}<br>
	 * Full parameter: {@code rad2degI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 radian to degree conversion
	 */
	public Tensor rad2degI(Tensor a) {
		if (a == null) System.err.println("[ERROR][RAD2DEG][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).rad2deg(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code round(String a, String out)}<br>
	 * Full parameter: {@code round(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code round(String a, String out)}<br>
	 * Full parameter: {@code round(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Full parameter: {@code round(String a, String out)}<br>
	 * Full parameter: {@code round(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
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
	 * Applies rounding to the given Tensor.
	 * <p>
	 * Full parameter: {@code round(String a, String out)}<br>
	 * Full parameter: {@code round(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge round(Tensor a) {
		if (a == null) System.err.println("[ERROR][ROUND][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).round(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code round(String a, String out)}<br>
	 * Full parameter: {@code round(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge round(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][ROUND][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).round(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code roundI(String a)}<br>
	 * Full parameter: {@code roundI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 rounding
	 */
	public Tensor roundI() {
		String oName = "Imm_" + genRandomName();

		return round("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code roundI(String a)}<br>
	 * Full parameter: {@code roundI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 rounding
	 */
	public Tensor roundI(String a) {
		String oName = "Imm_" + genRandomName();

		return round(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies rounding to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code roundI(String a)}<br>
	 * Full parameter: {@code roundI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 rounding
	 */
	public Tensor roundI(Tensor a) {
		if (a == null) System.err.println("[ERROR][ROUND][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).round(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling (⌈x⌉) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code ceil(String a, String out)}<br>
	 * Full parameter: {@code ceil(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies ceiling (⌈x⌉) to the specified tensor.
	 * <p>
	 * Full parameter: {@code ceil(String a, String out)}<br>
	 * Full parameter: {@code ceil(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies ceiling (⌈x⌉) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code ceil(String a, String out)}<br>
	 * Full parameter: {@code ceil(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
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
	 * Applies ceiling (⌈x⌉) to the given Tensor.
	 * <p>
	 * Full parameter: {@code ceil(String a, String out)}<br>
	 * Full parameter: {@code ceil(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ceil(Tensor a) {
		if (a == null) System.err.println("[ERROR][CEIL][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ceil(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling (⌈x⌉) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code ceil(String a, String out)}<br>
	 * Full parameter: {@code ceil(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ceil(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][CEIL][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ceil(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling (⌈x⌉) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code ceilI(String a)}<br>
	 * Full parameter: {@code ceilI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 ceiling (⌈x⌉)
	 */
	public Tensor ceilI() {
		String oName = "Imm_" + genRandomName();

		return ceil("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling (⌈x⌉) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code ceilI(String a)}<br>
	 * Full parameter: {@code ceilI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 ceiling (⌈x⌉)
	 */
	public Tensor ceilI(String a) {
		String oName = "Imm_" + genRandomName();

		return ceil(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies ceiling (⌈x⌉) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code ceilI(String a)}<br>
	 * Full parameter: {@code ceilI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 ceiling (⌈x⌉)
	 */
	public Tensor ceilI(Tensor a) {
		if (a == null) System.err.println("[ERROR][CEIL][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).ceil(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor (⌊x⌋) to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code floor(String a, String out)}<br>
	 * Full parameter: {@code floor(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Applies floor (⌊x⌋) to the specified tensor.
	 * <p>
	 * Full parameter: {@code floor(String a, String out)}<br>
	 * Full parameter: {@code floor(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
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
	 * Applies floor (⌊x⌋) to the specified tensor and stores the result.
	 * <p>
	 * Full parameter: {@code floor(String a, String out)}<br>
	 * Full parameter: {@code floor(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
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
	 * Applies floor (⌊x⌋) to the given Tensor.
	 * <p>
	 * Full parameter: {@code floor(String a, String out)}<br>
	 * Full parameter: {@code floor(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge floor(Tensor a) {
		if (a == null) System.err.println("[ERROR][FLOOR][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).floor(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor (⌊x⌋) to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code floor(String a, String out)}<br>
	 * Full parameter: {@code floor(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge floor(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][FLOOR][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).floor(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor (⌊x⌋) to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code floorI(String a)}<br>
	 * Full parameter: {@code floorI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 floor (⌊x⌋)
	 */
	public Tensor floorI() {
		String oName = "Imm_" + genRandomName();

		return floor("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor (⌊x⌋) to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code floorI(String a)}<br>
	 * Full parameter: {@code floorI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 floor (⌊x⌋)
	 */
	public Tensor floorI(String a) {
		String oName = "Imm_" + genRandomName();

		return floor(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies floor (⌊x⌋) to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code floorI(String a)}<br>
	 * Full parameter: {@code floorI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 floor (⌊x⌋)
	 */
	public Tensor floorI(Tensor a) {
		if (a == null) System.err.println("[ERROR][FLOOR][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).floor(aName, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the most recent tensor in the queue.
	 * <p>
	 * Full parameter: {@code not(String a, String out)}<br>
	 * Full parameter: {@code not(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>Operation is applied element-wise.</li>
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
	 * Full parameter: {@code not(String a, String out)}<br>
	 * Full parameter: {@code not(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input tensor is identified by name.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
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
	 * Full parameter: {@code not(String a, String out)}<br>
	 * Full parameter: {@code not(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>Input and output tensors are both specified by name.</li>
	 * <li>Operation is applied element-wise.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a   the name of the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge not(String a, String out) {
		if (!CuBridgeJNI.not(a, out))
			System.err.println("[ERROR][NOT][Cannot Execute][Tensor " + a + ", " + out + "]");

		return instance;
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the given Tensor.
	 * <p>
	 * Full parameter: {@code not(String a, String out)}<br>
	 * Full parameter: {@code not(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is stored under an auto-generated name.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge not(Tensor a) {
		if (a == null) System.err.println("[ERROR][NOT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).not(aName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the given Tensor and stores the result under the given name.
	 * <p>
	 * Full parameter: {@code not(String a, String out)}<br>
	 * Full parameter: {@code not(Tensor a, String out)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The output name is explicitly specified.</li>
	 * <li>The input is pushed into the queue temporarily for processing.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @param out the output name to store result
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge not(Tensor a, String out) {
		if (a == null) System.err.println("[ERROR][NOT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).not(aName, out);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the most recent tensor in the queue and returns the result immediately.
	 * <p>
	 * Full parameter: {@code notI(String a)}<br>
	 * Full parameter: {@code notI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The top tensor is selected automatically.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return resulting Tensor after applying
	 * @since v1.3 logical NOT operation
	 */
	public Tensor notI() {
		String oName = "Imm_" + genRandomName();

		return not("", oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the specified tensor and returns the result immediately.
	 * <p>
	 * Full parameter: {@code notI(String a)}<br>
	 * Full parameter: {@code notI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input tensor is identified by name.</li>
	 * <li>The result is returned directly as a new Tensor.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 logical NOT operation
	 */
	public Tensor notI(String a) {
		String oName = "Imm_" + genRandomName();

		return not(a, oName).get(oName);
	}

	/**
	 * Unary Operation (Axis-Independent)
	 *
	 * Applies logical NOT operation to the given Tensor and returns the result directly.
	 * <p>
	 * Full parameter: {@code notI(String a)}<br>
	 * Full parameter: {@code notI(Tensor a)}<br>
	 * This version:
	 * <ul>
	 * <li>The input is provided as a Tensor object.</li>
	 * <li>The result is returned immediately.</li>
	 * <li>Temporary internal name is used; no effect on queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input Tensor
	 * @return resulting Tensor after applying
	 * @since v1.3 logical NOT operation
	 */
	public Tensor notI(Tensor a) {
		if (a == null) System.err.println("[ERROR][NOT][Null Tensor Input]");

		String aName = genRandomName();
		String oName = genRandomName();

		return put(a, aName).not(aName, oName).get(oName);
	}

	// 이항
	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(Tensor a) {
		if (a == null) System.err.println("[ERROR][ADD][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).add(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(Tensor a, String b) {
		return add(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(String a, Tensor b) {
		return add(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(Tensor a, Tensor b) {
		return add(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][ADD][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).add(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][ADD][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).add(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code add(String a, String b, String out)}<br>
	 * {@code add(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge add(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][ADD][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][ADD][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).add(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise addition (a + b).
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise addition
	 */
	public Tensor addI() {
		String oName = "Imm_" + genRandomName();
		return add("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Adds a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after addition
	 * @since v1.3
	 */
	public Tensor addI(String a) {
		String oName = "Imm_" + genRandomName();
		return add(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Adds two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after addition
	 * @since v1.3
	 */
	public Tensor addI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return add(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Adds a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after addition
	 * @since v1.3
	 */
	public Tensor addI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return add(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Adds a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after addition
	 * @since v1.3
	 */
	public Tensor addI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return add(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Adds a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after addition
	 * @since v1.3
	 */
	public Tensor addI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return add(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Adds two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code addI(String a, String b)}<br>
	 * {@code addI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after addition
	 * @since v1.3
	 */
	public Tensor addI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return add(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(Tensor a) {
		if (a == null) System.err.println("[ERROR][SUB][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sub(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(Tensor a, String b) {
		return sub(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(String a, Tensor b) {
		return sub(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(Tensor a, Tensor b) {
		return sub(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][SUB][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).sub(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][SUB][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).sub(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code sub(String a, String b, String out)}<br>
	 * {@code sub(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge sub(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][SUB][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][SUB][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).sub(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise subition (a - b).
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise subition
	 */
	public Tensor subI() {
		String oName = "Imm_" + genRandomName();
		return sub("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Subtracts a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after subition
	 * @since v1.3
	 */
	public Tensor subI(String a) {
		String oName = "Imm_" + genRandomName();
		return sub(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Subtracts two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after subition
	 * @since v1.3
	 */
	public Tensor subI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return sub(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Subtracts a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after subition
	 * @since v1.3
	 */
	public Tensor subI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return sub(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Subtracts a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after subition
	 * @since v1.3
	 */
	public Tensor subI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return sub(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Subtracts a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after subition
	 * @since v1.3
	 */
	public Tensor subI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return sub(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Subtracts two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code subI(String a, String b)}<br>
	 * {@code subI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after subition
	 * @since v1.3
	 */
	public Tensor subI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return sub(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(Tensor a) {
		if (a == null) System.err.println("[ERROR][MUL][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).mul(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(Tensor a, String b) {
		return mul(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(String a, Tensor b) {
		return mul(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(Tensor a, Tensor b) {
		return mul(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][MUL][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).mul(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][MUL][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).mul(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mul(String a, String b, String out)}<br>
	 * {@code mul(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mul(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][MUL][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][MUL][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).mul(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise mulition (a * b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise mulition
	 */
	public Tensor mulI() {
		String oName = "Imm_" + genRandomName();
		return mul("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Multiplies a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after mulition
	 * @since v1.3
	 */
	public Tensor mulI(String a) {
		String oName = "Imm_" + genRandomName();
		return mul(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Multiplies two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after mulition
	 * @since v1.3
	 */
	public Tensor mulI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return mul(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Multiplies a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after mulition
	 * @since v1.3
	 */
	public Tensor mulI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return mul(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Multiplies a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after mulition
	 * @since v1.3
	 */
	public Tensor mulI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return mul(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Multiplies a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after mulition
	 * @since v1.3
	 */
	public Tensor mulI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return mul(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Multiplies two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code mulI(String a, String b)}<br>
	 * {@code mulI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after mulition
	 * @since v1.3
	 */
	public Tensor mulI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return mul(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(Tensor a) {
		if (a == null) System.err.println("[ERROR][DIV][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).div(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(Tensor a, String b) {
		return div(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(String a, Tensor b) {
		return div(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(Tensor a, Tensor b) {
		return div(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][DIV][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).div(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][DIV][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).div(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code div(String a, String b, String out)}<br>
	 * {@code div(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge div(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][DIV][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][DIV][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).div(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise divition (a / b).
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise divition
	 */
	public Tensor divI() {
		String oName = "Imm_" + genRandomName();
		return div("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Divides a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after divition
	 * @since v1.3
	 */
	public Tensor divI(String a) {
		String oName = "Imm_" + genRandomName();
		return div(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Divides two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after divition
	 * @since v1.3
	 */
	public Tensor divI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return div(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Divides a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after divition
	 * @since v1.3
	 */
	public Tensor divI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return div(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Divides a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after divition
	 * @since v1.3
	 */
	public Tensor divI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return div(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Divides a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after divition
	 * @since v1.3
	 */
	public Tensor divI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return div(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Divides two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code divI(String a, String b)}<br>
	 * {@code divI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after divition
	 * @since v1.3
	 */
	public Tensor divI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return div(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
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
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(Tensor a) {
		if (a == null) System.err.println("[ERROR][POW][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).pow(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(Tensor a, String b) {
		return pow(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(String a, Tensor b) {
		return pow(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(Tensor a, Tensor b) {
		return pow(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][POW][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).pow(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][POW][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).pow(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code pow(String a, String b, String out)}<br>
	 * {@code pow(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge pow(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][POW][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][POW][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).pow(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise powition (a ^ b).
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise powition
	 */
	public Tensor powI() {
		String oName = "Imm_" + genRandomName();
		return pow("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Raises a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after powition
	 * @since v1.3
	 */
	public Tensor powI(String a) {
		String oName = "Imm_" + genRandomName();
		return pow(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Raises two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after powition
	 * @since v1.3
	 */
	public Tensor powI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return pow(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Raises a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after powition
	 * @since v1.3
	 */
	public Tensor powI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return pow(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Raises a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after powition
	 * @since v1.3
	 */
	public Tensor powI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return pow(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Raises a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after powition
	 * @since v1.3
	 */
	public Tensor powI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return pow(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Raises two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code powI(String a, String b)}<br>
	 * {@code powI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after powition
	 * @since v1.3
	 */
	public Tensor powI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return pow(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
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
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(Tensor a) {
		if (a == null) System.err.println("[ERROR][MOD][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).mod(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(Tensor a, String b) {
		return mod(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(String a, Tensor b) {
		return mod(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(Tensor a, Tensor b) {
		return mod(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][MOD][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).mod(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][MOD][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).mod(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code mod(String a, String b, String out)}<br>
	 * {@code mod(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge mod(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][MOD][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][MOD][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).mod(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise modition (a % b).
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise modition
	 */
	public Tensor modI() {
		String oName = "Imm_" + genRandomName();
		return mod("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies modulo on a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after modition
	 * @since v1.3
	 */
	public Tensor modI(String a) {
		String oName = "Imm_" + genRandomName();
		return mod(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies modulo on two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after modition
	 * @since v1.3
	 */
	public Tensor modI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return mod(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies modulo on a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after modition
	 * @since v1.3
	 */
	public Tensor modI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return mod(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies modulo on a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after modition
	 * @since v1.3
	 */
	public Tensor modI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return mod(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies modulo on a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after modition
	 * @since v1.3
	 */
	public Tensor modI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return mod(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies modulo on two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code modI(String a, String b)}<br>
	 * {@code modI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after modition
	 * @since v1.3
	 */
	public Tensor modI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return mod(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(Tensor a) {
		if (a == null) System.err.println("[ERROR][GT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).gt(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(Tensor a, String b) {
		return gt(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(String a, Tensor b) {
		return gt(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(Tensor a, Tensor b) {
		return gt(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][GT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).gt(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][GT][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).gt(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gt(String a, String b, String out)}<br>
	 * {@code gt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge gt(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][GT][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][GT][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).gt(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise gtition (a > b).
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise gtition
	 */
	public Tensor gtI() {
		String oName = "Imm_" + genRandomName();
		return gt("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after gtition
	 * @since v1.3
	 */
	public Tensor gtI(String a) {
		String oName = "Imm_" + genRandomName();
		return gt(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after gtition
	 * @since v1.3
	 */
	public Tensor gtI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return gt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after gtition
	 * @since v1.3
	 */
	public Tensor gtI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return gt(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after gtition
	 * @since v1.3
	 */
	public Tensor gtI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return gt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after gtition
	 * @since v1.3
	 */
	public Tensor gtI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return gt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code gtI(String a, String b)}<br>
	 * {@code gtI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after gtition
	 * @since v1.3
	 */
	public Tensor gtI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return gt(a, b, oName).get(oName);
	}
	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(Tensor a) {
		if (a == null) System.err.println("[ERROR][LT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).lt(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(Tensor a, String b) {
		return lt(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(String a, Tensor b) {
		return lt(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(Tensor a, Tensor b) {
		return lt(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][LT][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).lt(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][LT][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).lt(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code lt(String a, String b, String out)}<br>
	 * {@code lt(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge lt(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][LT][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][LT][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).lt(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise ltition (a < b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise ltition
	 */
	public Tensor ltI() {
		String oName = "Imm_" + genRandomName();
		return lt("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after ltition
	 * @since v1.3
	 */
	public Tensor ltI(String a) {
		String oName = "Imm_" + genRandomName();
		return lt(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after ltition
	 * @since v1.3
	 */
	public Tensor ltI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return lt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after ltition
	 * @since v1.3
	 */
	public Tensor ltI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return lt(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after ltition
	 * @since v1.3
	 */
	public Tensor ltI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return lt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after ltition
	 * @since v1.3
	 */
	public Tensor ltI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return lt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code ltI(String a, String b)}<br>
	 * {@code ltI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after ltition
	 * @since v1.3
	 */
	public Tensor ltI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return lt(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(Tensor a) {
		if (a == null) System.err.println("[ERROR][GE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ge(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(Tensor a, String b) {
		return ge(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(String a, Tensor b) {
		return ge(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(Tensor a, Tensor b) {
		return ge(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][GE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ge(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][GE][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).ge(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ge(String a, String b, String out)}<br>
	 * {@code ge(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ge(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][GE][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][GE][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).ge(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise geition (a >= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise geition
	 */
	public Tensor geI() {
		String oName = "Imm_" + genRandomName();
		return ge("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after geition
	 * @since v1.3
	 */
	public Tensor geI(String a) {
		String oName = "Imm_" + genRandomName();
		return ge(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after geition
	 * @since v1.3
	 */
	public Tensor geI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return ge(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after geition
	 * @since v1.3
	 */
	public Tensor geI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return ge(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after geition
	 * @since v1.3
	 */
	public Tensor geI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return ge(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after geition
	 * @since v1.3
	 */
	public Tensor geI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return ge(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code geI(String a, String b)}<br>
	 * {@code geI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after geition
	 * @since v1.3
	 */
	public Tensor geI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return ge(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(Tensor a) {
		if (a == null) System.err.println("[ERROR][LE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).le(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(Tensor a, String b) {
		return le(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(String a, Tensor b) {
		return le(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(Tensor a, Tensor b) {
		return le(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][LE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).le(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][LE][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).le(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code le(String a, String b, String out)}<br>
	 * {@code le(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge le(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][LE][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][LE][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).le(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise leition (a <= b).
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise leition
	 */
	public Tensor leI() {
		String oName = "Imm_" + genRandomName();
		return le("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after leition
	 * @since v1.3
	 */
	public Tensor leI(String a) {
		String oName = "Imm_" + genRandomName();
		return le(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after leition
	 * @since v1.3
	 */
	public Tensor leI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return le(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after leition
	 * @since v1.3
	 */
	public Tensor leI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return le(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after leition
	 * @since v1.3
	 */
	public Tensor leI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return le(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after leition
	 * @since v1.3
	 */
	public Tensor leI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return le(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Compares whether two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code leI(String a, String b)}<br>
	 * {@code leI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after leition
	 * @since v1.3
	 */
	public Tensor leI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return le(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(Tensor a) {
		if (a == null) System.err.println("[ERROR][EQ][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).eq(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(Tensor a, String b) {
		return eq(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(String a, Tensor b) {
		return eq(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(Tensor a, Tensor b) {
		return eq(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][EQ][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).eq(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][EQ][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).eq(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eq(String a, String b, String out)}<br>
	 * {@code eq(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge eq(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][EQ][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][EQ][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).eq(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise eqition (a == b).
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise eqition
	 */
	public Tensor eqI() {
		String oName = "Imm_" + genRandomName();
		return eq("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after eqition
	 * @since v1.3
	 */
	public Tensor eqI(String a) {
		String oName = "Imm_" + genRandomName();
		return eq(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after eqition
	 * @since v1.3
	 */
	public Tensor eqI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return eq(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after eqition
	 * @since v1.3
	 */
	public Tensor eqI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return eq(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after eqition
	 * @since v1.3
	 */
	public Tensor eqI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return eq(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after eqition
	 * @since v1.3
	 */
	public Tensor eqI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return eq(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code eqI(String a, String b)}<br>
	 * {@code eqI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after eqition
	 * @since v1.3
	 */
	public Tensor eqI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return eq(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(Tensor a) {
		if (a == null) System.err.println("[ERROR][NE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ne(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(Tensor a, String b) {
		return ne(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(String a, Tensor b) {
		return ne(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(Tensor a, Tensor b) {
		return ne(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][NE][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).ne(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][NE][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).ne(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code ne(String a, String b, String out)}<br>
	 * {@code ne(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge ne(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][NE][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][NE][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).ne(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise neition (a != b).
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise neition
	 */
	public Tensor neI() {
		String oName = "Imm_" + genRandomName();
		return ne("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after neition
	 * @since v1.3
	 */
	public Tensor neI(String a) {
		String oName = "Imm_" + genRandomName();
		return ne(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after neition
	 * @since v1.3
	 */
	public Tensor neI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return ne(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after neition
	 * @since v1.3
	 */
	public Tensor neI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return ne(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after neition
	 * @since v1.3
	 */
	public Tensor neI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return ne(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after neition
	 * @since v1.3
	 */
	public Tensor neI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return ne(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Checks whether two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code neI(String a, String b)}<br>
	 * {@code neI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after neition
	 * @since v1.3
	 */
	public Tensor neI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return ne(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
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
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(Tensor a) {
		if (a == null) System.err.println("[ERROR][AND][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).and(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(Tensor a, String b) {
		return and(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(String a, Tensor b) {
		return and(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(Tensor a, Tensor b) {
		return and(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][AND][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).and(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][AND][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).and(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code and(String a, String b, String out)}<br>
	 * {@code and(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge and(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][AND][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][AND][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).and(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise andition (a && b).
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise andition
	 */
	public Tensor andI() {
		String oName = "Imm_" + genRandomName();
		return and("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical AND on a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after andition
	 * @since v1.3
	 */
	public Tensor andI(String a) {
		String oName = "Imm_" + genRandomName();
		return and(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical AND on two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after andition
	 * @since v1.3
	 */
	public Tensor andI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return and(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical AND on a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after andition
	 * @since v1.3
	 */
	public Tensor andI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return and(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical AND on a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after andition
	 * @since v1.3
	 */
	public Tensor andI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return and(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical AND on a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after andition
	 * @since v1.3
	 */
	public Tensor andI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return and(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical AND on two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code andI(String a, String b)}<br>
	 * {@code andI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after andition
	 * @since v1.3
	 */
	public Tensor andI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return and(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
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
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge or(String a, String b, String out) {
		if (!CuBridgeJNI.or(a, b, out))
			System.err.println("[ERROR][OR][Cannot Execute][Tensor " + a + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>One tensor is specified; the other is taken from the queue.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(Tensor a) {
		if (a == null) System.err.println("[ERROR][OR][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).or(aName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(Tensor a, String b) {
		return or(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(String a, Tensor b) {
		return or(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input tensors are specified by name or object.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(Tensor a, Tensor b) {
		return or(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the name of the second input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(Tensor a, String b, String out) {
		if (a == null) System.err.println("[ERROR][OR][Null Tensor Input]");

		String aName = genRandomName();

		return put(a, aName).or(aName, b, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the name of the first input tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(String a, Tensor b, String out) {
		if (b == null) System.err.println("[ERROR][OR][Null Tensor Input]");

		String bName = genRandomName();

		return put(b, bName).or(a, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code or(String a, String b, String out)}<br>
	 * {@code or(Tensor a, Tensor b, String out)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The input and output tensors are all specified explicitly.</li>
	 * </ul>
	 * </p>
	 * @param a the first input Tensor
	 * @param b the second input Tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 * @since v1.3
	 */
	public CuBridge or(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][OR][Null Tensor Input]");
		if (b == null) System.err.println("[ERROR][OR][Null Tensor Input]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).or(aName, bName, out);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Applies element-wise orition (a || b).
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The top two tensors in the queue are used as inputs.</li>
	 * </ul>
	 * </p>
	 * @return resulting Tensor after applying element-wise orition
	 */
	public Tensor orI() {
		String oName = "Imm_" + genRandomName();
		return or("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical OR on a named tensor and the top tensor from the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is specified by name.</li>
	 * <li>The second operand is taken from the queue.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand tensor name
	 * @return resulting Tensor after orition
	 * @since v1.3
	 */
	public Tensor orI(String a) {
		String oName = "Imm_" + genRandomName();
		return or(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical OR on two named tensors and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are identified by name.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand name
	 * @return resulting Tensor after orition
	 * @since v1.3
	 */
	public Tensor orI(String a, String b) {
		String oName = "Imm_" + genRandomName();
		return or(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical OR on a Tensor object and the top tensor in the queue, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object.</li>
	 * <li>The second is taken from the queue.</li>
	 * <li>The Tensor is temporarily registered.</li>
	 * <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @return resulting Tensor after orition
	 * @since v1.3
	 */
	public Tensor orI(Tensor a) {
		String oName = "Imm_" + genRandomName();
		return or(a, "", oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical OR on a Tensor object and a named tensor, returning the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is a Tensor object, pushed temporarily.</li>
	 * <li>The second is identified by name.</li>
	 * <li>The result is returned directly.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand name
	 * @return resulting Tensor after orition
	 * @since v1.3
	 */
	public Tensor orI(Tensor a, String b) {
		String oName = "Imm_" + genRandomName();
		return or(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical OR on a named tensor and a Tensor object, returning the result.
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>The first operand is identified by name.</li>
	 * <li>The second operand is a Tensor object, pushed temporarily.</li>
	 * <li>The result is returned immediately.</li>
	 * </ul>
	 * @param a the first operand name
	 * @param b the second operand Tensor
	 * @return resulting Tensor after orition
	 * @since v1.3
	 */
	public Tensor orI(String a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return or(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Axis-Independent)
	 *
	 * Performs logical OR on two Tensor objects and returns the result directly.
	 * <p>
	 * Full parameter:<br>
	 * {@code orI(String a, String b)}<br>
	 * {@code orI(Tensor a, Tensor b)}<br>
	 * <p>
	 * This version:
	 * <ul>
	 * <li>Both operands are given as Tensor objects.</li>
	 * <li>They are pushed temporarily and do not affect queue state.</li>
	 * <li>Result is returned as a new Tensor.</li>
	 * </ul>
	 * @param a the first operand Tensor
	 * @param b the second operand Tensor
	 * @return resulting Tensor after orition
	 * @since v1.3
	 */
	public Tensor orI(Tensor a, Tensor b) {
		String oName = "Imm_" + genRandomName();
		return or(a, b, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing all elements across all axes.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs reduction by summing elements from the specified axis downward.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis axis to reduce from
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
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
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
	 * Performs reduction by summing elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by summing all elements across all axes of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs reduction by summing elements from the specified axis of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by summing all elements across all axes of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(Tensor a) {
		return sum(a, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(Tensor a, int axis) {
		return sum(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing all elements across all axes of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(Tensor a, String out) {
		return sum(a, out, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by summing elements from the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sum(String a, String out, int axis)}<br>
	 * {@code sum(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge sum(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][SUM][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).sum(aName, out, axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by summing elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sumI(String a, int axis)}<br>
	 * {@code sumI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input and performs full-axis reduction.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param — input specification
	 * @return result tensor
	 */
	public Tensor sumI() {
		String oName = genRandomName();
		return sum("", oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by summing elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sumI(String a, int axis)}<br>
	 * {@code sumI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue and reduces from the specified axis.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis input specification
	 * @return result tensor
	 */
	public Tensor sumI(int axis) {
		String oName = genRandomName();
		return sum("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by summing elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sumI(String a, int axis)}<br>
	 * {@code sumI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the specified tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor sumI(String a) {
		String oName = genRandomName();
		return sum(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by summing elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sumI(String a, int axis)}<br>
	 * {@code sumI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the named tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor sumI(String a, int axis) {
		String oName = genRandomName();
		return sum(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by summing elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sumI(String a, int axis)}<br>
	 * {@code sumI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor sumI(Tensor a) {
		String oName = genRandomName();
		return sum(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by summing elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code sumI(String a, int axis)}<br>
	 * {@code sumI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor sumI(Tensor a, int axis) {
		String oName = genRandomName();
		return sum(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by meanming all elements across all axes.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs reduction by meanming elements from the specified axis downward.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis axis to reduce from
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
	 * Performs reduction by meanming all elements across all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
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
	 * Performs reduction by meanming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by meanming all elements across all axes of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs reduction by meanming elements from the specified axis of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by meanming all elements across all axes of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(Tensor a) {
		return mean(a, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by meanming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(Tensor a, int axis) {
		return mean(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by meanming all elements across all axes of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(Tensor a, String out) {
		return mean(a, out, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by meanming elements from the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mean(String a, String out, int axis)}<br>
	 * {@code mean(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mean(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][MEAN][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).mean(aName, out, axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by meanming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code meanI(String a, int axis)}<br>
	 * {@code meanI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input and performs full-axis reduction.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param — input specification
	 * @return result tensor
	 */
	public Tensor meanI() {
		String oName = genRandomName();
		return mean("", oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by meanming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code meanI(String a, int axis)}<br>
	 * {@code meanI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue and reduces from the specified axis.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis input specification
	 * @return result tensor
	 */
	public Tensor meanI(int axis) {
		String oName = genRandomName();
		return mean("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by meanming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code meanI(String a, int axis)}<br>
	 * {@code meanI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the specified tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor meanI(String a) {
		String oName = genRandomName();
		return mean(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by meanming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code meanI(String a, int axis)}<br>
	 * {@code meanI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the named tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor meanI(String a, int axis) {
		String oName = genRandomName();
		return mean(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by meanming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code meanI(String a, int axis)}<br>
	 * {@code meanI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor meanI(Tensor a) {
		String oName = genRandomName();
		return mean(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by meanming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code meanI(String a, int axis)}<br>
	 * {@code meanI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor meanI(Tensor a, int axis) {
		String oName = genRandomName();
		return mean(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by varming all elements across all axes.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs reduction by varming elements from the specified axis downward.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis axis to reduce from
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
	 * Performs reduction by varming all elements across all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
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
	 * Performs reduction by varming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by varming all elements across all axes of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs reduction by varming elements from the specified axis of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by varming all elements across all axes of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(Tensor a) {
		return var(a, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by varming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(Tensor a, int axis) {
		return var(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by varming all elements across all axes of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(Tensor a, String out) {
		return var(a, out, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by varming elements from the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code var(String a, String out, int axis)}<br>
	 * {@code var(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge var(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][VAR][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).var(aName, out, axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by varming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code varI(String a, int axis)}<br>
	 * {@code varI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input and performs full-axis reduction.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param — input specification
	 * @return result tensor
	 */
	public Tensor varI() {
		String oName = genRandomName();
		return var("", oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by varming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code varI(String a, int axis)}<br>
	 * {@code varI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue and reduces from the specified axis.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis input specification
	 * @return result tensor
	 */
	public Tensor varI(int axis) {
		String oName = genRandomName();
		return var("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by varming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code varI(String a, int axis)}<br>
	 * {@code varI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the specified tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor varI(String a) {
		String oName = genRandomName();
		return var(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by varming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code varI(String a, int axis)}<br>
	 * {@code varI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the named tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor varI(String a, int axis) {
		String oName = genRandomName();
		return var(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by varming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code varI(String a, int axis)}<br>
	 * {@code varI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor varI(Tensor a) {
		String oName = genRandomName();
		return var(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by varming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code varI(String a, int axis)}<br>
	 * {@code varI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor varI(Tensor a, int axis) {
		String oName = genRandomName();
		return var(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by stdming all elements across all axes.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs reduction by stdming elements from the specified axis downward.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis axis to reduce from
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
	 * Performs reduction by stdming all elements across all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
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
	 * Performs reduction by stdming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by stdming all elements across all axes of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs reduction by stdming elements from the specified axis of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by stdming all elements across all axes of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(Tensor a) {
		return std(a, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by stdming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(Tensor a, int axis) {
		return std(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by stdming all elements across all axes of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(Tensor a, String out) {
		return std(a, out, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by stdming elements from the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code std(String a, String out, int axis)}<br>
	 * {@code std(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge std(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][STD][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).std(aName, out, axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by stdming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code stdI(String a, int axis)}<br>
	 * {@code stdI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input and performs full-axis reduction.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param — input specification
	 * @return result tensor
	 */
	public Tensor stdI() {
		String oName = genRandomName();
		return std("", oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by stdming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code stdI(String a, int axis)}<br>
	 * {@code stdI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue and reduces from the specified axis.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis input specification
	 * @return result tensor
	 */
	public Tensor stdI(int axis) {
		String oName = genRandomName();
		return std("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by stdming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code stdI(String a, int axis)}<br>
	 * {@code stdI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the specified tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor stdI(String a) {
		String oName = genRandomName();
		return std(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by stdming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code stdI(String a, int axis)}<br>
	 * {@code stdI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the named tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor stdI(String a, int axis) {
		String oName = genRandomName();
		return std(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by stdming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code stdI(String a, int axis)}<br>
	 * {@code stdI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor stdI(Tensor a) {
		String oName = genRandomName();
		return std(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by stdming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code stdI(String a, int axis)}<br>
	 * {@code stdI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor stdI(Tensor a, int axis) {
		String oName = genRandomName();
		return std(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by maxming all elements across all axes.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs reduction by maxming elements from the specified axis downward.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis axis to reduce from
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
	 * Performs reduction by maxming all elements across all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
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
	 * Performs reduction by maxming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by maxming all elements across all axes of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs reduction by maxming elements from the specified axis of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by maxming all elements across all axes of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(Tensor a) {
		return max(a, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by maxming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(Tensor a, int axis) {
		return max(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by maxming all elements across all axes of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(Tensor a, String out) {
		return max(a, out, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by maxming elements from the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code max(String a, String out, int axis)}<br>
	 * {@code max(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge max(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][MAX][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).max(aName, out, axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by maxming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code maxI(String a, int axis)}<br>
	 * {@code maxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input and performs full-axis reduction.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param — input specification
	 * @return result tensor
	 */
	public Tensor maxI() {
		String oName = genRandomName();
		return max("", oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by maxming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code maxI(String a, int axis)}<br>
	 * {@code maxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue and reduces from the specified axis.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis input specification
	 * @return result tensor
	 */
	public Tensor maxI(int axis) {
		String oName = genRandomName();
		return max("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by maxming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code maxI(String a, int axis)}<br>
	 * {@code maxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the specified tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor maxI(String a) {
		String oName = genRandomName();
		return max(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by maxming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code maxI(String a, int axis)}<br>
	 * {@code maxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the named tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor maxI(String a, int axis) {
		String oName = genRandomName();
		return max(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by maxming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code maxI(String a, int axis)}<br>
	 * {@code maxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor maxI(Tensor a) {
		String oName = genRandomName();
		return max(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by maxming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code maxI(String a, int axis)}<br>
	 * {@code maxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor maxI(Tensor a, int axis) {
		String oName = genRandomName();
		return max(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by minming all elements across all axes.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs reduction by minming elements from the specified axis downward.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis axis to reduce from
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
	 * Performs reduction by minming all elements across all axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
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
	 * Performs reduction by minming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce from
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
	 * Performs reduction by minming all elements across all axes of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs reduction by minming elements from the specified axis of the given tensor and stores the result to a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs reduction from the specified axis downward through all subsequent axes.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(String a, String out, int axis) {
		if (!CuBridgeJNI.min(a, out, axis))
			System.err.println("[ERROR][MIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by minming all elements across all axes of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(Tensor a) {
		return min(a, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by minming elements from the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(Tensor a, int axis) {
		return min(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by minming all elements across all axes of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction across all axes (axis = -1).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(Tensor a, String out) {
		return min(a, out, -1);
	}

	/**
	 * Axis Operation (Cascaded-Axis)
	 *
	 * Performs reduction by minming elements from the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code min(String a, String out, int axis)}<br>
	 * {@code min(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as object.</li>
	 *   <li>Performs reduction starting from the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce from
	 * @return CuBridge instance for chaining
	 */
	public CuBridge min(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][MIN][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).min(aName, out, axis);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by minming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code minI(String a, int axis)}<br>
	 * {@code minI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input and performs full-axis reduction.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param — input specification
	 * @return result tensor
	 */
	public Tensor minI() {
		String oName = genRandomName();
		return min("", oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by minming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code minI(String a, int axis)}<br>
	 * {@code minI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue and reduces from the specified axis.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis input specification
	 * @return result tensor
	 */
	public Tensor minI(int axis) {
		String oName = genRandomName();
		return min("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by minming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code minI(String a, int axis)}<br>
	 * {@code minI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the specified tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor minI(String a) {
		String oName = genRandomName();
		return min(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by minming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code minI(String a, int axis)}<br>
	 * {@code minI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the named tensor.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor minI(String a, int axis) {
		String oName = genRandomName();
		return min(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by minming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code minI(String a, int axis)}<br>
	 * {@code minI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces all axes of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input specification
	 * @return result tensor
	 */
	public Tensor minI(Tensor a) {
		String oName = genRandomName();
		return min(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Cascaded-Axis, Immediate)
	 *
	 * Performs reduction by minming elements and directly returns the result tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code minI(String a, int axis)}<br>
	 * {@code minI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Reduces from the specified axis of the given tensor object.</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce from
	 * @return result tensor
	 */
	public Tensor minI(Tensor a, int axis) {
		String oName = genRandomName();
		return min(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, accumulation is performed along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
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
	 * Performs accumulation (cumulative summation) along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis the axis to perform accumulation along; {@code -1} means use the first axis
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
	 * Performs accumulation (cumulative summation) along the first axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs accumulation along the first axis (index 0) by default.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
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
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to perform accumulation along
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
	 * Performs accumulation (cumulative summation) along the first axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs accumulation along the first axis (index 0) by default.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
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
	 * Performs accumulation (cumulative summation) along the specified axis and stores the result in the given output tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis to perform accumulation along
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
	 * Performs accumulation (cumulative summation) along the first axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs accumulation along the first axis (index 0) by default.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(Tensor a) {
		return accumulate(a, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to perform accumulation along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(Tensor a, int axis) {
		return accumulate(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the first axis of the given tensor object and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs accumulation along the first axis (index 0) by default.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(Tensor a, String out) {
		return accumulate(a, out, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor object and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulate(String a, String out, int axis)}<br>
	 * {@code accumulate(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @param axis the axis to perform accumulation along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge accumulate(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][ACCUMULATE][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).accumulate(aName, out, axis);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs accumulation (cumulative summation) along the first axis of the top tensor in the queue and directly returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulateI(String a, int axis)}<br>
	 * {@code accumulateI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs accumulation along the first axis (index 0).</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @return result tensor
	 */
	public Tensor accumulateI() {
		String oName = genRandomName();
		return accumulate("", oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the top tensor in the queue and directly returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulateI(String a, int axis)}<br>
	 * {@code accumulateI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis the axis to perform accumulation along
	 * @return result tensor
	 */
	public Tensor accumulateI(int axis) {
		String oName = genRandomName();
		return accumulate("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs accumulation (cumulative summation) along the first axis of the specified tensor and directly returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulateI(String a, int axis)}<br>
	 * {@code accumulateI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs accumulation along the first axis (index 0).</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @return result tensor
	 */
	public Tensor accumulateI(String a) {
		String oName = genRandomName();
		return accumulate(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs accumulation (cumulative summation) along the first axis of the given tensor object and directly returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulateI(String a, int axis)}<br>
	 * {@code accumulateI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs accumulation along the first axis (index 0).</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return result tensor
	 */
	public Tensor accumulateI(Tensor a) {
		String oName = genRandomName();
		return accumulate(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor and directly returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulateI(String a, int axis)}<br>
	 * {@code accumulateI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to perform accumulation along
	 * @return result tensor
	 */
	public Tensor accumulateI(String a, int axis) {
		String oName = genRandomName();
		return accumulate(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs accumulation (cumulative summation) along the specified axis of the given tensor object and directly returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code accumulateI(String a, int axis)}<br>
	 * {@code accumulateI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs accumulation along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is returned directly as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to perform accumulation along
	 * @return result tensor
	 */
	public Tensor accumulateI(Tensor a, int axis) {
		String oName = genRandomName();
		return accumulate(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs compression (cumulative mean) along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, compression is performed along the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
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
	 * Performs compression (cumulative mean) along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs compression along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis the axis to compress along; -1 means use the first axis
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
	 * Performs compression (cumulative mean) along the first axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, compression is performed along the first axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
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
	 * Performs compression (cumulative mean) along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Performs compression along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to compress along
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
	 * Performs compression (cumulative mean) along the first axis and stores result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input/output tensor names are specified.</li>
	 *   <li>Compression is performed along the first axis.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
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
	 * Performs compression (cumulative mean) along the specified axis and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input/output tensor names are specified.</li>
	 *   <li>Compression is performed along the specified axis.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis the axis to compress along
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
	 * Performs compression (cumulative mean) along the first axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Compression is performed along the first axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(Tensor a) {
		return compress(a, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs compression (cumulative mean) along the specified axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Compression is performed along the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to compress along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(Tensor a, int axis) {
		return compress(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs compression (cumulative mean) along the first axis and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Compression is performed along the first axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(Tensor a, String out) {
		return compress(a, out, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Performs compression (cumulative mean) along the specified axis and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compress(String a, String out, int axis)}<br>
	 * {@code compress(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Compression is performed along the specified axis.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to compress along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge compress(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][COMPRESS][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).compress(aName, out, axis);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs compression along the first axis of the top tensor in the queue and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compressI(String a, int axis)}<br>
	 * {@code compressI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Compression is performed along the first axis.</li>
	 *   <li>Returns the result tensor directly.</li>
	 * </ul>
	 *
	 * @return result tensor
	 */
	public Tensor compressI() {
		String oName = genRandomName();
		return compress("", oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs compression along the specified axis of the top tensor in the queue and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compressI(String a, int axis)}<br>
	 * {@code compressI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Compression is performed along the specified axis.</li>
	 *   <li>Returns the result tensor directly.</li>
	 * </ul>
	 *
	 * @param axis axis to compress along
	 * @return result tensor
	 */
	public Tensor compressI(int axis) {
		String oName = genRandomName();
		return compress("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs compression along the first axis of the given tensor and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compressI(String a, int axis)}<br>
	 * {@code compressI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Compression is performed along the first axis.</li>
	 *   <li>Returns the result tensor directly.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @return result tensor
	 */
	public Tensor compressI(String a) {
		String oName = genRandomName();
		return compress(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs compression along the first axis of the given tensor object and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compressI(String a, int axis)}<br>
	 * {@code compressI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Compression is performed along the first axis.</li>
	 *   <li>Returns the result tensor directly.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return result tensor
	 */
	public Tensor compressI(Tensor a) {
		String oName = genRandomName();
		return compress(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs compression along the specified axis of the given tensor and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compressI(String a, int axis)}<br>
	 * {@code compressI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Compression is performed along the specified axis.</li>
	 *   <li>Returns the result tensor directly.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to compress along
	 * @return result tensor
	 */
	public Tensor compressI(String a, int axis) {
		String oName = genRandomName();
		return compress(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Performs compression along the specified axis of the given tensor object and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code compressI(String a, int axis)}<br>
	 * {@code compressI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Compression is performed along the specified axis.</li>
	 *   <li>Returns the result tensor directly.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to compress along
	 * @return result tensor
	 */
	public Tensor compressI(Tensor a, int axis) {
		String oName = genRandomName();
		return compress(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Expands the specified axis by copying it N times.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code expand(String a, String out, int axis, int N)}<br>
	 * {@code expand(Tensor a, String out, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>Requires that N is divisible by the size of the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code expand(String a, String out, int axis, int N)}<br>
	 * {@code expand(Tensor a, String out, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>Requires that N is divisible by the size of the specified axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code expand(String a, String out, int axis, int N)}<br>
	 * {@code expand(Tensor a, String out, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>Requires that N is divisible by the size of the specified axis.</li>
	 * </ul>
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
	 * Expands the specified axis of the given tensor by copying it N times.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code expand(String a, String out, int axis, int N)}<br>
	 * {@code expand(Tensor a, String out, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis
	 * @return CuBridge instance for chaining
	 */
	public CuBridge expand(Tensor a, int axis, int N) {
		return expand(a, genRandomName(), axis, N);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Expands the specified axis of the given tensor and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code expand(String a, String out, int axis, int N)}<br>
	 * {@code expand(Tensor a, String out, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>The result is stored in the specified output name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis
	 * @return CuBridge instance for chaining
	 */
	public CuBridge expand(Tensor a, String out, int axis, int N) {
		if (a == null) System.err.println("[ERROR][EXPAND][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).expand(aName, out, axis, N);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Expands the specified axis of the top tensor in the queue and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code expandI(String a, int axis, int N)}<br>
	 * {@code expandI(Tensor a, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis
	 * @return result tensor
	 */
	public Tensor expandI(int axis, int N) {
		String oName = genRandomName();
		return expand("", oName, axis, N).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Expands the specified axis of the given tensor and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code expandI(String a, int axis, int N)}<br>
	 * {@code expandI(Tensor a, int axis, int N)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Duplicates the specified axis N times.</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to expand
	 * @param N the target size for the expanded axis
	 * @return result tensor
	 */
	public Tensor expandI(String a, int axis, int N) {
		String oName = genRandomName();
		return expand(a, oName, axis, N).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, the first axis (index 0) is used.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax() {
		if (!CuBridgeJNI.argMax("", genRandomName(), -1))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Specified axis is used (use -1 to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis the axis to operate along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(int axis) {
		if (!CuBridgeJNI.argMax("", genRandomName(), axis))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the first axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>First axis is used (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(String a) {
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
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Specified axis is used (use -1 to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to operate along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(String a, int axis) {
		if (!CuBridgeJNI.argMax(a, genRandomName(), axis))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the first axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>First axis is used (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(Tensor a) {
		return argmax(a, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Specified axis is used (use -1 to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to operate along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(Tensor a, int axis) {
		return argmax(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the first axis and stores it in the specified output name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output name are specified.</li>
	 *   <li>First axis is used (index 0).</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(String a, String out) {
		if (!CuBridgeJNI.argMax(a, out, -1))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis and stores it in the specified output name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output name are specified.</li>
	 *   <li>Specified axis is used (use -1 to select the first axis).</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param out name of the output tensor
	 * @param axis axis to operate along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(String a, String out, int axis) {
		if (!CuBridgeJNI.argMax(a, out, axis))
			System.err.println("[ERROR][ARGMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the first axis and stores it in the specified output name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as an object.</li>
	 *   <li>First axis is used (index 0).</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(Tensor a, String out) {
		return argmax(a, out, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the maximum value along the specified axis and stores it in the specified output name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmax(String a, String out, int axis)}<br>
	 * {@code argmax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as an object.</li>
	 *   <li>Specified axis is used (use -1 to select the first axis).</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @param axis axis to operate along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmax(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][ARGMAX][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).argmax(aName, out, axis);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result tensor containing the index of the maximum value along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmaxI(String a, int axis)}<br>
	 * {@code argmaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Top tensor in the queue is used.</li>
	 *   <li>First axis is used (index 0).</li>
	 *   <li>The result tensor is returned directly.</li>
	 * </ul>
	 *
	 * @return result tensor
	 */
	public Tensor argmaxI() {
		String oName = genRandomName();
		return argmax("", oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result tensor containing the index of the maximum value along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmaxI(String a, int axis)}<br>
	 * {@code argmaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Top tensor in the queue is used.</li>
	 *   <li>Specified axis is used.</li>
	 *   <li>The result tensor is returned directly.</li>
	 * </ul>
	 *
	 * @param axis axis to operate along
	 * @return result tensor
	 */
	public Tensor argmaxI(int axis) {
		String oName = genRandomName();
		return argmax("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result tensor containing the index of the maximum value along the first axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmaxI(String a, int axis)}<br>
	 * {@code argmaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>First axis is used.</li>
	 *   <li>The result tensor is returned directly.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @return result tensor
	 */
	public Tensor argmaxI(String a) {
		String oName = genRandomName();
		return argmax(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result tensor containing the index of the maximum value along the first axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmaxI(String a, int axis)}<br>
	 * {@code argmaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as an object.</li>
	 *   <li>First axis is used.</li>
	 *   <li>The result tensor is returned directly.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return result tensor
	 */
	public Tensor argmaxI(Tensor a) {
		String oName = genRandomName();
		return argmax(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result tensor containing the index of the maximum value along the specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmaxI(String a, int axis)}<br>
	 * {@code argmaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Specified axis is used.</li>
	 *   <li>The result tensor is returned directly.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param axis axis to operate along
	 * @return result tensor
	 */
	public Tensor argmaxI(String a, int axis) {
		String oName = genRandomName();
		return argmax(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result tensor containing the index of the maximum value along the specified axis of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmaxI(String a, int axis)}<br>
	 * {@code argmaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as an object.</li>
	 *   <li>Specified axis is used.</li>
	 *   <li>The result tensor is returned directly.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to operate along
	 * @return result tensor
	 */
	public Tensor argmaxI(Tensor a, int axis) {
		String oName = genRandomName();
		return argmax(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, the first axis (index 0) is used.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin() {
		if (!CuBridgeJNI.argMin("", genRandomName(), -1))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor -, -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Specifies axis to perform reduction. Use {@code -1} for the first axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis the axis along which to find the minimum index
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(int axis) {
		if (!CuBridgeJNI.argMin("", genRandomName(), axis))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the first axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Specifies input tensor by name.</li>
	 *   <li>Uses the first axis (index 0) by default.</li>
	 *   <li>Result name is auto-generated.</li>
	 * </ul>
	 *
	 * @param a name of the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(String a) {
		if (!CuBridgeJNI.argMin(a, genRandomName(), -1))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", -, axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the specified axis of the named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Specifies input tensor by name.</li>
	 *   <li>Axis is explicitly given.</li>
	 *   <li>Result name is auto-generated.</li>
	 * </ul>
	 *
	 * @param a name of the input tensor
	 * @param axis axis to find the minimum index along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(String a, int axis) {
		if (!CuBridgeJNI.argMin(a, genRandomName(), axis))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Stores the index of the minimum value into a given output tensor along the first axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensor names specified.</li>
	 *   <li>Default axis is the first.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(String a, String out) {
		if (!CuBridgeJNI.argMin(a, out, -1))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Stores the index of the minimum value into a given output tensor along the specified axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensor names specified.</li>
	 *   <li>Explicit axis specified.</li>
	 * </ul>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis to find the minimum index along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(String a, String out, int axis) {
		if (!CuBridgeJNI.argMin(a, out, axis))
			System.err.println("[ERROR][ARGMIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along the first axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Tensor is passed as an object.</li>
	 *   <li>Default axis is the first.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(Tensor a) {
		return argmin(a, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the index of the minimum value along a specified axis of the given tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Tensor is passed as an object.</li>
	 *   <li>Explicit axis specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce over
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(Tensor a, int axis) {
		return argmin(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Stores the result of argmin of the input tensor object to the given output name along the first axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(Tensor a, String out) {
		return argmin(a, out, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Stores the result of argmin of the input tensor object to the given output name along the specified axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argmin(String a, String out, int axis)}<br>
	 * {@code argmin(Tensor a, String out, int axis)}
	 * </p>
	 *
	 * @param a input tensor object
	 * @param out output tensor name
	 * @param axis axis to reduce along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge argmin(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][ARGMIN][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).argmin(aName, out, axis);
	}





	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns a tensor with indices of the minimum values along the first axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argminI(String a, int axis)}<br>
	 * {@code argminI(Tensor a, int axis)}
	 * </p>
	 *
	 * @return result tensor
	 */
	public Tensor argminI() {
		String oName = genRandomName();
		return argmin("", oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result of argmin along the given axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argminI(String a, int axis)}<br>
	 * {@code argminI(Tensor a, int axis)}
	 * </p>
	 *
	 * @param axis axis to reduce
	 * @return result tensor
	 */
	public Tensor argminI(int axis) {
		String oName = genRandomName();
		return argmin("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result of argmin of the given tensor name along the first axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argminI(String a, int axis)}<br>
	 * {@code argminI(Tensor a, int axis)}
	 * </p>
	 *
	 * @param a input tensor name
	 * @return result tensor
	 */
	public Tensor argminI(String a) {
		String oName = genRandomName();
		return argmin(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result of argmin of the given tensor name along the given axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argminI(String a, int axis)}<br>
	 * {@code argminI(Tensor a, int axis)}
	 * </p>
	 *
	 * @param a input tensor name
	 * @param axis axis to reduce
	 * @return result tensor
	 */
	public Tensor argminI(String a, int axis) {
		String oName = genRandomName();
		return argmin(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result of argmin of the given tensor object along the first axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argminI(String a, int axis)}<br>
	 * {@code argminI(Tensor a, int axis)}
	 * </p>
	 *
	 * @param a input tensor object
	 * @return result tensor
	 */
	public Tensor argminI(Tensor a) {
		String oName = genRandomName();
		return argmin(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Returns the result of argmin of the given tensor object along the given axis.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code argminI(String a, int axis)}<br>
	 * {@code argminI(Tensor a, int axis)}
	 * </p>
	 *
	 * @param a input tensor object
	 * @param axis axis to reduce
	 * @return result tensor
	 */
	public Tensor argminI(Tensor a, int axis) {
		String oName = genRandomName();
		return argmin(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum value along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
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
	 * Finds the maximum value along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs reduction along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis the axis to reduce along
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
	 * Finds the maximum value along the first axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
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
	 * Finds the maximum value along the specified axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Specified axis is used for maximum extraction (use {@code -1} for the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to reduce along
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
	 * Finds the maximum value along the first axis of the input tensor and stores the result in a named output tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(String a, String out) {
		if (!CuBridgeJNI.axisMax(a, out, -1))
			System.err.println("[ERROR][AXISMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum value along the specified axis of the input tensor and stores the result in a named output tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Specified axis is used for maximum extraction (use {@code -1} for the first axis).</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis to reduce along
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
	 * Finds the maximum value along the first axis of the specified tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Defaults to the first axis if axis is not specified.</li>
	 *   <li>Result is stored in an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(Tensor a) {
		return axisMax(a, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum value along the specified axis of the specified tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Specified axis is used for reduction (use {@code -1} for first axis).</li>
	 *   <li>Result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to reduce along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(Tensor a, int axis) {
		return axisMax(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum value along the first axis of the specified tensor object and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Defaults to the first axis.</li>
	 *   <li>Stores result in a specified output name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(Tensor a, String out) {
		return axisMax(a, out, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the maximum value along the specified axis of the specified tensor object and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMax(String a, String out, int axis)}<br>
	 * {@code axisMax(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs reduction along the specified axis (use {@code -1} for first axis).</li>
	 *   <li>Stores result in the specified output tensor.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @param axis the axis to reduce along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMax(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][AXISMAX][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).axisMax(aName, out, axis);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the maximum value along the first axis of the top tensor in the queue and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMaxI(String a, int axis)}<br>
	 * {@code axisMaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs reduction along the first axis (index 0).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @return result tensor
	 */
	public Tensor axisMaxI() {
		String oName = genRandomName();
		return axisMax("", oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the maximum value along the specified axis of the top tensor in the queue and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMaxI(String a, int axis)}<br>
	 * {@code axisMaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs reduction along the specified axis (use {@code -1} for the first axis).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis the axis to reduce along
	 * @return result tensor
	 */
	public Tensor axisMaxI(int axis) {
		String oName = genRandomName();
		return axisMax("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the maximum value along the first axis of the specified tensor and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMaxI(String a, int axis)}<br>
	 * {@code axisMaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @return result tensor
	 */
	public Tensor axisMaxI(String a) {
		String oName = genRandomName();
		return axisMax(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the maximum value along the specified axis of the specified tensor and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMaxI(String a, int axis)}<br>
	 * {@code axisMaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Specified axis is used for maximum extraction (use {@code -1} for the first axis).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to reduce along
	 * @return result tensor
	 */
	public Tensor axisMaxI(String a, int axis) {
		String oName = genRandomName();
		return axisMax(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the maximum value along the first axis of the specified tensor object and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMaxI(String a, int axis)}<br>
	 * {@code axisMaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return result tensor
	 */
	public Tensor axisMaxI(Tensor a) {
		String oName = genRandomName();
		return axisMax(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the maximum value along the specified axis of the specified tensor object and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMaxI(String a, int axis)}<br>
	 * {@code axisMaxI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Specified axis is used for maximum extraction (use {@code -1} for the first axis).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to reduce along
	 * @return result tensor
	 */
	public Tensor axisMaxI(Tensor a, int axis) {
		String oName = genRandomName();
		return axisMax(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum value along the first axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis (index 0).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
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
	 * Finds the minimum value along the specified axis of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs reduction along the specified axis (use {@code -1} to select the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param axis the axis to reduce along
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
	 * Finds the minimum value along the first axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
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
	 * Finds the minimum value along the specified axis of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Specified axis is used for minimum extraction (use {@code -1} for the first axis).</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to reduce along
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
	 * Finds the minimum value along the first axis of the input tensor and stores the result in a named output tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(String a, String out) {
		if (!CuBridgeJNI.axisMin(a, out, -1))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=-1]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum value along the specified axis of the input tensor and stores the result in a named output tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor and output tensor are specified by name.</li>
	 *   <li>Specified axis is used for minimum extraction (use {@code -1} for the first axis).</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param out the name of the output tensor
	 * @param axis the axis to reduce along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(String a, String out, int axis) {
		if (!CuBridgeJNI.axisMin(a, out, axis))
			System.err.println("[ERROR][AXISMIN][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum value along the first axis of the specified tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Defaults to the first axis if axis is not specified.</li>
	 *   <li>Result is stored in an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(Tensor a) {
		return axisMin(a, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum value along the specified axis of the specified tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Specified axis is used for reduction (use {@code -1} for first axis).</li>
	 *   <li>Result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to reduce along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(Tensor a, int axis) {
		return axisMin(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum value along the first axis of the specified tensor object and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Defaults to the first axis.</li>
	 *   <li>Stores result in a specified output name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(Tensor a, String out) {
		return axisMin(a, out, -1);
	}

	/**
	 * Axis Operation (Single-Axis)
	 *
	 * Finds the minimum value along the specified axis of the specified tensor object and stores the result in a named tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMin(String a, String out, int axis)}<br>
	 * {@code axisMin(Tensor a, String out, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Performs reduction along the specified axis (use {@code -1} for first axis).</li>
	 *   <li>Stores result in the specified output tensor.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @param axis the axis to reduce along
	 * @return CuBridge instance for chaining
	 */
	public CuBridge axisMin(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][AXISMIN][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).axisMin(aName, out, axis);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the minimum value along the first axis of the top tensor in the queue and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMinI(String a, int axis)}<br>
	 * {@code axisMinI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs reduction along the first axis (index 0).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @return result tensor
	 */
	public Tensor axisMinI() {
		String oName = genRandomName();
		return axisMin("", oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the minimum value along the specified axis of the top tensor in the queue and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMinI(String a, int axis)}<br>
	 * {@code axisMinI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Performs reduction along the specified axis (use {@code -1} for the first axis).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param axis the axis to reduce along
	 * @return result tensor
	 */
	public Tensor axisMinI(int axis) {
		String oName = genRandomName();
		return axisMin("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the minimum value along the first axis of the specified tensor and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMinI(String a, int axis)}<br>
	 * {@code axisMinI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @return result tensor
	 */
	public Tensor axisMinI(String a) {
		String oName = genRandomName();
		return axisMin(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the minimum value along the specified axis of the specified tensor and returns the result as a new tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMinI(String a, int axis)}<br>
	 * {@code axisMinI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Specified axis is used for minimum extraction (use {@code -1} for the first axis).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis to reduce along
	 * @return result tensor
	 */
	public Tensor axisMinI(String a, int axis) {
		String oName = genRandomName();
		return axisMin(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the minimum value along the first axis of the specified tensor object and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMinI(String a, int axis)}<br>
	 * {@code axisMinI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>If {@code axis = -1}, the operation defaults to the first axis.</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return result tensor
	 */
	public Tensor axisMinI(Tensor a) {
		String oName = genRandomName();
		return axisMin(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Single-Axis, Immediate)
	 *
	 * Finds the minimum value along the specified axis of the specified tensor object and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code axisMinI(String a, int axis)}<br>
	 * {@code axisMinI(Tensor a, int axis)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Specified axis is used for minimum extraction (use {@code -1} for the first axis).</li>
	 *   <li>The result is returned immediately as a Tensor instance.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis the axis to reduce along
	 * @return result tensor
	 */
	public Tensor axisMinI(Tensor a, int axis) {
		String oName = genRandomName();
		return axisMin(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the last two axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Operates on the top tensor in the queue.</li>
	 *   <li>Swaps the last two axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Swaps the last two axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensors are specified by name.</li>
	 *   <li>Swaps the last two axes.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Operates on the top tensor in the queue.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input and output tensors are specified by name.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 * </ul>
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
	 * Axis Operation (Transpose)
	 *
	 * Swaps the two specified axes of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Swaps the last two axes by default.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(Tensor a) {
		return transpose(a, genRandomName());
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the last two axes of the given tensor object and stores the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Swaps the last two axes by default.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(Tensor a, String out) {
		return transpose(a, out, 0, -1);
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the two specified axes of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis1 the first axis to swap; -1 means swap the last two axes
	 * @param axis2 the second axis to swap; -1 means swap the last two axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(Tensor a, int axis1, int axis2) {
		return transpose(a, genRandomName(), axis1, axis2);
	}

	/**
	 * Axis Operation (Transpose)
	 *
	 * Swaps the two specified axes of the given tensor object and stores the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code transpose(String a, String out, int axis1, int axis2)}<br>
	 * {@code transpose(Tensor a, String out, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>If either {@code axis1} or {@code axis2} is -1, swaps the last two axes.</li>
	 *   <li>Otherwise, swaps the specified axes.</li>
	 *   <li>Output tensor name is specified.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param out name of the output tensor
	 * @param axis1 the first axis to swap; -1 means swap the last two axes
	 * @param axis2 the second axis to swap; -1 means swap the last two axes
	 * @return CuBridge instance for chaining
	 */
	public CuBridge transpose(Tensor a, String out, int axis1, int axis2) {
		if (a == null) System.err.println("[ERROR][TRANSPOSE][Null Tensor Input]");
		String aName = genRandomName();
		return put(a, aName).transpose(aName, out, axis1, axis2);
	}

	/**
	 * Axis Operation (Transpose, Immediate)
	 *
	 * Returns the result of transposing the last two axes of the top tensor in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code T(String a, int axis1, int axis2)}<br>
	 * {@code T(Tensor a, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Swaps the last two axes (axis 0 and -1 by default).</li>
	 *   <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 *
	 * @return transposed tensor
	 */
	public Tensor T() {
		String oName = genRandomName();
		return transpose("", oName).get(oName);
	}

	/**
	 * Axis Operation (Transpose, Immediate)
	 *
	 * Returns the result of transposing the last two axes of the specified tensor.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code T(String a, int axis1, int axis2)}<br>
	 * {@code T(Tensor a, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Swaps the last two axes (axis 0 and -1 by default).</li>
	 *   <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @return transposed tensor
	 */
	public Tensor T(String a) {
		String oName = genRandomName();
		return transpose(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Transpose, Immediate)
	 *
	 * Returns the result of transposing the last two axes of the specified tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code T(String a, int axis1, int axis2)}<br>
	 * {@code T(Tensor a, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Swaps the last two axes (axis 0 and -1 by default).</li>
	 *   <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @return transposed tensor
	 */
	public Tensor T(Tensor a) {
		String oName = genRandomName();
		return transpose(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Transpose, Immediate)
	 *
	 * Returns the result of transposing the specified axes of the given tensor by name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code T(String a, int axis1, int axis2)}<br>
	 * {@code T(Tensor a, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Swaps the specified axes.</li>
	 *   <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 *
	 * @param a the name of the input tensor
	 * @param axis1 the first axis to swap; -1 means the last axis
	 * @param axis2 the second axis to swap; -1 means the last axis
	 * @return transposed tensor
	 */
	public Tensor T(String a, int axis1, int axis2) {
		String oName = genRandomName();
		return transpose(a, oName, axis1, axis2).get(oName);
	}

	/**
	 * Axis Operation (Transpose, Immediate)
	 *
	 * Returns the result of transposing the specified axes of the given tensor object.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code T(String a, int axis1, int axis2)}<br>
	 * {@code T(Tensor a, int axis1, int axis2)}
	 * </p>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified as an object.</li>
	 *   <li>Swaps the specified axes.</li>
	 *   <li>The result is returned as a new Tensor object.</li>
	 * </ul>
	 *
	 * @param a input tensor object
	 * @param axis1 the first axis to swap; -1 means the last axis
	 * @param axis2 the second axis to swap; -1 means the last axis
	 * @return transposed tensor
	 */
	public Tensor T(Tensor a, int axis1, int axis2) {
		String oName = genRandomName();
		return transpose(a, oName, axis1, axis2).get(oName);
	}

	/**
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product (a ⋅ b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dot(String a, String b, String out)}<br>
	 * {@code dot(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both input tensors are taken from the queue.</li>
	 *   <li>Only supports 1D or 2D tensors. For higher dimensions, uses {@code matmul()} internally.</li>
	 *   <li>If one input is 1D and the other is 2D, the 1D tensor is reshaped automatically.</li>
	 *   <li>Axis match condition: last axis of {@code a} must match first axis of {@code b}.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs the dot product between the specified tensor and the top tensor from the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dot(String a, String b, String out)}<br>
	 * {@code dot(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>The first input is specified by name; the second is taken from the queue.</li>
	 *   <li>1D/2D tensors only; higher ranks use {@code matmul()} internally.</li>
	 *   <li>Axis condition: {@code a.lastAxis == b.firstAxis}.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs the dot product between two named tensors.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dot(String a, String b, String out)}<br>
	 * {@code dot(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both input tensors are specified by name.</li>
	 *   <li>Supports 1D or 2D tensors only. Higher dimensions use {@code matmul()}.</li>
	 *   <li>If one is 1D, it's reshaped appropriately.</li>
	 *   <li>Axis condition: {@code a.lastAxis == b.firstAxis}.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs the dot product of two specified tensors and stores the result with a custom name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dot(String a, String b, String out)}<br>
	 * {@code dot(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>1D or 2D tensors only.</li>
	 *   <li>Automatically reshapes 1D inputs when necessary.</li>
	 *   <li>Result is stored under the given output name.</li>
	 * </ul>
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
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product between two tensor objects.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dot(String a, String b, String out)}<br>
	 * {@code dot(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are given as {@code Tensor} objects.</li>
	 *   <li>Internally generates temporary names and stores the result with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a first input tensor
	 * @param b second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(Tensor a, Tensor b) {
		return dot(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Dot Product)
	 *
	 * Performs the dot product between two tensor objects and stores the result with a specified name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dot(String a, String b, String out)}<br>
	 * {@code dot(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Temporary names are auto-generated for internal use.</li>
	 *   <li>Result is saved with the provided name.</li>
	 * </ul>
	 *
	 * @param a first input tensor
	 * @param b second input tensor
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge dot(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][DOT][Null Tensor Input A]");
		if (b == null) System.err.println("[ERROR][DOT][Null Tensor Input B]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).dot(aName, bName, out);
	}

	/**
	 * Binary Operation (Dot Product, Immediate)
	 *
	 * Performs the dot product between the top two tensors in the queue and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dotI(String a, String b)}<br>
	 * {@code dotI(Tensor a, Tensor b)}
	 * </p>
	 * <ul>
	 *   <li>Supports only 1D or 2D tensors.</li>
	 *   <li>Returns the output as a {@code Tensor} object.</li>
	 * </ul>
	 *
	 * @return result of the dot product
	 */
	public Tensor dotI() {
		String oName = genRandomName();
		return dot("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Dot Product, Immediate)
	 *
	 * Performs the dot product between two named tensors and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dotI(String a, String b)}<br>
	 * {@code dotI(Tensor a, Tensor b)}
	 * </p>
	 * <ul>
	 *   <li>Inputs are specified by name.</li>
	 *   <li>Result is returned as a {@code Tensor} object.</li>
	 * </ul>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return result of the dot product
	 */
	public Tensor dotI(String a, String b) {
		String oName = genRandomName();
		return dot(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Dot Product, Immediate)
	 *
	 * Performs the dot product between two tensor objects and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code dotI(String a, String b)}<br>
	 * {@code dotI(Tensor a, Tensor b)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Result is returned as a {@code Tensor} object.</li>
	 * </ul>
	 *
	 * @param a first input tensor
	 * @param b second input tensor
	 * @return result of the dot product
	 */
	public Tensor dotI(Tensor a, Tensor b) {
		String oName = genRandomName();
		return dot(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs the matrix multiplication (a @ b) between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmul(String a, String b, String out)}<br>
	 * {@code matmul(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both input tensors are taken from the queue.</li>
	 *   <li>Supports tensors with rank ≥ 2.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>Broadcasting is applied to the leading axes if necessary.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs the matrix multiplication between the specified tensor and the top tensor from the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmul(String a, String b, String out)}<br>
	 * {@code matmul(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>The first input is specified by name; the second is taken from the queue.</li>
	 *   <li>Supports tensors with rank ≥ 2.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>Broadcasting is applied to the leading axes if necessary.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs the matrix multiplication between two named tensors.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmul(String a, String b, String out)}<br>
	 * {@code matmul(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both input tensors are specified by name.</li>
	 *   <li>Supports tensors with rank ≥ 2.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>Broadcasting is applied to the leading axes if necessary.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
	 * </ul>
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
	 * Performs the matrix multiplication of two specified tensors and stores the result with a custom name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmul(String a, String b, String out)}<br>
	 * {@code matmul(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Supports tensors with rank ≥ 2.</li>
	 *   <li>The last axis of {@code a} must match the second-to-last axis of {@code b}.</li>
	 *   <li>Result is stored under the given output name.</li>
	 * </ul>
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

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs the matrix multiplication between two tensor objects.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmul(String a, String b, String out)}<br>
	 * {@code matmul(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are given as {@code Tensor} objects.</li>
	 *   <li>Internally generates temporary names and stores the result with an auto-generated name.</li>
	 * </ul>
	 *
	 * @param a first input tensor
	 * @param b second input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge matmul(Tensor a, Tensor b) {
		return matmul(a, b, genRandomName());
	}

	/**
	 * Binary Operation (Matrix Multiplication)
	 *
	 * Performs the matrix multiplication between two tensor objects and stores the result with a specified name.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmul(String a, String b, String out)}<br>
	 * {@code matmul(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Temporary names are auto-generated for internal use.</li>
	 *   <li>Result is saved with the provided name.</li>
	 * </ul>
	 *
	 * @param a first input tensor
	 * @param b second input tensor
	 * @param out name of the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge matmul(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][MATMUL][Null Tensor Input A]");
		if (b == null) System.err.println("[ERROR][MATMUL][Null Tensor Input B]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).matmul(aName, bName, out);
	}

	/**
	 * Binary Operation (Matrix Multiplication, Immediate)
	 *
	 * Performs the matrix multiplication between the top two tensors in the queue and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmulI(String a, String b)}<br>
	 * {@code matmulI(Tensor a, Tensor b)}
	 * </p>
	 * <ul>
	 *   <li>Supports tensors with rank ≥ 2.</li>
	 *   <li>Returns the output as a {@code Tensor} object.</li>
	 * </ul>
	 *
	 * @return result of the matrix multiplication
	 */
	public Tensor matmulI() {
		String oName = genRandomName();
		return matmul("", "", oName).get(oName);
	}

	/**
	 * Binary Operation (Matrix Multiplication, Immediate)
	 *
	 * Performs the matrix multiplication between two named tensors and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmulI(String a, String b)}<br>
	 * {@code matmulI(Tensor a, Tensor b)}
	 * </p>
	 * <ul>
	 *   <li>Inputs are specified by name.</li>
	 *   <li>Result is returned as a {@code Tensor} object.</li>
	 * </ul>
	 *
	 * @param a the name of the first input tensor
	 * @param b the name of the second input tensor
	 * @return result of the matrix multiplication
	 */
	public Tensor matmulI(String a, String b) {
		String oName = genRandomName();
		return matmul(a, b, oName).get(oName);
	}

	/**
	 * Binary Operation (Matrix Multiplication, Immediate)
	 *
	 * Performs the matrix multiplication between two tensor objects and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code matmulI(String a, String b)}<br>
	 * {@code matmulI(Tensor a, Tensor b)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Result is returned as a {@code Tensor} object.</li>
	 * </ul>
	 *
	 * @param a first input tensor
	 * @param b second input tensor
	 * @return result of the matrix multiplication
	 */
	public Tensor matmulI(Tensor a, Tensor b) {
		String oName = genRandomName();
		return matmul(a, b, oName).get(oName);
	}

	// 신경망 특화
	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the Mean Squared Error (MSE) loss between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mse(String a, String b, String out)}<br>
	 * {@code mse(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Uses the top two tensors in the queue as inputs.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse() {
		if (!CuBridgeJNI.mse("", "", genRandomName()))
			System.err.println("[ERROR][MSE][Cannot Execute][Tensor -, -]");
		return instance;
	}

	/**
	 * Loss Operation (Mean Squared Error, Immediate)
	 *
	 * Computes the MSE loss between the specified prediction and target tensors and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mse(String a, String b, String out)}<br>
	 * {@code mse(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Returns a scalar (1x1) {@code Tensor} representing the loss.</li>
	 * </ul>
	 *
	 * @param a predicted output tensor
	 * @param b target tensor
	 * @return tensor containing the total MSE loss
	 */
	public Tensor mse(Tensor a, Tensor b) {
		String oName = genRandomName();
		return mse(a, b, oName).get(oName);
	}

	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the MSE loss between the specified prediction and target tensors and stores the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mse(String a, String b, String out)}<br>
	 * {@code mse(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Result is stored with the specified output name.</li>
	 * </ul>
	 *
	 * @param a predicted output tensor
	 * @param b target tensor
	 * @param out name of the output tensor to store the result
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][MSE][Null Tensor Input A]");
		if (b == null) System.err.println("[ERROR][MSE][Null Tensor Input B]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).mse(aName, bName, out);
	}

	/**
	 * Loss Operation (Mean Squared Error)
	 *
	 * Computes the MSE loss between the specified prediction and target tensors and stores the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code mse(String a, String b, String out)}<br>
	 * {@code mse(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Prediction, target, and output tensors are all specified by name.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>The result is a scalar (1x1 tensor) containing the total loss.</li>
	 * </ul>
	 *
	 * @param a the name of the predicted output tensor
	 * @param b  the name of the target tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge mse(String a, String b, String out) {
		if (!CuBridgeJNI.mse(a, b, out))
			System.err.println("[ERROR][MSE][Cannot Execute][Tensor " + a + ", " + b + ", out=" + out + "]");
		return instance;
	}

	/**
	 * Loss Operation (Cross Entropy Error)
	 *
	 * Computes the Cross Entropy Error (CEE) loss between the top two tensors in the queue.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code cee(String a, String b, String out)}<br>
	 * {@code cee(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Uses the top two tensors in the queue as inputs.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>Returns a scalar (1x1 tensor) containing the total loss.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 *
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee() {
		if (!CuBridgeJNI.cee("", "", genRandomName()))
			System.err.println("[ERROR][CEE][Cannot Execute][Tensor -, -]");
		return instance;
	}

	/**
	 * Loss Operation (Cross Entropy Error, Immediate)
	 *
	 * Computes the CEE loss between the specified prediction and target tensors and returns the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code cee(String a, String b, String out)}<br>
	 * {@code cee(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Returns a scalar (1x1) {@code Tensor} representing the loss.</li>
	 * </ul>
	 *
	 * @param a predicted output tensor
	 * @param b target tensor
	 * @return tensor containing the total CEE loss
	 */
	public Tensor cee(Tensor a, Tensor b) {
		String oName = genRandomName();
		return cee(a, b, oName).get(oName);
	}

	/**
	 * Loss Operation (Cross Entropy Error)
	 *
	 * Computes the CEE loss between the specified prediction and target tensors and stores the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code cee(String a, String b, String out)}<br>
	 * {@code cee(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Both inputs are {@code Tensor} objects.</li>
	 *   <li>Result is stored with the specified output name.</li>
	 * </ul>
	 *
	 * @param a predicted output tensor
	 * @param b target tensor
	 * @param out name of the output tensor to store the result
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee(Tensor a, Tensor b, String out) {
		if (a == null) System.err.println("[ERROR][CEE][Null Tensor Input A]");
		if (b == null) System.err.println("[ERROR][CEE][Null Tensor Input B]");

		String aName = genRandomName();
		String bName = genRandomName();

		return put(a, aName).put(b, bName).cee(aName, bName, out);
	}

	/**
	 * Loss Operation (Cross Entropy Error)
	 *
	 * Computes the CEE loss between the specified prediction and target tensors and stores the result.
	 *
	 * <p>
	 * Full parameter:<br>
	 * {@code cee(String a, String b, String out)}<br>
	 * {@code cee(Tensor a, Tensor b, String out)}
	 * </p>
	 * <ul>
	 *   <li>Prediction, target, and output tensors are all specified by name.</li>
	 *   <li>Both tensors must have the same shape.</li>
	 *   <li>The result is a scalar (1x1 tensor) containing the total loss.</li>
	 * </ul>
	 *
	 * @param a the name of the predicted output tensor
	 * @param b the name of the target tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge cee(String a, String b, String out) {
		if (!CuBridgeJNI.cee(a, b, out))
			System.err.println("[ERROR][CEE][Cannot Execute][Tensor " + a + ", " + b + ", out=" + out + "]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation: x · w + b using the top three tensors in the queue.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * Full parameter: {@code affine(Tensor x, Tensor w, Tensor b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top three tensors in the queue as inputs (input, weight, bias).</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w}.</li>
	 *   <li>{@code w} supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} must match the output’s column count (last axis of {@code w}), and can always be broadcasted.</li>
	 *   <li>The result is stored with an auto-generated name.</li>
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
	 * Computes affine transformation from named tensors and returns the result.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>All input tensors are specified by name.</li>
	 *   <li>The last axis of {@code x} must match the second-to-last axis of {@code w}.</li>
	 *   <li>{@code w} supports broadcasting in all leading axes except the last two.</li>
	 *   <li>{@code b} must match the output’s column count and can always be broadcasted.</li>
	 *   <li>The result is returned as a {@code Tensor} with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x the name of the input tensor
	 * @param w the name of the weight tensor
	 * @param b the name of the bias tensor
	 * @return resulting Tensor of the affine operation
	 */
	public Tensor affine(String x, String w, String b) {
		String oName = genRandomName();
		return affine(x, w, b, oName).get(oName);
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Computes affine transformation from given tensor objects and returns the result.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>All inputs are passed as {@code Tensor} objects.</li>
	 *   <li>Internal temporary names are generated for registration.</li>
	 *   <li>The result is returned as a {@code Tensor} with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x input tensor
	 * @param w weight tensor
	 * @param b bias tensor
	 * @return resulting Tensor of the affine operation
	 */
	public Tensor affine(Tensor x, Tensor w, Tensor b) {
		String oName = genRandomName();
		return affine(x, w, b, oName).get(oName);
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation with all input and output tensors specified by name.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>All four tensors are specified by name.</li>
	 *   <li>{@code x.lastAxis} must match {@code w.secondLastAxis}.</li>
	 *   <li>{@code w} may be broadcasted in all leading axes except the last two.</li>
	 *   <li>{@code b} must match the output column count and may always be broadcasted.</li>
	 *   <li>Stores result under the given output name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x the name of the input tensor
	 * @param w the name of the weight tensor
	 * @param b the name of the bias tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine(String x, String w, String b, String out) {
		if (!CuBridgeJNI.affine(x, w, b, out))
			System.err.println("[ERROR][AFFINE][Cannot Execute][Tensor " + x + ", " + w + ", " + b + ", " + out + "]");
		return instance;
	}

	/**
	 * Affine Operation (x · w + b)
	 *
	 * Performs affine transformation and stores result under a custom output name.
	 *
	 * <p>
	 * Full parameter: {@code affine(String x, String w, String b, String out)}<br>
	 * This version:
	 * <ul>
	 *   <li>All tensors are specified as {@code Tensor} objects.</li>
	 *   <li>{@code x.lastAxis == w.secondLastAxis} must hold true.</li>
	 *   <li>{@code w} supports partial broadcasting; {@code b} may be broadcasted freely.</li>
	 *   <li>Input tensors are registered with temporary names.</li>
	 *   <li>Result is saved under the provided output name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param x input tensor
	 * @param w weight tensor
	 * @param b bias tensor
	 * @param out the name to store the result
	 * @return CuBridge instance for chaining
	 */
	public CuBridge affine(Tensor x, Tensor w, Tensor b, String out) {
		if (x == null) System.err.println("[ERROR][AFFINE][Null Tensor Input X]");
		if (w == null) System.err.println("[ERROR][AFFINE][Null Tensor Input W]");
		if (b == null) System.err.println("[ERROR][AFFINE][Null Tensor Input B]");

		String xName = genRandomName();
		String wName = genRandomName();
		String bName = genRandomName();

		return put(x, xName).put(w, wName).put(b, bName).affine(xName, wName, bName, out);
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies the softmax function to the top tensor in the queue along axis 1.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Uses the top tensor in the queue as input.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Result is stored with an auto-generated name.</li>
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
	 * Axis Operation (Softmax)
	 *
	 * Applies softmax to the specified {@code Tensor} along axis 1.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Applies softmax along axis 1.</li>
	 *   <li>Result is stored with an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String a) {
		if (!CuBridgeJNI.softmax(a, genRandomName(), 1))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + a + ", -, axis=1]");
		return instance;
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies softmax to the specified {@code Tensor} and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both input and output tensors are specified by name.</li>
	 *   <li>Applies softmax along axis 1.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String a, String out) {
		if (!CuBridgeJNI.softmax(a, out, 1))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=1]");
		return instance;
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies softmax to the top tensor in the queue along a specified axis.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Applies softmax along the given axis.</li>
	 *   <li>Input is taken from the top of the queue.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(int axis) {
		if (!CuBridgeJNI.softmax("", genRandomName(), axis))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor -, -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies softmax to a named {@code Tensor} along a specified axis.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input is specified by name.</li>
	 *   <li>Softmax is applied along the given axis.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param axis axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String a, int axis) {
		if (!CuBridgeJNI.softmax(a, genRandomName(), axis))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + a + ", -, axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies softmax to a named {@code Tensor} along a given axis and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Both input and output tensors are specified by name.</li>
	 *   <li>Softmax is applied along the given axis.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a input tensor name
	 * @param out output tensor name
	 * @param axis axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(String a, String out, int axis) {
		if (!CuBridgeJNI.softmax(a, out, axis))
			System.err.println("[ERROR][SOFTMAX][Cannot Execute][Tensor " + a + ", " + out + ", axis=" + axis + "]");
		return instance;
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies the softmax function to the specified {@code Tensor} along axis 1.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided directly as a {@code Tensor} object.</li>
	 *   <li>The tensor is pushed to the queue under a temporary name.</li>
	 *   <li>Softmax is applied along axis 1 (second dimension).</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(Tensor a) {
		return softmax(a, genRandomName());
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies the softmax function to the specified {@code Tensor} along axis 1 and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as a {@code Tensor} object.</li>
	 *   <li>Softmax is applied along axis 1 (second dimension).</li>
	 *   <li>The result is stored under the specified name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input tensor
	 * @param out the name for the output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(Tensor a, String out) {
		return softmax(a, out, 1);
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies the softmax function to the specified {@code Tensor} along the given axis.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as a {@code Tensor} object.</li>
	 *   <li>Softmax is applied along the specified axis.</li>
	 *   <li>The result is stored under an auto-generated name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input tensor
	 * @param axis the axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(Tensor a, int axis) {
		return softmax(a, genRandomName(), axis);
	}

	/**
	 * Axis Operation (Softmax)
	 *
	 * Applies the softmax function to the specified {@code Tensor} along the given axis and stores the result.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided as a {@code Tensor} object.</li>
	 *   <li>Softmax is applied along the specified axis.</li>
	 *   <li>The result is stored under the specified name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input tensor
	 * @param out the name for the output tensor
	 * @param axis the axis along which to apply softmax
	 * @return CuBridge instance for chaining
	 */
	public CuBridge softmax(Tensor a, String out, int axis) {
		if (a == null) System.err.println("[ERROR][SOFTMAX][Null Tensor Input Name]");

		String aName = genRandomName();

		return put(a, aName).softmax(aName, out, axis);
	}

	/**
	 * Axis Operation (Softmax Immediate)
	 *
	 * Applies the softmax function to the top tensor in the queue along axis 1 and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Returns the result immediately as a {@code Tensor} object.</li>
	 * </ul>
	 * </p>
	 *
	 * @return the resulting tensor after applying softmax
	 */
	public Tensor softmaxI() {
		String oName = genRandomName();
		return softmax("", oName).get(oName);
	}

	/**
	 * Axis Operation (Softmax Immediate)
	 *
	 * Applies the softmax function to the specified tensor by name along axis 1 and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Returns the result immediately as a {@code Tensor} object.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @return the resulting tensor after applying softmax
	 */
	public Tensor softmaxI(String a) {
		String oName = genRandomName();
		return softmax(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Softmax Immediate)
	 *
	 * Applies the softmax function to the specified tensor object along axis 1 and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided directly as a {@code Tensor} object.</li>
	 *   <li>Applies softmax along axis 1 (second dimension).</li>
	 *   <li>Returns the result immediately as a {@code Tensor} object.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input tensor
	 * @return the resulting tensor after applying softmax
	 */
	public Tensor softmaxI(Tensor a) {
		String oName = genRandomName();
		return softmax(a, oName).get(oName);
	}

	/**
	 * Axis Operation (Softmax Immediate)
	 *
	 * Applies the softmax function to the top tensor in the queue along the specified axis and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Takes the top tensor in the queue as input.</li>
	 *   <li>Applies softmax along the specified axis.</li>
	 *   <li>Returns the result immediately as a {@code Tensor} object.</li>
	 * </ul>
	 * </p>
	 *
	 * @param axis the axis along which to apply softmax
	 * @return the resulting tensor after applying softmax
	 */
	public Tensor softmaxI(int axis) {
		String oName = genRandomName();
		return softmax("", oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Softmax Immediate)
	 *
	 * Applies the softmax function to the specified tensor name along the given axis and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is specified by name.</li>
	 *   <li>Applies softmax along the specified axis.</li>
	 *   <li>Returns the result immediately as a {@code Tensor} object.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the name of the input tensor
	 * @param axis the axis along which to apply softmax
	 * @return the resulting tensor after applying softmax
	 */
	public Tensor softmaxI(String a, int axis) {
		String oName = genRandomName();
		return softmax(a, oName, axis).get(oName);
	}

	/**
	 * Axis Operation (Softmax Immediate)
	 *
	 * Applies the softmax function to the specified tensor object along the given axis and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code softmax(String name, String out, int axis)}<br>
	 * Full parameter: {@code softmax(Tensor name, String out, int axis)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input tensor is provided directly as a {@code Tensor} object.</li>
	 *   <li>Applies softmax along the specified axis.</li>
	 *   <li>Returns the result immediately as a {@code Tensor} object.</li>
	 * </ul>
	 * </p>
	 *
	 * @param a the input tensor
	 * @param axis the axis along which to apply softmax
	 * @return the resulting tensor after applying softmax
	 */
	public Tensor softmaxI(Tensor a, int axis) {
		String oName = genRandomName();
		return softmax(a, oName, axis).get(oName);
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
	 *   <li>The result is stored under the specified output name.</li>
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
	 *   <li>The result is stored under the specified output name.</li>
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
	 * Transformation Operation (im2col 1D)
	 *
	 * Converts a 1D input tensor into column format using kernel, padding, and stride.
	 *
	 * <p>
	 * Full parameter: {@code im2col1D(String input, String kernel, String out, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>{@code input} and {@code kernel} are Tensor objects passed directly.</li>
	 *   <li>All layout requirements for input, kernel, and stride apply.</li>
	 *   <li>The result is stored under the specified output name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the input Tensor object
	 * @param kernel the kernel Tensor object
	 * @param out    the name for the output tensor
	 * @param pad    padding applied to both sides
	 * @param stride stride value between kernel applications
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col1D(Tensor input, Tensor kernel, String out, int pad, int stride) {
		if (input == null) System.err.println("[ERROR][IM2COL1D][Null Tensor Input Input]");
		if (kernel == null) System.err.println("[ERROR][IM2COL1D][Null Tensor Input Kernel]");

		String iName = genRandomName();
		String kName = genRandomName();

		return put(input, iName).put(kernel, kName).im2col1D(iName, kName, out, pad, stride);
	}

	/**
	 * Transformation Operation (im2col 1D)
	 *
	 * Performs im2col 1D and returns the result as a Tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col1D(String input, String kernel, String out, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>All input parameters are provided by name.</li>
	 *   <li>The output tensor is auto-named and immediately returned.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the input tensor
	 * @param kernel the name of the kernel tensor
	 * @param pad    padding value
	 * @param stride stride value
	 * @return the resulting Tensor
	 * @since v1.1
	 */
	public Tensor im2col1D(String input, String kernel, int pad, int stride) {
		String oName = genRandomName();
		return im2col1D(input, kernel, oName, pad, stride).get(oName);
	}

	/**
	 * Transformation Operation (im2col 1D)
	 *
	 * Performs im2col 1D and returns the result as a Tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col1D(String input, String kernel, String out, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>{@code input} and {@code kernel} are Tensor objects passed directly.</li>
	 *   <li>The output tensor is auto-named and returned immediately.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the input Tensor object
	 * @param kernel the kernel Tensor object
	 * @param pad    padding value
	 * @param stride stride value
	 * @return the resulting Tensor
	 * @since v1.1
	 */
	public Tensor im2col1D(Tensor input, Tensor kernel, int pad, int stride) {
		String oName = genRandomName();
		return im2col1D(input, kernel, oName, pad, stride).get(oName);
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
	 *   <li>Input and kernel tensors are specified by name.</li>
	 *   <li>Applies default output length (-1), padding (0), and stride (1).</li>
	 *   <li>The output length is automatically inferred when oL = -1.</li>
	 *   <li>The reconstructed result is stored under the specified name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the reconstructed output tensor
	 * @return CuBridge instance for chaining
	 */
	public CuBridge col2im1D(String input, String kernel, String out) {
		if (!CuBridgeJNI.col2im1D(input, kernel, out, -1, 0, 1))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		return instance;
	}

	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Reconstructs a 1D input tensor using kernel and specified padding.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input and kernel tensors are specified by name.</li>
	 *   <li>Uses default output length (-1) and stride (1).</li>
	 *   <li>Padding is applied symmetrically as specified by the user.</li>
	 *   <li>Output tensor is stored under the specified name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @param pad    padding value used in im2col1D
	 * @return CuBridge instance for chaining
	 */
	public CuBridge col2im1D(String input, String kernel, String out, int pad) {
		if (!CuBridgeJNI.col2im1D(input, kernel, out, -1, pad, 1))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		return instance;
	}

	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Reconstructs a 1D input tensor using kernel, padding, and stride.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>All input tensor names and reconstruction parameters are provided.</li>
	 *   <li>Output length remains inferred (oL = -1) based on kernel and stride.</li>
	 *   <li>Both padding and stride values can be customized.</li>
	 *   <li>Output is stored under the given name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the output tensor
	 * @param pad    padding used during im2col
	 * @param stride stride used during im2col
	 * @return CuBridge instance for chaining
	 */
	public CuBridge col2im1D(String input, String kernel, String out, int pad, int stride) {
		if (!CuBridgeJNI.col2im1D(input, kernel, out, -1, pad, stride))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		return instance;
	}

	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Fully reconstructs a 1D input tensor using all parameters.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>Specifies input/output names, kernel, output length, padding, and stride.</li>
	 *   <li>Provides complete control for accurate reconstruction of input.</li>
	 *   <li>Useful when shape inference is not desirable.</li>
	 *   <li>The output tensor is stored with the given name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param out    the name for the reconstructed output tensor
	 * @param oL     original length of the input signal
	 * @param pad    padding value used during im2col
	 * @param stride stride value used during im2col
	 * @return CuBridge instance for chaining
	 */
	public CuBridge col2im1D(String input, String kernel, String out, int oL, int pad, int stride) {
		if (!CuBridgeJNI.col2im1D(input, kernel, out, oL, pad, stride))
			System.err.println("[ERROR][COL2IM][Cannot Execute][Tensor " + input + ", " + out + "] Please verify tensor existence and parameters.");
		return instance;
	}

	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Reconstructs a 1D tensor using tensor objects and full parameters.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(Tensor input, Tensor kernel, String out, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>Accepts {@code Tensor} objects as input and kernel directly.</li>
	 *   <li>Assigns temporary names internally and stores the result under {@code out}.</li>
	 *   <li>All reconstruction parameters (length, pad, stride) are explicitly set.</li>
	 *   <li>Recommended when tensors are created programmatically.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the input tensor in column format
	 * @param kernel the kernel tensor
	 * @param out    the name for the output tensor
	 * @param oL     original input length
	 * @param pad    padding used in im2col
	 * @param stride stride used in im2col
	 * @return CuBridge instance for chaining
	 */
	public CuBridge col2im1D(Tensor input, Tensor kernel, String out, int oL, int pad, int stride) {
		if (input == null) System.err.println("[ERROR][COL2IM1D][Null Tensor Input]");
		if (kernel == null) System.err.println("[ERROR][COL2IM1D][Null Tensor Kernel]");
		String iName = genRandomName();
		String kName = genRandomName();
		return put(input, iName).put(kernel, kName).col2im1D(iName, kName, out, oL, pad, stride);
	}

	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Executes full col2im1D operation and returns the result immediately.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(String input, String kernel, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input and kernel are specified by name.</li>
	 *   <li>Output is returned as {@code Tensor} instead of stored in queue.</li>
	 *   <li>Used when only immediate result is needed.</li>
	 *   <li>All reconstruction parameters are manually controlled.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the name of the column input tensor
	 * @param kernel the name of the kernel tensor
	 * @param oL     original input length
	 * @param pad    padding used during im2col
	 * @param stride stride used during im2col
	 * @return the reconstructed output tensor
	 */
	public Tensor col2im1D(String input, String kernel, int oL, int pad, int stride) {
		String oName = genRandomName();
		return col2im1D(input, kernel, oName, oL, pad, stride).get(oName);
	}

	/**
	 * Transformation Operation (col2im 1D)
	 *
	 * Executes full col2im1D operation from {@code Tensor} objects and returns result.
	 *
	 * <p>
	 * Full parameter: {@code col2im1D(Tensor input, Tensor kernel, int oL, int pad, int stride)}<br>
	 * This version:
	 * <ul>
	 *   <li>Input and kernel are passed as {@code Tensor} objects.</li>
	 *   <li>Output is returned immediately and not stored in queue.</li>
	 *   <li>Used for concise flow when temporary result is needed.</li>
	 *   <li>All transformation parameters are controlled directly.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input  the input tensor in column format
	 * @param kernel the kernel tensor
	 * @param oL     original input length
	 * @param pad    padding used in im2col
	 * @param stride stride used in im2col
	 * @return the reconstructed output tensor
	 */
	public Tensor col2im1D(Tensor input, Tensor kernel, int oL, int pad, int stride) {
		String oName = genRandomName();
		return col2im1D(input, kernel, oName, oL, pad, stride).get(oName);
	}

	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format using a 2D kernel tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(String input, String kernel, String out)}<br>
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
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(String input, String kernel, String out, int pad)}<br>
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
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(String input, String kernel, String out, int pad, int stride)}<br>
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
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
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
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format using full tensor objects.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * <ul>
	 *   <li>The input tensor must be 3D or 4D (shape: [N, H, W] or [N, C, H, W]).</li>
	 *   <li>Both input and kernel are passed directly as Tensor objects.</li>
	 *   <li>Supports asymmetric padding and non-uniform stride across height and width.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the 2D input tensor object
	 * @param kernel   the 2D kernel tensor object
	 * @param out      the name for the output tensor
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return CuBridge instance for chaining
	 * @since v1.1
	 */
	public CuBridge im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW) {
		if (input == null) System.err.println("[ERROR][IM2COL2D][Null Tensor Input Input]");
		if (kernel == null) System.err.println("[ERROR][IM2COL2D][Null Tensor Input Kernel]");

		String iName = genRandomName();
		String kName = genRandomName();

		return put(input, iName).put(kernel, kName).im2col2D(iName, kName, out, padH, padW, strideH, strideW);
	}

	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format and returns the result tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(String input, String kernel, int padH, int padW, int strideH, int strideW)}<br>
	 * <ul>
	 *   <li>Returns a new tensor by applying the im2col transformation.</li>
	 *   <li>Input and kernel are referenced by name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the name of the 2D input tensor
	 * @param kernel   the name of the 2D kernel tensor
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return resulting Tensor object
	 * @since v1.1
	 */
	public Tensor im2col2D(String input, String kernel, int padH, int padW, int strideH, int strideW) {
		String oName = genRandomName();
		return im2col2D(input, kernel, oName, padH, padW, strideH, strideW).get(oName);
	}

	/**
	 * Transformation Operation (im2col 2D)
	 *
	 * Converts a 2D input tensor into column format and returns the result tensor.
	 *
	 * <p>
	 * Full parameter: {@code im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW)} / {@code im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code im2col2D(Tensor input, Tensor kernel, int padH, int padW, int strideH, int strideW)}<br>
	 * <ul>
	 *   <li>Returns a new tensor by applying the im2col transformation.</li>
	 *   <li>Input and kernel are passed directly as Tensor objects.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the 2D input tensor object
	 * @param kernel   the 2D kernel tensor object
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return resulting Tensor object
	 * @since v1.1
	 */
	public Tensor im2col2D(Tensor input, Tensor kernel, int padH, int padW, int strideH, int strideW) {
		String oName = genRandomName();
		return im2col2D(input, kernel, oName, padH, padW, strideH, strideW).get(oName);
	}

	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Reconstructs the original 2D input tensor from a column matrix.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(String input, String kernel, String out)}<br>
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
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(String input, String kernel, String out, int pad)}<br>
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
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(String input, String kernel, String out, int pad, int stride)}<br>
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
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
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

	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Fully reconstructs a 2D input tensor using full tensor objects and all parameters.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * <ul>
	 *   <li>Both input and kernel are passed as tensor objects.</li>
	 *   <li>Reconstruction matches configuration used during im2col2D.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the column input tensor object
	 * @param kernel   the kernel tensor object
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
	public CuBridge col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW) {
		if (input == null) System.err.println("[ERROR][COL2IM2D][Null Tensor Input Input]");
		if (kernel == null) System.err.println("[ERROR][COL2IM2D][Null Tensor Input Kernel]");

		String iName = genRandomName();
		String kName = genRandomName();

		return put(input, iName).put(kernel, kName).col2im2D(iName, kName, out, oH, oW, padH, padW, strideH, strideW);
	}

	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Fully reconstructs and returns a 2D input tensor from column format.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(String input, String kernel, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * <ul>
	 *   <li>Returns the result of the full col2im2D reconstruction.</li>
	 *   <li>Input and kernel are referenced by name.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the name of the column input tensor
	 * @param kernel   the name of the kernel tensor
	 * @param oH       original input height
	 * @param oW       original input width
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return resulting Tensor object
	 * @since v1.1
	 */
	public Tensor col2im2D(String input, String kernel, int oH, int oW, int padH, int padW, int strideH, int strideW) {
		String oName = genRandomName();
		return col2im2D(input, kernel, oName, oH, oW, padH, padW, strideH, strideW).get(oName);
	}

	/**
	 * Transformation Operation (col2im 2D)
	 *
	 * Fully reconstructs and returns a 2D input tensor from column format using Tensor objects.
	 *
	 * <p>
	 * Full parameter: {@code col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)} / {@code col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * This version: {@code col2im2D(Tensor input, Tensor kernel, int oH, int oW, int padH, int padW, int strideH, int strideW)}<br>
	 * <ul>
	 *   <li>Returns the result of the full col2im2D reconstruction.</li>
	 *   <li>Input and kernel are passed as tensor objects.</li>
	 * </ul>
	 * </p>
	 *
	 * @param input    the column input tensor object
	 * @param kernel   the kernel tensor object
	 * @param oH       original input height
	 * @param oW       original input width
	 * @param padH     padding height
	 * @param padW     padding width
	 * @param strideH  stride height
	 * @param strideW  stride width
	 * @return resulting Tensor object
	 * @since v1.1
	 */
	public Tensor col2im2D(Tensor input, Tensor kernel, int oH, int oW, int padH, int padW, int strideH, int strideW) {
		String oName = genRandomName();
		return col2im2D(input, kernel, oName, oH, oW, padH, padW, strideH, strideW).get(oName);
	}
}
