package CuBridge;

import java.util.UUID;

public class SystemOps {

    private String genRandomNameSystem() {
        return "SystemOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
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
        CuBridgeJNI.put(data.toArray(), data.getShape(), data.getSize(), data.getAxis(), 1, genRandomNameSystem(), broadcast);
        return CuBridge.getInstance();
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
            return CuBridge.getInstance();
        }

        if (usageCount == 0) {
            System.err.println("Error: Please UsageCount modify.");
            return CuBridge.getInstance();
        }

        if((usageCount < 0) && !name.startsWith("_")) {
            System.err.println("[Error] Constant tensor must start with '_'. Given name: " + name);
            return CuBridge.getInstance();
        }

        if (name == null || name.isEmpty()) {
            System.err.println("Error: Tensor name must be defined.");
            return CuBridge.getInstance();
        }

        if (!CuBridgeJNI.put(data.toArray(), data.getShape(), data.getSize(), data.getAxis(), usageCount, name,
                broadcast))
            System.err.println("Error: Tensor name is duplicated. Please choose another name.");

        return CuBridge.getInstance();
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
            return CuBridge.getInstance();
        }

        if (!CuBridgeJNI.duple(name, usageCount))
            System.err.println("Error: Failed to update usage count for tensor '" + name + "' in the queue.");

        return CuBridge.getInstance();
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
        return CuBridge.getInstance();
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
        return CuBridge.getInstance();
    }
}
