package CuBridge;

import java.util.UUID;

public interface AlgebraOps {

    private String genRandomNameAlgebra() {
        return "AlgebraOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /**
     * **L2Normalize — Basic L2 vector normalization with empty tensor reference**
     *
     * Computes the L2-normalized unit vector of the specified tensor
     * using an automatically assigned output name when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the L2 normalization operation.
     * @see #l2normalize(String, String)
     */
    default CuBridge l2normalize() {
        return l2normalize("", genRandomNameAlgebra());
    }

    /**
     * **L2Normalize — Core normalization operation**
     *
     * Normalizes the input tensor to have a unit L2 norm (‖x‖₂ = 1).
     * <p>
     * Each element is divided by the L2 norm of the tensor:
     * <pre>
     * out = a / sqrt(sum(a_i^2))
     * </pre>
     * If the norm is zero, the output will be a zero tensor.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the normalized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge l2normalize(String a, String out) {
        if (CuBridgeJNI.l2normalize(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | l2normalize | " + a + " | " + out);
        return null;
    }

    /**
     * **L2Normalize — Overload using a Tensor object**
     *
     * Normalizes the given {@link Tensor} object to have a unit L2 norm.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #l2normalize(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the normalized output tensor.
     * @return A {@link CuBridge} instance representing the normalization operation.
     * @see #l2normalize(String, String)
     */
    default CuBridge l2normalize(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return l2normalize(aName, out);
    }

    /**
     * **L2NormalizeI — Immediate normalization with empty tensor reference**
     *
     * Computes the L2-normalized version of a tensor in the internal queue
     * and directly returns the resulting {@link Tensor}.
     *
     * @return A {@link Tensor} representing the L2-normalized tensor.
     * @see #l2normalize(String, String)
     */
    default Tensor l2normalizeI() {
        String oName = genRandomNameAlgebra();
        return l2normalize("", oName).get(oName);
    }

    /**
     * **L2NormalizeI — Immediate normalization on a named tensor**
     *
     * Normalizes the specified named tensor to have a unit L2 norm
     * and directly returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} representing the L2-normalized tensor.
     * @see #l2normalize(String, String)
     */
    default Tensor l2normalizeI(String a) {
        String oName = genRandomNameAlgebra();
        return l2normalize(a, oName).get(oName);
    }

    /**
     * **L2NormalizeI — Immediate normalization on a Tensor object**
     *
     * Normalizes the given {@link Tensor} object to have a unit L2 norm
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #l2normalize(String, String)}.
     * </p>
     *
     * @param a The input tensor object.
     * @return A {@link Tensor} representing the L2-normalized tensor.
     * @see #l2normalize(String, String)
     */
    default Tensor l2normalizeI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return l2normalize(aName, oName).get(oName);
    }


    /**
     * **Dot — Basic dot product with empty tensor references**
     *
     * Performs a dot product using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the dot operation.
     * @see #dot(String, String, String)
     */
    default CuBridge dot() {
        return dot("", "", genRandomNameAlgebra());
    }

    /**
     * **Dot — Overload using a Tensor object as the first operand**
     *
     * Performs a dot product between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #dot(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the dot operation.
     * @see #dot(String, String, String)
     */
    default CuBridge dot(Tensor a, String b, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return dot(aName, b, out);
    }

    /**
     * **Dot — Overload using a Tensor object as the second operand**
     *
     * Performs a dot product between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #dot(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the dot operation.
     * @see #dot(String, String, String)
     */
    default CuBridge dot(String a, Tensor b, String out) {
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        return dot(a, bName, out);
    }

    /**
     * **Dot — Overload using two Tensor objects as operands**
     *
     * Performs a dot product between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #dot(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the dot operation.
     * @see #dot(String, String, String)
     */
    default CuBridge dot(Tensor a, Tensor b, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        return dot(aName, bName, out);
    }

    /**
     * **Dot Product — Inner product of two tensors**
     *
     * Performs a dot product (inner product) between two tensors.
     * <p>
     * Automatically adapts for vector or matrix multiplication based on input shape.
     * </p>
     *
     * @param a   The name of the first tensor.
     * @param b   The name of the second tensor.
     * @param out The name to store the resulting tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge dot(String a, String b, String out) {
        if (CuBridgeJNI.dot(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | dot | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **DotI — Immediate dot product with empty tensor references**
     *
     * Performs a dot product using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the dot product.
     * @see #dot(String, String, String)
     */
    default Tensor dotI() {
        String oName = genRandomNameAlgebra();
        return dot("", "", oName).get(oName);
    }

    /**
     * **DotI — Immediate dot product between two named tensors**
     *
     * Performs a dot product between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #dot(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the dot product.
     * @see #dot(String, String, String)
     */
    default Tensor dotI(String a, String b){
        String oName = genRandomNameAlgebra();
        return dot(a, b, oName).get(oName);
    }

    /**
     * **DotI — Immediate dot product with a Tensor and a named operand**
     *
     * Performs a dot product between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #dot(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the dot product.
     * @see #dot(String, String, String)
     */
    default Tensor dotI(Tensor a, String b) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return dot(aName, b, oName).get(oName);
    }

    /**
     * **DotI — Immediate dot product with a named and a Tensor operand**
     *
     * Performs a dot product between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #dot(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the dot product.
     * @see #dot(String, String, String)
     */
    default Tensor dotI(String a, Tensor b) {
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameAlgebra();
        return dot(a, bName, oName).get(oName);
    }

    /**
     * **DotI — Immediate dot product between two Tensor objects**
     *
     * Performs a dot product (inner product) between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #dot(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing the dot product result.
     * @see #dot(String, String, String)
     */
    default Tensor dotI(Tensor a, Tensor b) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameAlgebra();
        return dot(aName, bName, oName).get(oName);
    }


    /**
     * **MatMul — Basic matrix multiplication with empty tensor references**
     *
     * Performs matrix multiplication using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default CuBridge matmul() {
        return matmul("", "", genRandomNameAlgebra());
    }

    /**
     * **MatMul — Overload using a Tensor object as the first operand**
     *
     * Performs matrix multiplication between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #matmul(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default CuBridge matmul(Tensor a, String b, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return matmul(aName, b, out);
    }

    /**
     * **MatMul — Overload using a Tensor object as the second operand**
     *
     * Performs matrix multiplication between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #matmul(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default CuBridge matmul(String a, Tensor b, String out) {
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        return matmul(a, bName, out);
    }

    /**
     * **MatMul — Overload using two Tensor objects as operands**
     *
     * Performs matrix multiplication between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #matmul(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default CuBridge matmul(Tensor a, Tensor b, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        return matmul(aName, bName, out);
    }

    /**
     * **MatMul — Core matrix multiplication operation**
     *
     * Performs matrix multiplication between two tensors.
     * <p>
     * Automatically adapts to tensor dimensions and supports batched multiplication.
     * </p>
     *
     * @param a   The name of the first tensor.
     * @param b   The name of the second tensor.
     * @param out The name to store the resulting tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge matmul(String a, String b, String out) {
        if (CuBridgeJNI.matmul(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | matmul | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **MatMulI — Immediate matrix multiplication with empty tensor references**
     *
     * Performs matrix multiplication using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default Tensor matmulI() {
        String oName = genRandomNameAlgebra();
        return matmul("", "", oName).get(oName);
    }

    /**
     * **MatMulI — Immediate matrix multiplication between two named tensors**
     *
     * Performs matrix multiplication between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #matmul(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default Tensor matmulI(String a, String b) {
        String oName = genRandomNameAlgebra();
        return matmul(a, b, oName).get(oName);
    }

    /**
     * **MatMulI — Immediate matrix multiplication with a Tensor and a named operand**
     *
     * Performs matrix multiplication between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #matmul(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default Tensor matmulI(Tensor a, String b) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return matmul(aName, b, oName).get(oName);
    }

    /**
     * **MatMulI — Immediate matrix multiplication with a named and a Tensor operand**
     *
     * Performs matrix multiplication between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #matmul(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default Tensor matmulI(String a, Tensor b) {
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameAlgebra();
        return matmul(a, bName, oName).get(oName);
    }

    /**
     * **MatMulI — Immediate matrix multiplication between two Tensor objects**
     *
     * Performs matrix multiplication between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #matmul(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing the result of the matrix multiplication.
     * @see #matmul(String, String, String)
     */
    default Tensor matmulI(Tensor a, Tensor b) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameAlgebra();
        return matmul(aName, bName, oName).get(oName);
    }


    /**
     * **Transpose — Basic transpose with empty tensor reference**
     *
     * Performs a transpose operation using the default axes (0, -1),
     * representing the last two axes of the tensor,
     * when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the transpose operation.
     * @see #transpose(String, String, int, int)
     */
    default CuBridge transpose() {
        return transpose("", genRandomNameAlgebra(), 0, -1);
    }

    /**
     * **Transpose — Overload with specified axes and empty tensor reference**
     *
     * Performs a transpose operation on the tensor currently stored in the queue,
     * swapping the specified axes.
     *
     * @param axis1 The first axis to swap.
     * @param axis2 The second axis to swap.
     * @return A {@link CuBridge} instance representing the transpose operation.
     * @see #transpose(String, String, int, int)
     */
    default CuBridge transpose(int axis1, int axis2) {
        return transpose("", genRandomNameAlgebra(), axis1, axis2);
    }

    /**
     * **Transpose — Overload using a Tensor object and output name**
     *
     * Performs a transpose operation on the given {@link Tensor} object
     * using the default axes (0, -1), representing the last two axes of the tensor.
     *
     * @param a   The input tensor to transpose.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the transpose operation.
     * @see #transpose(String, String, int, int)
     */
    default CuBridge transpose(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return transpose(aName, out, 0, -1);
    }

    /**
     * **Transpose — Overload using a Tensor object with specified axes**
     *
     * Performs a transpose operation on the given {@link Tensor} object,
     * swapping the specified axes.
     *
     * @param a     The input tensor to transpose.
     * @param out   The name to store the resulting tensor.
     * @param axis1 The first axis to swap.
     * @param axis2 The second axis to swap.
     * @return A {@link CuBridge} instance representing the transpose operation.
     * @see #transpose(String, String, int, int)
     */
    default CuBridge transpose(Tensor a, String out, int axis1, int axis2) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return transpose(aName, out, axis1, axis2);
    }

    /**
     * **Transpose — Core transpose operation**
     *
     * Performs a transpose operation on a tensor by swapping two specified axes.
     * <p>
     * When {@code axis1 = 0} and {@code axis2 = -1}, the operation transposes
     * the last two axes of the tensor by default.
     * </p>
     *
     * @param name  The name of the input tensor.
     * @param out   The name to store the transposed tensor.
     * @param axis1 The first axis to swap.
     * @param axis2 The second axis to swap.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge transpose(String name, String out, int axis1, int axis2) {
        if (CuBridgeJNI.transpose(name, out, axis1, axis2)) return CuBridge.getInstance();
        else System.err.println("Error | transpose | " + name + " | " + out + " | " + axis1 + " | " + axis2);
        return null;
    }

    /**
     * **TransposeI — Immediate transpose with default axes**
     *
     * Performs a transpose operation using the default axes (0, -1),
     * representing the last two axes of the tensor,
     * and directly returns the resulting {@link Tensor}.
     *
     * @return A {@link Tensor} containing the transposed result.
     * @see #transpose(String, String, int, int)
     */
    default Tensor transposeI() {
        String oName = genRandomNameAlgebra();
        return transpose("", oName, 0, -1).get(oName);
    }

    /**
     * **TransposeI — Immediate transpose with specified axes**
     *
     * Performs a transpose operation by swapping the specified axes
     * of the tensor currently stored in the internal queue.
     *
     * @param axis1 The first axis to swap.
     * @param axis2 The second axis to swap.
     * @return A {@link Tensor} containing the transposed result.
     * @see #transpose(String, String, int, int)
     */
    default Tensor transposeI(int axis1, int axis2) {
        String oName = genRandomNameAlgebra();
        return transpose("", oName, axis1, axis2).get(oName);
    }

    /**
     * **TransposeI — Immediate transpose of a named tensor**
     *
     * Performs a transpose operation on a named tensor
     * using the default axes (0, -1), representing the last two axes of the tensor.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the transposed result.
     * @see #transpose(String, String, int, int)
     */
    default Tensor transposeI(String a) {
        String oName = genRandomNameAlgebra();
        return transpose(a, oName, 0, -1).get(oName);
    }

    /**
     * **TransposeI — Immediate transpose of a Tensor object**
     *
     * Performs a transpose operation on the given {@link Tensor} object
     * using the default axes (0, -1), representing the last two axes of the tensor.
     *
     * @param a The input tensor to transpose.
     * @return A {@link Tensor} containing the transposed result.
     * @see #transpose(String, String, int, int)
     */
    default Tensor transposeI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return transpose(aName, oName, 0, -1).get(oName);
    }

    /**
     * **TransposeI — Immediate transpose of a named tensor with specified axes**
     *
     * Performs a transpose operation on a named tensor,
     * swapping the specified axes.
     *
     * @param a     The name of the input tensor.
     * @param axis1 The first axis to swap.
     * @param axis2 The second axis to swap.
     * @return A {@link Tensor} containing the transposed result.
     * @see #transpose(String, String, int, int)
     */
    default Tensor transposeI(String a, int axis1, int axis2) {
        String oName = genRandomNameAlgebra();
        return transpose(a, oName, axis1, axis2).get(oName);
    }

    /**
     * **TransposeI — Immediate transpose of a Tensor object with specified axes**
     *
     * Performs a transpose operation on the given {@link Tensor} object,
     * swapping the specified axes.
     *
     * @param a     The input tensor to transpose.
     * @param axis1 The first axis to swap.
     * @param axis2 The second axis to swap.
     * @return A {@link Tensor} containing the transposed result.
     * @see #transpose(String, String, int, int)
     */
    default Tensor transposeI(Tensor a, int axis1, int axis2) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return transpose(aName, oName, axis1, axis2).get(oName);
    }


    /**
     * **Trace — Basic trace operation with empty tensor reference**
     *
     * Calculates the trace (sum of diagonal elements) of a square matrix
     * using an automatically assigned output name when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the trace operation.
     * @see #trace(String, String)
     */
    default CuBridge trace() {
        return trace("", genRandomNameAlgebra());
    }

    /**
     * **Trace — Core trace operation**
     *
     * Calculates the trace (sum of diagonal elements) of a square matrix tensor.
     * <p>
     * The result is a scalar tensor containing the sum of all diagonal elements.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge trace(String a, String out) {
        if (CuBridgeJNI.trace(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | trace | " + a + " | " + out);
        return null;
    }

    /**
     * **Trace — Overload using a Tensor object**
     *
     * Calculates the trace (sum of diagonal elements) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #trace(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the trace operation.
     * @see #trace(String, String)
     */
    default CuBridge trace(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return trace(aName, out);
    }

    /**
     * **TraceI — Immediate trace operation with empty tensor reference**
     *
     * Calculates the trace (sum of diagonal elements) of a square matrix
     * using an automatically assigned output name and directly returns the resulting {@link Tensor}.
     *
     * @return A {@link Tensor} representing the scalar trace value.
     * @see #trace(String, String)
     */
    default Tensor traceI() {
        String oName = genRandomNameAlgebra();
        return trace("", oName).get(oName);
    }

    /**
     * **TraceI — Immediate trace operation on a named tensor**
     *
     * Calculates the trace (sum of diagonal elements) of a named tensor
     * and directly returns the resulting scalar {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} representing the scalar trace value.
     * @see #trace(String, String)
     */
    default Tensor traceI(String a) {
        String oName = genRandomNameAlgebra();
        return trace(a, oName).get(oName);
    }

    /**
     * **TraceI — Immediate trace operation on a Tensor object**
     *
     * Calculates the trace (sum of diagonal elements) of the given {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #trace(String, String)}.
     * </p>
     *
     * @param a The input tensor object.
     * @return A {@link Tensor} representing the scalar trace value.
     * @see #trace(String, String)
     */
    default Tensor traceI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return trace(aName, oName).get(oName);
    }


    /**
     * **Inverse — Basic matrix inversion with empty tensor reference**
     *
     * Computes the inverse of a square matrix tensor using numerical decomposition
     * and an automatically assigned output name when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the inverse operation.
     * @see #inverse(String, String)
     */
    default CuBridge inverse() {
        return inverse("", genRandomNameAlgebra());
    }

    /**
     * **Inverse — Core matrix inversion operation**
     *
     * Computes the inverse of a square matrix tensor using numerical decomposition.
     * <p>
     * The input tensor must be a non-singular square matrix.
     * </p>
     *
     * @param a   The name of the input tensor (must be square).
     * @param out The name to store the inverted tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge inverse(String a, String out) {
        if (CuBridgeJNI.inverse(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | inverse | " + a + " | " + out);
        return null;
    }

    /**
     * **Inverse — Overload using a Tensor object**
     *
     * Computes the inverse of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #inverse(String, String)}.
     * </p>
     *
     * @param a   The input square matrix tensor.
     * @param out The name to store the inverted tensor.
     * @return A {@link CuBridge} instance representing the inverse operation.
     * @see #inverse(String, String)
     */
    default CuBridge inverse(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return inverse(aName, out);
    }

    /**
     * **InverseI — Immediate matrix inversion with empty tensor reference**
     *
     * Computes the inverse of a square matrix tensor using an automatically assigned
     * output name and directly returns the resulting {@link Tensor}.
     *
     * @return A {@link Tensor} representing the inverted matrix.
     * @see #inverse(String, String)
     */
    default Tensor inverseI() {
        String oName = genRandomNameAlgebra();
        return inverse("", oName).get(oName);
    }

    /**
     * **InverseI — Immediate matrix inversion on a named tensor**
     *
     * Computes the inverse of a named square matrix tensor
     * and directly returns the resulting {@link Tensor}.
     *
     * @param a The name of the input square matrix tensor.
     * @return A {@link Tensor} representing the inverted matrix.
     * @see #inverse(String, String)
     */
    default Tensor inverseI(String a) {
        String oName = genRandomNameAlgebra();
        return inverse(a, oName).get(oName);
    }

    /**
     * **InverseI — Immediate matrix inversion on a Tensor object**
     *
     * Computes the inverse of the given {@link Tensor} object
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #inverse(String, String)}.
     * </p>
     *
     * @param a The input square matrix tensor.
     * @return A {@link Tensor} representing the inverted matrix.
     * @see #inverse(String, String)
     */
    default Tensor inverseI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return inverse(aName, oName).get(oName);
    }


    /**
     * **Determinant — Basic determinant calculation with empty tensor reference**
     *
     * Computes the determinant of a square matrix tensor
     * using an automatically assigned output name when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the determinant operation.
     * @see #det(String, String)
     */
    default CuBridge det() {
        return det("", genRandomNameAlgebra());
    }

    /**
     * **Determinant — Core determinant operation**
     *
     * Computes the determinant of a square matrix tensor.
     * <p>
     * The result is a scalar tensor representing the determinant value.
     * </p>
     *
     * @param a   The name of the input tensor (must be square).
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge det(String a, String out) {
        if (CuBridgeJNI.det(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | det | " + a + " | " + out);
        return null;
    }

    /**
     * **Determinant — Overload using a Tensor object**
     *
     * Computes the determinant of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #det(String, String)}.
     * </p>
     *
     * @param a   The input square matrix tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the determinant operation.
     * @see #det(String, String)
     */
    default CuBridge det(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return det(aName, out);
    }

    /**
     * **DetI — Immediate determinant calculation with empty tensor reference**
     *
     * Computes the determinant of a square matrix tensor
     * using an automatically assigned output name and directly returns the result.
     *
     * @return A scalar {@link Tensor} containing the determinant value.
     * @see #det(String, String)
     */
    default Tensor detI() {
        String oName = genRandomNameAlgebra();
        return det("", oName).get(oName);
    }

    /**
     * **DetI — Immediate determinant calculation on a named tensor**
     *
     * Computes the determinant of a named square matrix tensor
     * and directly returns the resulting scalar {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A scalar {@link Tensor} containing the determinant value.
     * @see #det(String, String)
     */
    default Tensor detI(String a) {
        String oName = genRandomNameAlgebra();
        return det(a, oName).get(oName);
    }

    /**
     * **DetI — Immediate determinant calculation on a Tensor object**
     *
     * Computes the determinant of the given {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #det(String, String)}.
     * </p>
     *
     * @param a The input square matrix tensor.
     * @return A scalar {@link Tensor} containing the determinant value.
     * @see #det(String, String)
     */
    default Tensor detI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return det(aName, oName).get(oName);
    }


    /**
     * **Eigen — Eigenvalue and eigenvector decomposition**
     *
     * Performs eigen decomposition on a square matrix tensor.
     * <p>
     * Decomposes the input matrix {@code A} into its eigenvalues and eigenvectors:
     * {@code A * v = λ * v}, where:
     * <ul>
     *   <li>{@code λ} — eigenvalues (diagonal matrix or vector)</li>
     *   <li>{@code v} — eigenvectors (column-wise)</li>
     * </ul>
     * Suitable for symmetric or diagonalizable matrices.
     * </p>
     *
     * @param a      The name of the input matrix tensor.
     * @param outVal The name to store the eigenvalues tensor.
     * @param outVec The name to store the eigenvectors tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge eigen(String a, String outVal, String outVec) {
        if (CuBridgeJNI.eigen(a, outVal, outVec)) return CuBridge.getInstance();
        else System.err.println("Error | eigen | " + a + " | " + outVal + " | " + outVec);
        return null;
    }

    /**
     * **Eigen — Overload using a Tensor object**
     *
     * Performs eigen decomposition on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #eigen(String, String, String)}.
     * </p>
     *
     * @param a      The input square matrix tensor.
     * @param outVal The name to store the eigenvalues tensor.
     * @param outVec The name to store the eigenvectors tensor.
     * @return A {@link CuBridge} instance representing the eigen decomposition operation.
     * @see #eigen(String, String, String)
     */
    default CuBridge eigen(Tensor a, String outVal, String outVec) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return eigen(aName, outVal, outVec);
    }


    /**
     * **QR — QR decomposition**
     *
     * Performs QR decomposition on a matrix tensor.
     * <p>
     * Decomposes the input matrix {@code A} into orthogonal and upper-triangular matrices:
     * {@code A = Q * R}, where:
     * <ul>
     *   <li>{@code Q} — orthogonal matrix</li>
     *   <li>{@code R} — upper-triangular matrix</li>
     * </ul>
     * </p>
     *
     * @param a    The name of the input matrix tensor.
     * @param outQ The name to store the orthogonal matrix Q.
     * @param outR The name to store the upper-triangular matrix R.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge qr(String a, String outQ, String outR) {
        if (CuBridgeJNI.qr(a, outQ, outR)) return CuBridge.getInstance();
        else System.err.println("Error | qr | " + a + " | " + outQ + " | " + outR);
        return null;
    }

    /**
     * **QR — Overload using a Tensor object**
     *
     * Performs QR decomposition on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #qr(String, String, String)}.
     * </p>
     *
     * @param a    The input matrix tensor.
     * @param outQ The name to store the orthogonal matrix Q.
     * @param outR The name to store the upper-triangular matrix R.
     * @return A {@link CuBridge} instance representing the QR decomposition operation.
     * @see #qr(String, String, String)
     */
    default CuBridge qr(Tensor a, String outQ, String outR) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return qr(aName, outQ, outR);
    }


    /**
     * **SVD — Singular Value Decomposition**
     *
     * Performs Singular Value Decomposition on a matrix tensor.
     * <p>
     * Decomposes the input matrix {@code A} into three tensors:
     * {@code A = U * S * Vᵀ}, where:
     * <ul>
     *   <li>{@code U} — left singular vectors</li>
     *   <li>{@code S} — singular values (diagonal matrix)</li>
     *   <li>{@code Vᵀ} — right singular vectors</li>
     * </ul>
     * All output tensor names must be explicitly specified.
     * </p>
     *
     * @param a     The name of the input matrix tensor.
     * @param outU  The name to store the left singular matrix U.
     * @param outS  The name to store the singular values matrix S.
     * @param outVT The name to store the right singular matrix Vᵀ.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge svd(String a, String outU, String outS, String outVT) {
        if (CuBridgeJNI.svd(a, outU, outS, outVT)) return CuBridge.getInstance();
        else System.err.println("Error | svd | " + a + " | " + outU + " | " + outS + " | " + outVT);
        return null;
    }

    /**
     * **SVD — Overload using a Tensor object**
     *
     * Performs Singular Value Decomposition on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #svd(String, String, String, String)}.
     * </p>
     *
     * @param a     The input matrix tensor.
     * @param outU  The name to store the left singular matrix U.
     * @param outS  The name to store the singular values matrix S.
     * @param outVT The name to store the right singular matrix Vᵀ.
     * @return A {@link CuBridge} instance representing the SVD operation.
     * @see #svd(String, String, String, String)
     */
    default CuBridge svd(Tensor a, String outU, String outS, String outVT) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return svd(aName, outU, outS, outVT);
    }


    /**
     * **Cholesky — Basic Cholesky decomposition with empty tensor reference**
     *
     * Performs Cholesky decomposition on a symmetric positive-definite matrix
     * using an automatically assigned output name when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the Cholesky decomposition operation.
     * @see #cholesky(String, String)
     */
    default CuBridge cholesky() {
        return cholesky("", genRandomNameAlgebra());
    }

    /**
     * **Cholesky — Core Cholesky decomposition operation**
     *
     * Performs Cholesky decomposition on a symmetric positive-definite matrix.
     * <p>
     * The result is the lower-triangular matrix {@code L} such that {@code A = L * Lᵀ}.
     * </p>
     *
     * @param a   The name of the input symmetric positive-definite matrix tensor.
     * @param out The name to store the resulting lower-triangular matrix (L).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge cholesky(String a, String out) {
        if (CuBridgeJNI.cholesky(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | cholesky | " + a + " | " + out);
        return null;
    }

    /**
     * **Cholesky — Overload using a Tensor object**
     *
     * Performs Cholesky decomposition on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #cholesky(String, String)}.
     * </p>
     *
     * @param a   The input symmetric positive-definite matrix tensor.
     * @param out The name to store the resulting lower-triangular matrix (L).
     * @return A {@link CuBridge} instance representing the Cholesky decomposition operation.
     * @see #cholesky(String, String)
     */
    default CuBridge cholesky(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return cholesky(aName, out);
    }

    /**
     * **CholeskyI — Immediate Cholesky decomposition with empty tensor reference**
     *
     * Performs Cholesky decomposition on a symmetric positive-definite matrix tensor
     * using an automatically assigned output name and directly returns the result.
     *
     * @return The lower-triangular {@link Tensor} obtained from Cholesky decomposition.
     * @see #cholesky(String, String)
     */
    default Tensor choleskyI() {
        String oName = genRandomNameAlgebra();
        return cholesky("", oName).get(oName);
    }

    /**
     * **CholeskyI — Immediate Cholesky decomposition on a named tensor**
     *
     * Performs Cholesky decomposition on a named symmetric positive-definite matrix tensor
     * and directly returns the resulting lower-triangular matrix.
     *
     * @param a The name of the input matrix tensor.
     * @return The lower-triangular {@link Tensor} obtained from Cholesky decomposition.
     * @see #cholesky(String, String)
     */
    default Tensor choleskyI(String a) {
        String oName = genRandomNameAlgebra();
        return cholesky(a, oName).get(oName);
    }

    /**
     * **CholeskyI — Immediate Cholesky decomposition on a Tensor object**
     *
     * Performs Cholesky decomposition on the given {@link Tensor} object
     * and directly returns the resulting lower-triangular matrix tensor.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #cholesky(String, String)}.
     * </p>
     *
     * @param a The input symmetric positive-definite matrix tensor.
     * @return The lower-triangular {@link Tensor} obtained from Cholesky decomposition.
     * @see #cholesky(String, String)
     */
    default Tensor choleskyI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return cholesky(aName, oName).get(oName);
    }


    /**
     * **Rank — Basic matrix rank evaluation with empty tensor reference**
     *
     * Calculates the rank (number of linearly independent rows or columns) of a matrix
     * using an automatically assigned output name when no input tensor is specified.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the rank evaluation operation.
     * @see #rank(String, String)
     */
    default CuBridge rank() {
        return rank("", genRandomNameAlgebra());
    }

    /**
     * **Rank — Core matrix rank evaluation operation**
     *
     * Calculates the rank (number of linearly independent rows or columns) of a matrix tensor.
     * <p>
     * The result is a scalar tensor containing the rank value.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge rank(String a, String out) {
        if (CuBridgeJNI.rank(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | rank | " + a + " | " + out);
        return null;
    }

    /**
     * **Rank — Overload using a Tensor object**
     *
     * Calculates the rank (number of linearly independent rows or columns)
     * of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #rank(String, String)}.
     * </p>
     *
     * @param a   The input matrix tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the rank evaluation operation.
     * @see #rank(String, String)
     */
    default CuBridge rank(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return rank(aName, out);
    }

    /**
     * **RankI — Immediate matrix rank evaluation with empty tensor reference**
     *
     * Calculates the rank (number of linearly independent rows or columns) of a matrix tensor
     * using an automatically assigned output name and directly returns the resulting scalar tensor.
     *
     * @return A scalar {@link Tensor} containing the rank value.
     * @see #rank(String, String)
     */
    default Tensor rankI() {
        String oName = genRandomNameAlgebra();
        return rank("", oName).get(oName);
    }

    /**
     * **RankI — Immediate matrix rank evaluation on a named tensor**
     *
     * Calculates the rank (number of linearly independent rows or columns) of a named matrix tensor
     * and directly returns the resulting scalar tensor.
     *
     * @param a The name of the input tensor.
     * @return A scalar {@link Tensor} containing the rank value.
     * @see #rank(String, String)
     */
    default Tensor rankI(String a) {
        String oName = genRandomNameAlgebra();
        return rank(a, oName).get(oName);
    }

    /**
     * **RankI — Immediate matrix rank evaluation on a Tensor object**
     *
     * Calculates the rank (number of linearly independent rows or columns)
     * of the given {@link Tensor} object and directly returns the resulting scalar tensor.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #rank(String, String)}.
     * </p>
     *
     * @param a The input matrix tensor.
     * @return A scalar {@link Tensor} containing the rank value.
     * @see #rank(String, String)
     */
    default Tensor rankI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return rank(aName, oName).get(oName);
    }


    /**
     * **Normalize — Basic normalization with empty tensor reference**
     *
     * Normalizes the elements of a tensor along the first axis (when {@code axis = -1}).
     * <p>
     * This operation scales the values of the tensor so that they have unit norm
     * along the specified axis.
     * </p>
     * Typically used when a tensor is already stored in the internal queue.
     *
     * @return A {@link CuBridge} instance representing the normalization operation.
     * @see #normalize(String, String, int)
     */
    default CuBridge normalize() {
        return normalize("", genRandomNameAlgebra(), -1);
    }

    /**
     * **Normalize — Normalization along a specified axis**
     *
     * Normalizes the elements of a tensor along the specified axis.
     * <p>
     * When {@code axis = -1}, normalization is performed along the first axis.
     * </p>
     *
     * @param axis The axis along which to perform normalization. Use {@code -1} for the first axis.
     * @return A {@link CuBridge} instance representing the normalization operation.
     * @see #normalize(String, String, int)
     */
    default CuBridge normalize(int axis) {
        return normalize("", genRandomNameAlgebra(), axis);
    }

    /**
     * **Normalize — Overload using a Tensor object**
     *
     * Normalizes the given {@link Tensor} object along the first axis (when {@code axis = -1}).
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #normalize(String, String, int)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting normalized tensor.
     * @return A {@link CuBridge} instance representing the normalization operation.
     * @see #normalize(String, String, int)
     */
    default CuBridge normalize(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return normalize(aName, out, -1);
    }

    /**
     * **Normalize — Overload using a Tensor object with specified axis**
     *
     * Normalizes the given {@link Tensor} object along the specified axis.
     * <p>
     * When {@code axis = -1}, normalization is performed along the first axis.
     * </p>
     *
     * @param a    The input tensor.
     * @param out  The name to store the resulting normalized tensor.
     * @param axis The axis along which to perform normalization. Use {@code -1} for the first axis.
     * @return A {@link CuBridge} instance representing the normalization operation.
     * @see #normalize(String, String, int)
     */
    default CuBridge normalize(Tensor a, String out, int axis) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return normalize(aName, out, axis);
    }

    /**
     * **Normalize — Overload using named tensors**
     *
     * Normalizes the elements of the named input tensor along the first axis (when {@code axis = -1}).
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting normalized tensor.
     * @return A {@link CuBridge} instance representing the normalization operation.
     * @see #normalize(String, String, int)
     */
    default CuBridge normalize(String a, String out) {
        return normalize(a, out, -1);
    }

    /**
     * **Normalize — Core normalization operation**
     *
     * Normalizes the elements of a tensor along the specified axis.
     * <p>
     * When {@code axis = -1}, normalization is performed along the first axis.
     * </p>
     *
     * @param a    The name of the input tensor.
     * @param out  The name to store the resulting normalized tensor.
     * @param axis The axis along which to perform normalization. Use {@code -1} for the first axis.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge normalize(String a, String out, int axis) {
        if (CuBridgeJNI.normalize(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | normalize | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **NormalizeI — Immediate normalization with default axis (-1)**
     *
     * Normalizes the tensor along the first axis (when {@code axis = -1})
     * and directly returns the resulting normalized {@link Tensor}.
     *
     * @return A normalized {@link Tensor}.
     * @see #normalize(String, String, int)
     */
    default Tensor normalizeI() {
        String oName = genRandomNameAlgebra();
        return normalize("", oName, -1).get(oName);
    }

    /**
     * **NormalizeI — Immediate normalization along a specified axis**
     *
     * Normalizes the tensor along the specified axis
     * and directly returns the resulting normalized {@link Tensor}.
     *
     * @param axis The axis along which to perform normalization. Use {@code -1} for the first axis.
     * @return A normalized {@link Tensor}.
     * @see #normalize(String, String, int)
     */
    default Tensor normalizeI(int axis) {
        String oName = genRandomNameAlgebra();
        return normalize("", oName, axis).get(oName);
    }

    /**
     * **NormalizeI — Immediate normalization of a named tensor**
     *
     * Normalizes the named tensor along the first axis (when {@code axis = -1})
     * and directly returns the resulting normalized {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A normalized {@link Tensor}.
     * @see #normalize(String, String, int)
     */
    default Tensor normalizeI(String a) {
        String oName = genRandomNameAlgebra();
        return normalize(a, oName, -1).get(oName);
    }

    /**
     * **NormalizeI — Immediate normalization of a named tensor along a specified axis**
     *
     * Normalizes the named tensor along the specified axis
     * and directly returns the resulting normalized {@link Tensor}.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which to perform normalization. Use {@code -1} for the first axis.
     * @return A normalized {@link Tensor}.
     * @see #normalize(String, String, int)
     */
    default Tensor normalizeI(String a, int axis) {
        String oName = genRandomNameAlgebra();
        return normalize(a, oName, axis).get(oName);
    }

    /**
     * **NormalizeI — Immediate normalization of a Tensor object**
     *
     * Normalizes the given {@link Tensor} object along the first axis (when {@code axis = -1})
     * and directly returns the resulting normalized {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #normalize(String, String, int)}.
     * </p>
     *
     * @param a The input tensor to normalize.
     * @return A normalized {@link Tensor}.
     * @see #normalize(String, String, int)
     */
    default Tensor normalizeI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return normalize(aName, oName, -1).get(oName);
    }

    /**
     * **NormalizeI — Immediate normalization of a Tensor object along a specified axis**
     *
     * Normalizes the given {@link Tensor} object along the specified axis
     * and directly returns the resulting normalized {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #normalize(String, String, int)}.
     * </p>
     *
     * @param a    The input tensor to normalize.
     * @param axis The axis along which to perform normalization. Use {@code -1} for the first axis.
     * @return A normalized {@link Tensor}.
     * @see #normalize(String, String, int)
     */
    default Tensor normalizeI(Tensor a, int axis) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return normalize(aName, oName, axis).get(oName);
    }


    /**
     * **Standardize — Basic standardization with empty tensor reference**
     *
     * Standardizes the elements of a tensor along the first axis (when {@code axis = -1}).
     * <p>
     * Each element is scaled by subtracting the mean and dividing by the standard deviation
     * computed along the specified axis.
     * </p>
     * Typically used when a tensor is already stored in the internal queue.
     *
     * @return A {@link CuBridge} instance representing the standardization operation.
     * @see #standardize(String, String, int)
     */
    default CuBridge standardize() {
        return standardize("", genRandomNameAlgebra(), -1);
    }

    /**
     * **Standardize — Standardization along a specified axis**
     *
     * Standardizes the elements of a tensor along the specified axis.
     * <p>
     * When {@code axis = -1}, standardization is performed along the first axis.
     * </p>
     *
     * @param axis The axis along which standardization is performed. Use {@code -1} for the first axis.
     * @return A {@link CuBridge} instance representing the standardization operation.
     * @see #standardize(String, String, int)
     */
    default CuBridge standardize(int axis) {
        return standardize("", genRandomNameAlgebra(), axis);
    }

    /**
     * **Standardize — Overload using a Tensor object**
     *
     * Standardizes the given {@link Tensor} object along the first axis (when {@code axis = -1}).
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #standardize(String, String, int)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the standardized tensor.
     * @return A {@link CuBridge} instance representing the standardization operation.
     * @see #standardize(String, String, int)
     */
    default CuBridge standardize(Tensor a, String out) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return standardize(aName, out, -1);
    }

    /**
     * **Standardize — Overload using a Tensor object with specified axis**
     *
     * Standardizes the given {@link Tensor} object along the specified axis.
     * <p>
     * When {@code axis = -1}, standardization is performed along the first axis.
     * </p>
     *
     * @param a    The input tensor.
     * @param out  The name to store the standardized tensor.
     * @param axis The axis along which standardization is performed. Use {@code -1} for the first axis.
     * @return A {@link CuBridge} instance representing the standardization operation.
     * @see #standardize(String, String, int)
     */
    default CuBridge standardize(Tensor a, String out, int axis) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        return standardize(aName, out, axis);
    }

    /**
     * **Standardize — Overload using named tensors**
     *
     * Standardizes the elements of the named input tensor along the first axis (when {@code axis = -1}).
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the standardized tensor.
     * @return A {@link CuBridge} instance representing the standardization operation.
     * @see #standardize(String, String, int)
     */
    default CuBridge standardize(String a, String out) {
        return standardize(a, out, -1);
    }

    /**
     * **Standardize — Core standardization operation**
     *
     * Standardizes the input tensor by subtracting the mean and dividing by the standard deviation
     * along the specified axis.
     * <p>
     * When {@code axis = -1}, standardization is performed along the first axis.
     * </p>
     *
     * @param a    The name of the input tensor.
     * @param out  The name to store the standardized tensor.
     * @param axis The axis along which standardization is performed. Use {@code -1} for the first axis.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge standardize(String a, String out, int axis) {
        if (CuBridgeJNI.standardize(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | standardize | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **StandardizeI — Immediate standardization with default axis (-1)**
     *
     * Standardizes the tensor along the first axis (when {@code axis = -1})
     * and directly returns the resulting standardized {@link Tensor}.
     *
     * @return A standardized {@link Tensor}.
     * @see #standardize(String, String, int)
     */
    default Tensor standardizeI() {
        String oName = genRandomNameAlgebra();
        return standardize("", oName, -1).get(oName);
    }

    /**
     * **StandardizeI — Immediate standardization along a specified axis**
     *
     * Standardizes the tensor along the specified axis
     * and directly returns the resulting standardized {@link Tensor}.
     *
     * @param axis The axis along which standardization is performed. Use {@code -1} for the first axis.
     * @return A standardized {@link Tensor}.
     * @see #standardize(String, String, int)
     */
    default Tensor standardizeI(int axis) {
        String oName = genRandomNameAlgebra();
        return standardize("", oName, axis).get(oName);
    }

    /**
     * **StandardizeI — Immediate standardization of a named tensor**
     *
     * Standardizes the named tensor along the first axis (when {@code axis = -1})
     * and directly returns the resulting standardized {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A standardized {@link Tensor}.
     * @see #standardize(String, String, int)
     */
    default Tensor standardizeI(String a) {
        String oName = genRandomNameAlgebra();
        return standardize(a, oName, -1).get(oName);
    }

    /**
     * **StandardizeI — Immediate standardization of a named tensor along a specified axis**
     *
     * Standardizes the named tensor along the specified axis
     * and directly returns the resulting standardized {@link Tensor}.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which standardization is performed. Use {@code -1} for the first axis.
     * @return A standardized {@link Tensor}.
     * @see #standardize(String, String, int)
     */
    default Tensor standardizeI(String a, int axis) {
        String oName = genRandomNameAlgebra();
        return standardize(a, oName, axis).get(oName);
    }

    /**
     * **StandardizeI — Immediate standardization of a Tensor object**
     *
     * Standardizes the given {@link Tensor} object along the first axis (when {@code axis = -1})
     * and directly returns the resulting standardized {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #standardize(String, String, int)}.
     * </p>
     *
     * @param a The input tensor to standardize.
     * @return A standardized {@link Tensor}.
     * @see #standardize(String, String, int)
     */
    default Tensor standardizeI(Tensor a) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return standardize(aName, oName, -1).get(oName);
    }

    /**
     * **StandardizeI — Immediate standardization of a Tensor object along a specified axis**
     *
     * Standardizes the given {@link Tensor} object along the specified axis
     * and directly returns the resulting standardized {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #standardize(String, String, int)}.
     * </p>
     *
     * @param a    The input tensor to standardize.
     * @param axis The axis along which standardization is performed. Use {@code -1} for the first axis.
     * @return A standardized {@link Tensor}.
     * @see #standardize(String, String, int)
     */
    default Tensor standardizeI(Tensor a, int axis) {
        String aName = genRandomNameAlgebra(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAlgebra();
        return standardize(aName, oName, axis).get(oName);
    }


    /**
     * **Affine — Linear transformation with bias**
     *
     * Performs an affine transformation on the input tensor {@code x}
     * using weight tensor {@code w} and bias tensor {@code b},
     * and stores the result in {@code out}.
     * <p>
     * This operation computes {@code y = x·w + b}, where:
     * <ul>
     *   <li>{@code x} — input tensor</li>
     *   <li>{@code w} — weight tensor</li>
     *   <li>{@code b} — bias tensor</li>
     * </ul>
     * </p>
     *
     * @param x   The name of the input tensor.
     * @param w   The name of the weight tensor.
     * @param b   The name of the bias tensor.
     * @param out The name to store the resulting tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge affine(String x, String w, String b, String out) {
        if (CuBridgeJNI.affine(x, w, b, out))
            return CuBridge.getInstance();
        else
            System.err.println("Error | affine | " + x + " | " + w + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Affine — Overload using Tensor objects**
     *
     * Performs an affine transformation on the given {@link Tensor} objects
     * {@code x}, {@code w}, and {@code b}, and stores the result in {@code out}.
     * <p>
     * Automatically assigns random internal names to the input tensors before executing
     * {@link #affine(String, String, String, String)}.
     * </p>
     *
     * @param x   The input {@link Tensor}.
     * @param w   The weight {@link Tensor}.
     * @param b   The bias {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the affine transformation.
     * @see #affine(String, String, String, String)
     */
    default CuBridge affine(Tensor x, Tensor w, Tensor b, String out) {
        String xName = genRandomNameAlgebra(); CuBridge.getInstance().put(x, xName);
        String wName = genRandomNameAlgebra(); CuBridge.getInstance().put(w, wName);
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        return affine(xName, wName, bName, out);
    }

    /**
     * **AffineI — Immediate affine transformation**
     *
     * Immediately performs an affine transformation on the named tensors
     * {@code x}, {@code w}, and {@code b}, and directly returns the resulting {@link Tensor}.
     * <p>
     * This function executes {@code y = x·w + b} and returns the output tensor immediately,
     * without requiring manual retrieval from the CuBridge queue.
     * </p>
     *
     * @param x The name of the input tensor.
     * @param w The name of the weight tensor.
     * @param b The name of the bias tensor.
     * @return A {@link Tensor} containing the result of the affine transformation.
     * @see #affine(String, String, String, String)
     */
    default Tensor affineI(String x, String w, String b) {
        String oName = genRandomNameAlgebra();
        return affine(x, w, b, oName).get(oName);
    }

    /**
     * **AffineI — Immediate affine transformation (Tensor input)**
     *
     * Immediately performs an affine transformation on the given {@link Tensor} objects
     * {@code x}, {@code w}, and {@code b}, and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to the input tensors before executing
     * {@link #affine(String, String, String, String)}.
     * </p>
     *
     * @param x The input {@link Tensor}.
     * @param w The weight {@link Tensor}.
     * @param b The bias {@link Tensor}.
     * @return A {@link Tensor} containing the result of the affine transformation.
     * @see #affine(Tensor, Tensor, Tensor, String)
     */
    default Tensor affineI(Tensor x, Tensor w, Tensor b) {
        String xName = genRandomNameAlgebra(); CuBridge.getInstance().put(x, xName);
        String wName = genRandomNameAlgebra(); CuBridge.getInstance().put(w, wName);
        String bName = genRandomNameAlgebra(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameAlgebra();
        return affine(xName, wName, bName, oName).get(oName);
    }


    /**
     * **Softmax — Compute softmax across the entire tensor**
     *
     * Applies the softmax function to the input tensor {@code name},
     * normalizing all elements along the entire tensor (when {@code axis = -1}).
     * <p>
     * The softmax function is defined as:
     * {@code softmax(xᵢ) = exp(xᵢ) / Σ(exp(xⱼ))}.
     * </p>
     *
     * @param name The name of the input tensor.
     * @param out  The name to store the output tensor.
     * @return A {@link CuBridge} instance representing the softmax operation.
     * @see #softmax(String, String, int)
     */
    default CuBridge softmax(String name, String out) {
        return softmax(name, out, -1);
    }

    /**
     * **Softmax — Compute softmax across the entire tensor (Tensor input)**
     *
     * Applies the softmax function to the given {@link Tensor} object
     * along the entire tensor (when {@code axis = -1}).
     * <p>
     * Automatically assigns a random internal name to the input tensor before execution.
     * </p>
     *
     * @param name The input {@link Tensor}.
     * @param out  The name to store the output tensor.
     * @return A {@link CuBridge} instance representing the softmax operation.
     * @see #softmax(Tensor, String, int)
     */
    default CuBridge softmax(Tensor name, String out) {
        String n = genRandomNameAlgebra(); CuBridge.getInstance().put(name, n);
        return softmax(n, out, -1);
    }

    /**
     * **Softmax — Compute softmax along a specific axis**
     *
     * Applies the softmax function to the input tensor {@code name}
     * along the specified {@code axis}.
     * <p>
     * When {@code axis = -1}, softmax is applied across the entire tensor.
     * </p>
     *
     * @param name The name of the input tensor.
     * @param out  The name to store the output tensor.
     * @param axis The axis along which softmax is applied. Use {@code -1} for all elements.
     * @return A {@link CuBridge} instance representing the softmax operation.
     * @see #softmax(String, String)
     */
    default CuBridge softmax(String name, String out, int axis) {
        if (CuBridgeJNI.softmax(name, out, axis))
            return CuBridge.getInstance();
        else
            System.err.println("Error | softmax | " + name + " | " + out + " | axis=" + axis);
        return null;
    }

    /**
     * **Softmax — Compute softmax along a specific axis (Tensor input)**
     *
     * Applies the softmax function to the given {@link Tensor} object
     * along the specified {@code axis}.
     * <p>
     * When {@code axis = -1}, softmax is applied across the entire tensor.
     * </p>
     *
     * @param name The input {@link Tensor}.
     * @param out  The name to store the output tensor.
     * @param axis The axis along which softmax is applied. Use {@code -1} for all elements.
     * @return A {@link CuBridge} instance representing the softmax operation.
     * @see #softmax(String, String, int)
     */
    default CuBridge softmax(Tensor name, String out, int axis) {
        String n = genRandomNameAlgebra(); CuBridge.getInstance().put(name, n);
        return softmax(n, out, axis);
    }

    /**
     * **SoftmaxI — Immediate softmax across the entire tensor**
     *
     * Immediately applies the softmax function to the input tensor {@code name}
     * across all elements (when {@code axis = -1}), and directly returns
     * the resulting {@link Tensor}.
     * <p>
     * This is equivalent to calling {@link #softmax(String, String, int)} with {@code axis = -1}.
     * </p>
     *
     * @param name The name of the input tensor.
     * @return A {@link Tensor} containing the softmax-normalized result.
     * @see #softmax(String, String)
     */
    default Tensor softmaxI(String name) {
        String oName = genRandomNameAlgebra();
        return softmax(name, oName, -1).get(oName);
    }

    /**
     * **SoftmaxI — Immediate softmax across the entire tensor (Tensor input)**
     *
     * Immediately applies the softmax function to the given {@link Tensor} object
     * across all elements (when {@code axis = -1}), and directly returns
     * the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before execution.
     * </p>
     *
     * @param name The input {@link Tensor}.
     * @return A {@link Tensor} containing the softmax-normalized result.
     * @see #softmax(Tensor, String)
     */
    default Tensor softmaxI(Tensor name) {
        String n = genRandomNameAlgebra(); CuBridge.getInstance().put(name, n);
        String oName = genRandomNameAlgebra();
        return softmax(n, oName, -1).get(oName);
    }

    /**
     * **SoftmaxI — Immediate softmax along a specific axis**
     *
     * Immediately applies the softmax function to the named tensor {@code name}
     * along the specified {@code axis}, and directly returns the resulting {@link Tensor}.
     * <p>
     * When {@code axis = -1}, softmax is applied across the entire tensor.
     * </p>
     *
     * @param name The name of the input tensor.
     * @param axis The axis along which softmax is applied.
     * @return A {@link Tensor} containing the softmax-normalized result.
     * @see #softmax(String, String, int)
     */
    default Tensor softmaxI(String name, int axis) {
        String oName = genRandomNameAlgebra();
        return softmax(name, oName, axis).get(oName);
    }

    /**
     * **SoftmaxI — Immediate softmax along a specific axis (Tensor input)**
     *
     * Immediately applies the softmax function to the given {@link Tensor} object
     * along the specified {@code axis}, and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names before execution.
     * </p>
     *
     * @param name The input {@link Tensor}.
     * @param axis The axis along which softmax is applied.
     * @return A {@link Tensor} containing the softmax-normalized result.
     * @see #softmax(Tensor, String, int)
     */
    default Tensor softmaxI(Tensor name, int axis) {
        String n = genRandomNameAlgebra(); CuBridge.getInstance().put(name, n);
        String oName = genRandomNameAlgebra();
        return softmax(n, oName, axis).get(oName);
    }


}
