package CuBridge;

import java.util.UUID;

public interface AxisCascadeOps {

    private String genRandomNameAxisCascade() {
        return "AxisCascadeOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /**
     * **Sum — Basic reduction summation with empty tensor references**
     *
     * Performs a reduction summation on the most recent tensor stored in the internal queue.
     * <p>
     * When no tensor name is specified, the operation targets the latest tensor automatically.
     * If {@code axis = -1} or {@code 0}, all dimensions are reduced into a single scalar value.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the summation.
     * @see #sum(String, String, int)
     */
    default CuBridge sum() {
        return sum("", genRandomNameAxisCascade(), -1);
    }

    /**
     * **Sum — Reduction summation from a specified axis**
     *
     * Performs a reduction summation along all dimensions starting from {@code axis}
     * through the final axis (inclusive).
     * <p>
     * {@code axis = -1} or {@code 0} → reduction across all axes (scalar output).<br>
     * {@code axis = sLen - 1} → reduction only along the last dimension.
     * </p>
     *
     * @param axis The starting axis index for the reduction.
     * @return A {@link CuBridge} instance representing the result of the summation.
     * @see #sum(String, String, int)
     */
    default CuBridge sum(int axis) {
        return sum("", genRandomNameAxisCascade(), axis);
    }

    /**
     * **Sum — Reduction summation on a named tensor**
     *
     * Performs a reduction summation on the specified tensor {@code a},
     * summing all elements from {@code axis} (default: -1) down to the last axis,
     * and stores the result in {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #sum(String, String, int)
     */
    default CuBridge sum(String a, String out) {
        return sum(a, out, -1);
    }

    /**
     * **Sum — Reduction operation across multiple axes**
     *
     * Performs a reduction summation on tensor {@code a}, accumulating values
     * along all dimensions from {@code axis} to the final axis.
     * <p>
     * This operation compresses the specified axes into smaller dimensions
     * by computing their total sum.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name of the output tensor to store the result.
     * @param axis  The starting axis for reduction.
     * @return A {@link CuBridge} instance representing the result.
     */
    default CuBridge sum(String a, String out, int axis) {
        if (CuBridgeJNI.sum(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | sum | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Sum — Overload using a Tensor object**
     *
     * Performs a reduction summation on a {@link Tensor} object,
     * summing its values along all axes from {@code axis} to the last.
     * <p>
     * Automatically assigns a temporary internal name to the tensor
     * before executing {@link #sum(String, String, int)}.
     * </p>
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #sum(String, String, int)
     */
    default CuBridge sum(Tensor a, String out) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return sum(aName, out, -1);
    }

    /**
     * **Sum — Overload using a Tensor object with explicit axis**
     *
     * Performs a reduction summation on {@link Tensor} {@code a},
     * summing all elements from {@code axis} through the final dimension.
     * <p>
     * Automatically assigns an internal temporary name before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The starting axis index for reduction.
     * @return A {@link CuBridge} instance representing the result.
     * @see #sum(String, String, int)
     */
    default CuBridge sum(Tensor a, String out, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return sum(aName, out, axis);
    }

    /**
     * **SumI — Immediate reduction summation**
     *
     * Immediately performs a reduction summation on the most recent tensor in the queue,
     * summing across all axes and returning the result tensor directly.
     *
     * @return A {@link Tensor} containing the reduction result.
     * @see #sum(String, String, int)
     */
    default Tensor sumI() {
        String oName = genRandomNameAxisCascade();
        return sum("", oName, -1).get(oName);
    }

    /**
     * **SumI — Immediate reduction from a specified axis**
     *
     * Immediately performs a reduction summation starting from {@code axis}
     * through the last axis, returning the resulting tensor.
     *
     * @param axis The starting axis index for reduction.
     * @return A {@link Tensor} containing the reduction result.
     * @see #sum(String, String, int)
     */
    default Tensor sumI(int axis) {
        String oName = genRandomNameAxisCascade();
        return sum("", oName, axis).get(oName);
    }

    /**
     * **SumI — Immediate reduction on a named tensor**
     *
     * Performs a reduction summation on the tensor {@code a},
     * summing all dimensions down to the last axis and returning the result immediately.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the reduction result.
     * @see #sum(String, String, int)
     */
    default Tensor sumI(String a) {
        String oName = genRandomNameAxisCascade();
        return sum(a, oName, -1).get(oName);
    }

    /**
     * **SumI — Immediate reduction on a named tensor with axis**
     *
     * Performs a reduction summation on tensor {@code a},
     * summing all elements from {@code axis} through the final axis,
     * and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The starting axis index for reduction.
     * @return A {@link Tensor} containing the reduction result.
     * @see #sum(String, String, int)
     */
    default Tensor sumI(String a, int axis) {
        String oName = genRandomNameAxisCascade();
        return sum(a, oName, axis).get(oName);
    }

    /**
     * **SumI — Immediate reduction using a Tensor object**
     *
     * Performs a reduction summation on a {@link Tensor} object,
     * summing all its values along every dimension, and returns the reduced tensor.
     * <p>
     * The input tensor is temporarily registered before computation.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the reduction result.
     * @see #sum(String, String, int)
     */
    default Tensor sumI(Tensor a) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return sum(aName, oName, -1).get(oName);
    }

    /**
     * **SumI — Immediate reduction using a Tensor object with axis**
     *
     * Performs a reduction summation on {@link Tensor} {@code a},
     * summing all values from {@code axis} through the final dimension.
     *
     * @param a    The input {@link Tensor}.
     * @param axis The starting axis index for reduction.
     * @return A {@link Tensor} containing the reduction result.
     * @see #sum(String, String, int)
     */
    default Tensor sumI(Tensor a, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return sum(aName, oName, axis).get(oName);
    }


    /**
     * **Mean — Basic reduction mean with empty tensor references**
     *
     * Computes the mean value of the most recent tensor stored in the internal queue.
     * <p>
     * When no tensor name is specified, the operation targets the latest tensor automatically.
     * If {@code axis = -1} or {@code 0}, the mean is computed across all elements,
     * producing a single scalar result.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the mean operation.
     * @see #mean(String, String, int)
     */
    default CuBridge mean() {
        return mean("", genRandomNameAxisCascade(), -1);
    }

    /**
     * **Mean — Reduction mean from a specified axis**
     *
     * Computes the mean of all elements in the most recent tensor,
     * reducing values from {@code axis} through the final axis.
     * <p>
     * {@code axis = -1} or {@code 0} → mean across all axes (scalar output).<br>
     * {@code axis = sLen - 1} → mean only along the last dimension.
     * </p>
     *
     * @param axis The starting axis index for the reduction.
     * @return A {@link CuBridge} instance representing the result of the mean operation.
     * @see #mean(String, String, int)
     */
    default CuBridge mean(int axis) {
        return mean("", genRandomNameAxisCascade(), axis);
    }

    /**
     * **Mean — Reduction mean on a named tensor**
     *
     * Computes the mean of the specified tensor {@code a},
     * averaging all elements from {@code axis} (default: -1) down to the last axis,
     * and stores the result in {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #mean(String, String, int)
     */
    default CuBridge mean(String a, String out) {
        return mean(a, out, -1);
    }

    /**
     * **Mean — Reduction operation across multiple axes**
     *
     * Computes the mean of tensor {@code a}, averaging values
     * along all dimensions from {@code axis} to the final axis.
     * <p>
     * This operation compresses the specified axes into smaller dimensions
     * by computing their average value.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name of the output tensor to store the result.
     * @param axis  The starting axis for reduction.
     * @return A {@link CuBridge} instance representing the result.
     */
    default CuBridge mean(String a, String out, int axis) {
        if (CuBridgeJNI.mean(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | mean | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Mean — Overload using a Tensor object**
     *
     * Computes the mean of a {@link Tensor} object,
     * averaging its values along all axes from {@code axis} to the last.
     * <p>
     * Automatically assigns a temporary internal name to the tensor
     * before executing {@link #mean(String, String, int)}.
     * </p>
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #mean(String, String, int)
     */
    default CuBridge mean(Tensor a, String out) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return mean(aName, out, -1);
    }

    /**
     * **Mean — Overload using a Tensor object with explicit axis**
     *
     * Computes the mean of {@link Tensor} {@code a},
     * averaging all elements from {@code axis} through the final dimension.
     * <p>
     * Automatically assigns an internal temporary name before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The starting axis index for reduction.
     * @return A {@link CuBridge} instance representing the result.
     * @see #mean(String, String, int)
     */
    default CuBridge mean(Tensor a, String out, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return mean(aName, out, axis);
    }

    /**
     * **MeanI — Immediate reduction mean**
     *
     * Immediately computes the mean of the most recent tensor in the queue,
     * averaging across all axes and returning the result tensor directly.
     *
     * @return A {@link Tensor} containing the mean result.
     * @see #mean(String, String, int)
     */
    default Tensor meanI() {
        String oName = genRandomNameAxisCascade();
        return mean("", oName, -1).get(oName);
    }

    /**
     * **MeanI — Immediate mean from a specified axis**
     *
     * Immediately computes the mean starting from {@code axis}
     * through the last axis, returning the resulting tensor.
     *
     * @param axis The starting axis index for reduction.
     * @return A {@link Tensor} containing the mean result.
     * @see #mean(String, String, int)
     */
    default Tensor meanI(int axis) {
        String oName = genRandomNameAxisCascade();
        return mean("", oName, axis).get(oName);
    }

    /**
     * **MeanI — Immediate mean on a named tensor**
     *
     * Computes the mean of tensor {@code a},
     * averaging all dimensions down to the last axis and returning the result immediately.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the mean result.
     * @see #mean(String, String, int)
     */
    default Tensor meanI(String a) {
        String oName = genRandomNameAxisCascade();
        return mean(a, oName, -1).get(oName);
    }

    /**
     * **MeanI — Immediate mean on a named tensor with axis**
     *
     * Computes the mean of tensor {@code a},
     * averaging all elements from {@code axis} through the final axis,
     * and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The starting axis index for reduction.
     * @return A {@link Tensor} containing the mean result.
     * @see #mean(String, String, int)
     */
    default Tensor meanI(String a, int axis) {
        String oName = genRandomNameAxisCascade();
        return mean(a, oName, axis).get(oName);
    }

    /**
     * **MeanI — Immediate mean using a Tensor object**
     *
     * Computes the mean of a {@link Tensor} object,
     * averaging all its values along every dimension, and returns the reduced tensor.
     * <p>
     * The input tensor is temporarily registered before computation.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the mean result.
     * @see #mean(String, String, int)
     */
    default Tensor meanI(Tensor a) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return mean(aName, oName, -1).get(oName);
    }

    /**
     * **MeanI — Immediate mean using a Tensor object with axis**
     *
     * Computes the mean of {@link Tensor} {@code a},
     * averaging all values from {@code axis} through the final dimension.
     *
     * @param a    The input {@link Tensor}.
     * @param axis The starting axis index for reduction.
     * @return A {@link Tensor} containing the mean result.
     * @see #mean(String, String, int)
     */
    default Tensor meanI(Tensor a, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return mean(aName, oName, axis).get(oName);
    }


    /**
     * **Var — Basic reduction variance with empty tensor references**
     *
     * Performs variance computation on the most recent tensor stored in the internal queue.
     * <p>
     * When no tensor name is specified, the operation targets the latest tensor automatically.
     * If {@code axis = -1} or {@code 0}, variance is computed across all elements,
     * resulting in a single scalar output.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the variance operation.
     * @see #var(String, String, int)
     */
    default CuBridge var() {
        return var("", genRandomNameAxisCascade(), -1);
    }

    /**
     * **Var — Reduction variance from a specified axis**
     *
     * Performs variance computation on the most recent tensor,
     * reducing all elements from {@code axis} through the last axis (inclusive).
     * <p>
     * {@code axis = -1} or {@code 0} → reduction across all axes (scalar output).<br>
     * {@code axis = sLen - 1} → reduction only along the final dimension.
     * </p>
     *
     * @param axis The starting axis index for variance reduction.
     * @return A {@link CuBridge} instance representing the result of the variance operation.
     * @see #var(String, String, int)
     */
    default CuBridge var(int axis) {
        return var("", genRandomNameAxisCascade(), axis);
    }

    /**
     * **Var — Reduction variance on a named tensor**
     *
     * Performs variance computation on the specified tensor {@code a},
     * reducing all elements from {@code axis} (default: -1) down to the last axis,
     * and stores the result in {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #var(String, String, int)
     */
    default CuBridge var(String a, String out) {
        return var(a, out, -1);
    }

    /**
     * **Var — Reduction operation across multiple axes**
     *
     * Performs variance computation on tensor {@code a},
     * reducing values along all dimensions from {@code axis} to the final axis (inclusive).
     * <p>
     * The operation measures overall value spread across the specified range of axes
     * and produces a tensor with reduced dimensionality.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name of the output tensor to store the result.
     * @param axis  The starting axis for variance reduction.
     * @return A {@link CuBridge} instance representing the result.
     */
    default CuBridge var(String a, String out, int axis) {
        if (CuBridgeJNI.var(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | var | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Var — Overload using a Tensor object**
     *
     * Performs variance computation on a {@link Tensor} object,
     * reducing all elements along every axis from {@code axis} to the last axis.
     * <p>
     * Automatically assigns a temporary internal name to the tensor
     * before executing {@link #var(String, String, int)}.
     * </p>
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #var(String, String, int)
     */
    default CuBridge var(Tensor a, String out) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return var(aName, out, -1);
    }

    /**
     * **Var — Overload using a Tensor object with explicit axis**
     *
     * Performs variance computation on {@link Tensor} {@code a},
     * reducing all elements from {@code axis} through the last dimension.
     * <p>
     * Automatically assigns an internal temporary name before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The starting axis index for variance reduction.
     * @return A {@link CuBridge} instance representing the result.
     * @see #var(String, String, int)
     */
    default CuBridge var(Tensor a, String out, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return var(aName, out, axis);
    }

    /**
     * **VarI — Immediate reduction variance**
     *
     * Immediately performs variance computation on the most recent tensor in the queue,
     * reducing all axes from the starting dimension to the last, and returns the result directly.
     *
     * @return A {@link Tensor} containing the variance result.
     * @see #var(String, String, int)
     */
    default Tensor varI() {
        String oName = genRandomNameAxisCascade();
        return var("", oName, -1).get(oName);
    }

    /**
     * **VarI — Immediate variance from a specified axis**
     *
     * Immediately performs variance computation starting from {@code axis}
     * through the last axis and returns the resulting tensor.
     *
     * @param axis The starting axis index for variance reduction.
     * @return A {@link Tensor} containing the variance result.
     * @see #var(String, String, int)
     */
    default Tensor varI(int axis) {
        String oName = genRandomNameAxisCascade();
        return var("", oName, axis).get(oName);
    }

    /**
     * **VarI — Immediate variance on a named tensor**
     *
     * Performs variance computation on tensor {@code a},
     * reducing all dimensions from {@code axis} down to the last axis
     * and returning the result immediately.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the variance result.
     * @see #var(String, String, int)
     */
    default Tensor varI(String a) {
        String oName = genRandomNameAxisCascade();
        return var(a, oName, -1).get(oName);
    }

    /**
     * **VarI — Immediate variance on a named tensor with axis**
     *
     * Performs variance computation on tensor {@code a},
     * reducing all elements from {@code axis} through the final axis (inclusive),
     * and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The starting axis index for variance reduction.
     * @return A {@link Tensor} containing the variance result.
     * @see #var(String, String, int)
     */
    default Tensor varI(String a, int axis) {
        String oName = genRandomNameAxisCascade();
        return var(a, oName, axis).get(oName);
    }

    /**
     * **VarI — Immediate variance using a Tensor object**
     *
     * Performs variance computation on a {@link Tensor} object,
     * reducing all its values from the first to the last axis
     * and returning the reduced tensor directly.
     * <p>
     * The input tensor is temporarily registered before computation.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the variance result.
     * @see #var(String, String, int)
     */
    default Tensor varI(Tensor a) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return var(aName, oName, -1).get(oName);
    }

    /**
     * **VarI — Immediate variance using a Tensor object with axis**
     *
     * Performs variance computation on {@link Tensor} {@code a},
     * reducing all values from {@code axis} through the last dimension (inclusive).
     *
     * @param a    The input {@link Tensor}.
     * @param axis The starting axis index for variance reduction.
     * @return A {@link Tensor} containing the variance result.
     * @see #var(String, String, int)
     */
    default Tensor varI(Tensor a, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return var(aName, oName, axis).get(oName);
    }


    /**
     * **Std — Basic reduction standard deviation with empty tensor references**
     *
     * Performs standard deviation computation on the most recent tensor stored in the internal queue.
     * <p>
     * When no tensor name is specified, the operation targets the latest tensor automatically.
     * If {@code axis = -1} or {@code 0}, standard deviation is computed across all elements,
     * resulting in a single scalar output.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the standard deviation operation.
     * @see #std(String, String, int)
     */
    default CuBridge std() {
        return std("", genRandomNameAxisCascade(), -1);
    }

    /**
     * **Std — Reduction standard deviation from a specified axis**
     *
     * Performs standard deviation computation on the most recent tensor,
     * reducing all elements from {@code axis} through the last axis (inclusive).
     * <p>
     * {@code axis = -1} or {@code 0} → reduction across all axes (scalar output).<br>
     * {@code axis = sLen - 1} → reduction only along the final dimension.
     * </p>
     *
     * @param axis The starting axis index for standard deviation reduction.
     * @return A {@link CuBridge} instance representing the result of the standard deviation operation.
     * @see #std(String, String, int)
     */
    default CuBridge std(int axis) {
        return std("", genRandomNameAxisCascade(), axis);
    }

    /**
     * **Std — Reduction standard deviation on a named tensor**
     *
     * Performs standard deviation computation on the specified tensor {@code a},
     * reducing all elements from {@code axis} (default: -1) down to the last axis,
     * and stores the result in {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #std(String, String, int)
     */
    default CuBridge std(String a, String out) {
        return std(a, out, -1);
    }

    /**
     * **Std — Reduction operation across multiple axes**
     *
     * Performs standard deviation computation on tensor {@code a},
     * reducing values along all dimensions from {@code axis} to the final axis (inclusive).
     * <p>
     * The operation measures overall value dispersion across the specified range of axes
     * and produces a tensor with reduced dimensionality.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name of the output tensor to store the result.
     * @param axis  The starting axis for standard deviation reduction.
     * @return A {@link CuBridge} instance representing the result.
     */
    default CuBridge std(String a, String out, int axis) {
        if (CuBridgeJNI.std(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | std | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Std — Overload using a Tensor object**
     *
     * Performs standard deviation computation on a {@link Tensor} object,
     * reducing all elements along every axis from {@code axis} to the last axis.
     * <p>
     * Automatically assigns a temporary internal name to the tensor
     * before executing {@link #std(String, String, int)}.
     * </p>
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #std(String, String, int)
     */
    default CuBridge std(Tensor a, String out) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return std(aName, out, -1);
    }

    /**
     * **Std — Overload using a Tensor object with explicit axis**
     *
     * Performs standard deviation computation on {@link Tensor} {@code a},
     * reducing all elements from {@code axis} through the last dimension.
     * <p>
     * Automatically assigns an internal temporary name before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The starting axis index for standard deviation reduction.
     * @return A {@link CuBridge} instance representing the result.
     * @see #std(String, String, int)
     */
    default CuBridge std(Tensor a, String out, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return std(aName, out, axis);
    }

    /**
     * **StdI — Immediate reduction standard deviation**
     *
     * Immediately performs standard deviation computation on the most recent tensor in the queue,
     * reducing all axes from the starting dimension to the last, and returns the result directly.
     *
     * @return A {@link Tensor} containing the standard deviation result.
     * @see #std(String, String, int)
     */
    default Tensor stdI() {
        String oName = genRandomNameAxisCascade();
        return std("", oName, -1).get(oName);
    }

    /**
     * **StdI — Immediate standard deviation from a specified axis**
     *
     * Immediately performs standard deviation computation starting from {@code axis}
     * through the last axis and returns the resulting tensor.
     *
     * @param axis The starting axis index for standard deviation reduction.
     * @return A {@link Tensor} containing the standard deviation result.
     * @see #std(String, String, int)
     */
    default Tensor stdI(int axis) {
        String oName = genRandomNameAxisCascade();
        return std("", oName, axis).get(oName);
    }

    /**
     * **StdI — Immediate standard deviation on a named tensor**
     *
     * Performs standard deviation computation on tensor {@code a},
     * reducing all dimensions from {@code axis} down to the last axis
     * and returning the result immediately.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the standard deviation result.
     * @see #std(String, String, int)
     */
    default Tensor stdI(String a) {
        String oName = genRandomNameAxisCascade();
        return std(a, oName, -1).get(oName);
    }

    /**
     * **StdI — Immediate standard deviation on a named tensor with axis**
     *
     * Performs standard deviation computation on tensor {@code a},
     * reducing all elements from {@code axis} through the final axis (inclusive),
     * and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The starting axis index for standard deviation reduction.
     * @return A {@link Tensor} containing the standard deviation result.
     * @see #std(String, String, int)
     */
    default Tensor stdI(String a, int axis) {
        String oName = genRandomNameAxisCascade();
        return std(a, oName, axis).get(oName);
    }

    /**
     * **StdI — Immediate standard deviation using a Tensor object**
     *
     * Performs standard deviation computation on a {@link Tensor} object,
     * reducing all its values from the first to the last axis
     * and returning the reduced tensor directly.
     * <p>
     * The input tensor is temporarily registered before computation.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the standard deviation result.
     * @see #std(String, String, int)
     */
    default Tensor stdI(Tensor a) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return std(aName, oName, -1).get(oName);
    }

    /**
     * **StdI — Immediate standard deviation using a Tensor object with axis**
     *
     * Performs standard deviation computation on {@link Tensor} {@code a},
     * reducing all values from {@code axis} through the last dimension (inclusive).
     *
     * @param a    The input {@link Tensor}.
     * @param axis The starting axis index for standard deviation reduction.
     * @return A {@link Tensor} containing the standard deviation result.
     * @see #std(String, String, int)
     */
    default Tensor stdI(Tensor a, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return std(aName, oName, axis).get(oName);
    }


    /**
     * **Max — Basic reduction maximum with empty tensor references**
     *
     * Performs maximum value computation on the most recent tensor stored in the internal queue.
     * <p>
     * When no tensor name is specified, the operation targets the latest tensor automatically.
     * If {@code axis = -1} or {@code 0}, the maximum value is computed across all elements,
     * resulting in a single scalar output.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the maximum operation.
     * @see #max(String, String, int)
     */
    default CuBridge max() {
        return max("", genRandomNameAxisCascade(), -1);
    }

    /**
     * **Max — Reduction maximum from a specified axis**
     *
     * Performs maximum value computation on the most recent tensor,
     * reducing all elements from {@code axis} through the last axis (inclusive).
     * <p>
     * {@code axis = -1} or {@code 0} → reduction across all axes (scalar output).<br>
     * {@code axis = sLen - 1} → reduction only along the final dimension.
     * </p>
     *
     * @param axis The starting axis index for maximum reduction.
     * @return A {@link CuBridge} instance representing the result of the maximum operation.
     * @see #max(String, String, int)
     */
    default CuBridge max(int axis) {
        return max("", genRandomNameAxisCascade(), axis);
    }

    /**
     * **Max — Reduction maximum on a named tensor**
     *
     * Performs maximum value computation on the specified tensor {@code a},
     * reducing all elements from {@code axis} (default: -1) down to the last axis,
     * and stores the result in {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #max(String, String, int)
     */
    default CuBridge max(String a, String out) {
        return max(a, out, -1);
    }

    /**
     * **Max — Reduction operation across multiple axes**
     *
     * Performs maximum value computation on tensor {@code a},
     * reducing values along all dimensions from {@code axis} to the final axis (inclusive).
     * <p>
     * The operation finds the largest element across the specified range of axes
     * and produces a tensor with reduced dimensionality.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name of the output tensor to store the result.
     * @param axis  The starting axis for maximum reduction.
     * @return A {@link CuBridge} instance representing the result.
     */
    default CuBridge max(String a, String out, int axis) {
        if (CuBridgeJNI.max(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | max | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Max — Overload using a Tensor object**
     *
     * Performs maximum value computation on a {@link Tensor} object,
     * reducing all elements along every axis from {@code axis} to the last axis.
     * <p>
     * Automatically assigns a temporary internal name to the tensor
     * before executing {@link #max(String, String, int)}.
     * </p>
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #max(String, String, int)
     */
    default CuBridge max(Tensor a, String out) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return max(aName, out, -1);
    }

    /**
     * **Max — Overload using a Tensor object with explicit axis**
     *
     * Performs maximum value computation on {@link Tensor} {@code a},
     * reducing all elements from {@code axis} through the last dimension.
     * <p>
     * Automatically assigns an internal temporary name before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The starting axis index for maximum reduction.
     * @return A {@link CuBridge} instance representing the result.
     * @see #max(String, String, int)
     */
    default CuBridge max(Tensor a, String out, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return max(aName, out, axis);
    }

    /**
     * **MaxI — Immediate reduction maximum**
     *
     * Immediately performs maximum value computation on the most recent tensor in the queue,
     * reducing all axes from the starting dimension to the last, and returns the result directly.
     *
     * @return A {@link Tensor} containing the maximum result.
     * @see #max(String, String, int)
     */
    default Tensor maxI() {
        String oName = genRandomNameAxisCascade();
        return max("", oName, -1).get(oName);
    }

    /**
     * **MaxI — Immediate maximum from a specified axis**
     *
     * Immediately performs maximum value computation starting from {@code axis}
     * through the last axis and returns the resulting tensor.
     *
     * @param axis The starting axis index for maximum reduction.
     * @return A {@link Tensor} containing the maximum result.
     * @see #max(String, String, int)
     */
    default Tensor maxI(int axis) {
        String oName = genRandomNameAxisCascade();
        return max("", oName, axis).get(oName);
    }

    /**
     * **MaxI — Immediate maximum on a named tensor**
     *
     * Performs maximum value computation on tensor {@code a},
     * reducing all dimensions from {@code axis} down to the last axis
     * and returning the result immediately.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the maximum result.
     * @see #max(String, String, int)
     */
    default Tensor maxI(String a) {
        String oName = genRandomNameAxisCascade();
        return max(a, oName, -1).get(oName);
    }

    /**
     * **MaxI — Immediate maximum on a named tensor with axis**
     *
     * Performs maximum value computation on tensor {@code a},
     * reducing all elements from {@code axis} through the final axis (inclusive),
     * and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The starting axis index for maximum reduction.
     * @return A {@link Tensor} containing the maximum result.
     * @see #max(String, String, int)
     */
    default Tensor maxI(String a, int axis) {
        String oName = genRandomNameAxisCascade();
        return max(a, oName, axis).get(oName);
    }

    /**
     * **MaxI — Immediate maximum using a Tensor object**
     *
     * Performs maximum value computation on a {@link Tensor} object,
     * reducing all its values from the first to the last axis
     * and returning the reduced tensor directly.
     * <p>
     * The input tensor is temporarily registered before computation.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the maximum result.
     * @see #max(String, String, int)
     */
    default Tensor maxI(Tensor a) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return max(aName, oName, -1).get(oName);
    }

    /**
     * **MaxI — Immediate maximum using a Tensor object with axis**
     *
     * Performs maximum value computation on {@link Tensor} {@code a},
     * reducing all values from {@code axis} through the last dimension (inclusive).
     *
     * @param a    The input {@link Tensor}.
     * @param axis The starting axis index for maximum reduction.
     * @return A {@link Tensor} containing the maximum result.
     * @see #max(String, String, int)
     */
    default Tensor maxI(Tensor a, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return max(aName, oName, axis).get(oName);
    }


    /**
     * **Min — Basic reduction minimum with empty tensor references**
     *
     * Performs minimum value computation on the most recent tensor stored in the internal queue.
     * <p>
     * When no tensor name is specified, the operation targets the latest tensor automatically.
     * If {@code axis = -1} or {@code 0}, the minimum value is computed across all elements,
     * resulting in a single scalar output.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the minimum operation.
     * @see #min(String, String, int)
     */
    default CuBridge min() {
        return min("", genRandomNameAxisCascade(), -1);
    }

    /**
     * **Min — Reduction minimum from a specified axis**
     *
     * Performs minimum value computation on the most recent tensor,
     * reducing all elements from {@code axis} through the last axis (inclusive).
     * <p>
     * {@code axis = -1} or {@code 0} → reduction across all axes (scalar output).<br>
     * {@code axis = sLen - 1} → reduction only along the final dimension.
     * </p>
     *
     * @param axis The starting axis index for minimum reduction.
     * @return A {@link CuBridge} instance representing the result of the minimum operation.
     * @see #min(String, String, int)
     */
    default CuBridge min(int axis) {
        return min("", genRandomNameAxisCascade(), axis);
    }

    /**
     * **Min — Reduction minimum on a named tensor**
     *
     * Performs minimum value computation on the specified tensor {@code a},
     * reducing all elements from {@code axis} (default: -1) down to the last axis,
     * and stores the result in {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #min(String, String, int)
     */
    default CuBridge min(String a, String out) {
        return min(a, out, -1);
    }

    /**
     * **Min — Reduction operation across multiple axes**
     *
     * Performs minimum value computation on tensor {@code a},
     * reducing values along all dimensions from {@code axis} to the final axis (inclusive).
     * <p>
     * The operation finds the smallest element across the specified range of axes
     * and produces a tensor with reduced dimensionality.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name of the output tensor to store the result.
     * @param axis  The starting axis for minimum reduction.
     * @return A {@link CuBridge} instance representing the result.
     */
    default CuBridge min(String a, String out, int axis) {
        if (CuBridgeJNI.min(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | min | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Min — Overload using a Tensor object**
     *
     * Performs minimum value computation on a {@link Tensor} object,
     * reducing all elements along every axis from {@code axis} to the last axis.
     * <p>
     * Automatically assigns a temporary internal name to the tensor
     * before executing {@link #min(String, String, int)}.
     * </p>
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result.
     * @see #min(String, String, int)
     */
    default CuBridge min(Tensor a, String out) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return min(aName, out, -1);
    }

    /**
     * **Min — Overload using a Tensor object with explicit axis**
     *
     * Performs minimum value computation on {@link Tensor} {@code a},
     * reducing all elements from {@code axis} through the last dimension.
     * <p>
     * Automatically assigns an internal temporary name before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The starting axis index for minimum reduction.
     * @return A {@link CuBridge} instance representing the result.
     * @see #min(String, String, int)
     */
    default CuBridge min(Tensor a, String out, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        return min(aName, out, axis);
    }

    /**
     * **MinI — Immediate reduction minimum**
     *
     * Immediately performs minimum value computation on the most recent tensor in the queue,
     * reducing all axes from the starting dimension to the last, and returns the result directly.
     *
     * @return A {@link Tensor} containing the minimum result.
     * @see #min(String, String, int)
     */
    default Tensor minI() {
        String oName = genRandomNameAxisCascade();
        return min("", oName, -1).get(oName);
    }

    /**
     * **MinI — Immediate minimum from a specified axis**
     *
     * Immediately performs minimum value computation starting from {@code axis}
     * through the last axis and returns the resulting tensor.
     *
     * @param axis The starting axis index for minimum reduction.
     * @return A {@link Tensor} containing the minimum result.
     * @see #min(String, String, int)
     */
    default Tensor minI(int axis) {
        String oName = genRandomNameAxisCascade();
        return min("", oName, axis).get(oName);
    }

    /**
     * **MinI — Immediate minimum on a named tensor**
     *
     * Performs minimum value computation on tensor {@code a},
     * reducing all dimensions from {@code axis} down to the last axis
     * and returning the result immediately.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the minimum result.
     * @see #min(String, String, int)
     */
    default Tensor minI(String a) {
        String oName = genRandomNameAxisCascade();
        return min(a, oName, -1).get(oName);
    }

    /**
     * **MinI — Immediate minimum on a named tensor with axis**
     *
     * Performs minimum value computation on tensor {@code a},
     * reducing all elements from {@code axis} through the final axis (inclusive),
     * and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The starting axis index for minimum reduction.
     * @return A {@link Tensor} containing the minimum result.
     * @see #min(String, String, int)
     */
    default Tensor minI(String a, int axis) {
        String oName = genRandomNameAxisCascade();
        return min(a, oName, axis).get(oName);
    }

    /**
     * **MinI — Immediate minimum using a Tensor object**
     *
     * Performs minimum value computation on a {@link Tensor} object,
     * reducing all its values from the first to the last axis
     * and returning the reduced tensor directly.
     * <p>
     * The input tensor is temporarily registered before computation.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the minimum result.
     * @see #min(String, String, int)
     */
    default Tensor minI(Tensor a) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return min(aName, oName, -1).get(oName);
    }

    /**
     * **MinI — Immediate minimum using a Tensor object with axis**
     *
     * Performs minimum value computation on {@link Tensor} {@code a},
     * reducing all values from {@code axis} through the last dimension (inclusive).
     *
     * @param a    The input {@link Tensor}.
     * @param axis The starting axis index for minimum reduction.
     * @return A {@link Tensor} containing the minimum result.
     * @see #min(String, String, int)
     */
    default Tensor minI(Tensor a, int axis) {
        String aName = genRandomNameAxisCascade(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxisCascade();
        return min(aName, oName, axis).get(oName);
    }



}
