package CuBridge;

import java.util.UUID;

public interface AxisOps{

    private String genRandomNameAxis() {
        return "AxisOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /**
     * **Accumulate — Basic accumulation along a specified axis**
     *
     * Performs accumulation along the given {@code axis} of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1, while all other dimensions remain unchanged.
     * The operation aggregates values along the selected axis without affecting other axes.
     * </p>
     *
     * @param axis The axis along which accumulation is performed.
     * @return A {@link CuBridge} instance representing the result of the accumulation.
     * @see #accumulate(String, String, int)
     */
    default CuBridge accumulate(int axis) {
        return accumulate("", genRandomNameAxis(), axis);
    }

    /**
     * **Accumulate — Accumulation on a named tensor**
     *
     * Performs accumulation on the specified tensor {@code a} along the given {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The specified axis is reduced to length 1, while all other axes are preserved.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which accumulation is performed.
     * @return A {@link CuBridge} instance representing the result of the accumulation.
     * @see #accumulate(String, String, int)
     */
    default CuBridge accumulate(String a, String out, int axis) {
        if (CuBridgeJNI.accumulate(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | accumulate | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Accumulate — Overload using a Tensor object**
     *
     * Performs accumulation on the given {@link Tensor} object along the specified {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The operation reduces only the selected axis to size 1 and leaves all other dimensions unchanged.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which accumulation is performed.
     * @return A {@link CuBridge} instance representing the result of the accumulation.
     * @see #accumulate(String, String, int)
     */
    default CuBridge accumulate(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return accumulate(aName, out, axis);
    }

    /**
     * **AccumulateI — Immediate accumulation along a specified axis**
     *
     * Immediately performs accumulation on the most recent tensor in the queue
     * along the given {@code axis} and returns the result tensor directly.
     * <p>
     * Only the specified axis is reduced to size 1; all other axes remain intact.
     * </p>
     *
     * @param axis The axis along which accumulation is performed.
     * @return A {@link Tensor} containing the accumulated result.
     * @see #accumulate(String, String, int)
     */
    default Tensor accumulateI(int axis) {
        String oName = genRandomNameAxis();
        return accumulate("", oName, axis).get(oName);
    }

    /**
     * **AccumulateI — Immediate accumulation on a named tensor**
     *
     * Performs immediate accumulation on the tensor {@code a} along the specified {@code axis},
     * reducing that axis to size 1, and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which accumulation is performed.
     * @return A {@link Tensor} containing the accumulated result.
     * @see #accumulate(String, String, int)
     */
    default Tensor accumulateI(String a, int axis) {
        String oName = genRandomNameAxis();
        return accumulate(a, oName, axis).get(oName);
    }

    /**
     * **AccumulateI — Immediate accumulation using a Tensor object**
     *
     * Performs accumulation on a {@link Tensor} object along the specified {@code axis}
     * and directly returns the accumulated result.
     * <p>
     * The selected axis is reduced to size 1; all other dimensions are preserved.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which accumulation is performed.
     * @return A {@link Tensor} containing the accumulated result.
     * @see #accumulate(String, String, int)
     */
    default Tensor accumulateI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return accumulate(aName, oName, axis).get(oName);
    }


    /**
     * **Compress — Basic mean compression along a specified axis**
     *
     * Performs mean compression along the given {@code axis} of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1 by averaging its values,
     * while all other dimensions remain unchanged.
     * </p>
     *
     * @param axis The axis along which compression is performed.
     * @return A {@link CuBridge} instance representing the result of the compression.
     * @see #compress(String, String, int)
     */
    default CuBridge compress(int axis) {
        return compress("", genRandomNameAxis(), axis);
    }

    /**
     * **Compress — Mean compression on a named tensor**
     *
     * Performs mean compression on the specified tensor {@code a} along the given {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The operation computes the mean along the selected axis, reducing it to size 1,
     * while preserving all other axes.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which compression is performed.
     * @return A {@link CuBridge} instance representing the result of the compression.
     * @see #compress(String, String, int)
     */
    default CuBridge compress(String a, String out, int axis) {
        if (CuBridgeJNI.compress(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | compress | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **Compress — Overload using a Tensor object**
     *
     * Performs mean compression on a {@link Tensor} object along the specified {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1 by averaging its values;
     * all other dimensions are left unchanged.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which compression is performed.
     * @return A {@link CuBridge} instance representing the result of the compression.
     * @see #compress(String, String, int)
     */
    default CuBridge compress(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return compress(aName, out, axis);
    }

    /**
     * **CompressI — Immediate mean compression along a specified axis**
     *
     * Immediately performs mean compression on the most recent tensor in the queue
     * along the given {@code axis} and returns the result tensor directly.
     * <p>
     * The specified axis is reduced to length 1 by averaging values along that axis.
     * </p>
     *
     * @param axis The axis along which compression is performed.
     * @return A {@link Tensor} containing the compressed result.
     * @see #compress(String, String, int)
     */
    default Tensor compressI(int axis) {
        String oName = genRandomNameAxis();
        return compress("", oName, axis).get(oName);
    }

    /**
     * **CompressI — Immediate mean compression on a named tensor**
     *
     * Performs mean compression on the tensor {@code a} along the specified {@code axis},
     * reducing that axis to size 1, and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which compression is performed.
     * @return A {@link Tensor} containing the compressed result.
     * @see #compress(String, String, int)
     */
    default Tensor compressI(String a, int axis) {
        String oName = genRandomNameAxis();
        return compress(a, oName, axis).get(oName);
    }

    /**
     * **CompressI — Immediate mean compression using a Tensor object**
     *
     * Performs mean compression on a {@link Tensor} object along the specified {@code axis}
     * and returns the result tensor directly.
     * <p>
     * The operation averages values along the selected axis, reducing it to length 1,
     * while preserving all other dimensions.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which compression is performed.
     * @return A {@link Tensor} containing the compressed result.
     * @see #compress(String, String, int)
     */
    default Tensor compressI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return compress(aName, oName, axis).get(oName);
    }


    /**
     * **Expand — Basic axis expansion on the most recent tensor**
     *
     * Expands the specified axis of the most recent tensor in the internal queue
     * by the given expansion factor {@code expandN}.
     * <p>
     * The operation increases the size of the chosen axis by repeating or broadcasting its values.
     * Expansion is allowed when the original axis length is 1 or a divisor of {@code expandN}.
     * </p>
     *
     * @param axis     The axis along which expansion is applied.
     * @param expandN  The factor by which the axis size is expanded.
     * @return A {@link CuBridge} instance representing the result of the expansion.
     * @see #expand(String, String, int, int)
     */
    default CuBridge expand(int axis, int expandN) {
        return expand("", genRandomNameAxis(), axis, expandN);
    }

    /**
     * **Expand — Basic axis expansion on a named tensor**
     *
     * Expands the specified axis of tensor {@code a} by the given expansion factor {@code expandN}
     * and stores the result in {@code out}.
     * <p>
     * The operation increases the size of the chosen axis by repeating or broadcasting its values.
     * Expansion is allowed when the original axis length is 1 or a divisor of {@code expandN}.
     * </p>
     *
     * @param a        The name of the input tensor.
     * @param out      The name to store the expanded tensor.
     * @param axis     The axis along which expansion is applied.
     * @param expandN  The factor by which the axis size is expanded.
     * @return A {@link CuBridge} instance representing the result of the expansion.
     */
    default CuBridge expand(String a, String out, int axis, int expandN) {
        if (CuBridgeJNI.expand(a, out, axis, expandN)) return CuBridge.getInstance();
        else System.err.println("Error | expand | " + a + " | " + out + " | " + axis + " | " + expandN);
        return null;
    }

    /**
     * **Expand — Overload using a Tensor object**
     *
     * Expands the specified axis of a {@link Tensor} object by the given expansion factor {@code expandN}
     * and stores the result in {@code out}.
     * <p>
     * The chosen axis is extended by repeating or broadcasting its values,
     * while all other dimensions remain unchanged.
     * </p>
     *
     * @param a        The input {@link Tensor}.
     * @param out      The name to store the expanded tensor.
     * @param axis     The axis along which expansion is applied.
     * @param expandN  The factor by which the axis size is expanded.
     * @return A {@link CuBridge} instance representing the result of the expansion.
     * @see #expand(String, String, int, int)
     */
    default CuBridge expand(Tensor a, String out, int axis, int expandN) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return expand(aName, out, axis, expandN);
    }

    /**
     * **ExpandI — Immediate axis expansion on the most recent tensor**
     *
     * Immediately performs expansion on the most recent tensor in the queue
     * along the specified {@code axis} by {@code expandN} times,
     * and returns the expanded tensor directly.
     * <p>
     * The expansion repeats or broadcasts elements along the selected axis.
     * </p>
     *
     * @param axis     The axis along which expansion is applied.
     * @param expandN  The factor by which the axis size is expanded.
     * @return A {@link Tensor} containing the expanded result.
     * @see #expand(String, String, int, int)
     */
    default Tensor expandI(int axis, int expandN) {
        String oName = genRandomNameAxis();
        return expand("", oName, axis, expandN).get(oName);
    }

    /**
     * **ExpandI — Immediate axis expansion on a named tensor**
     *
     * Immediately performs expansion on the tensor {@code a}
     * along the specified {@code axis} by {@code expandN} times,
     * and returns the expanded tensor directly.
     * <p>
     * The expansion repeats or broadcasts elements along the chosen axis.
     * </p>
     *
     * @param a        The name of the input tensor.
     * @param axis     The axis along which expansion is applied.
     * @param expandN  The factor by which the axis size is expanded.
     * @return A {@link Tensor} containing the expanded result.
     * @see #expand(String, String, int, int)
     */
    default Tensor expandI(String a, int axis, int expandN) {
        String oName = genRandomNameAxis();
        return expand(a, oName, axis, expandN).get(oName);
    }

    /**
     * **ExpandI — Immediate axis expansion using a Tensor object**
     *
     * Immediately performs expansion on a {@link Tensor} object
     * along the specified {@code axis} by {@code expandN} times,
     * and returns the expanded tensor directly.
     * <p>
     * Expansion is applied by repeating or broadcasting values along the chosen axis.
     * </p>
     *
     * @param a        The input {@link Tensor}.
     * @param axis     The axis along which expansion is applied.
     * @param expandN  The factor by which the axis size is expanded.
     * @return A {@link Tensor} containing the expanded result.
     * @see #expand(String, String, int, int)
     */
    default Tensor expandI(Tensor a, int axis, int expandN) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return expand(aName, oName, axis, expandN).get(oName);
    }


    /**
     * **AxisMax — Basic maximum reduction along a specified axis**
     *
     * Performs maximum computation along the given {@code axis} of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1 by selecting the largest values along that axis,
     * while all other dimensions remain unchanged.
     * </p>
     *
     * @param axis The axis along which the maximum operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisMax(String, String, int)
     */
    default CuBridge axisMax(int axis) {
        return axisMax("", genRandomNameAxis(), axis);
    }

    /**
     * **AxisMax — Maximum reduction on a named tensor**
     *
     * Performs maximum computation on the specified tensor {@code a} along the given {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The operation finds the largest element along the selected axis,
     * reducing that axis to length 1 while preserving all other dimensions.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the maximum operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisMax(String, String, int)
     */
    default CuBridge axisMax(String a, String out, int axis) {
        if (CuBridgeJNI.axisMax(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | axisMax | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **AxisMax — Overload using a Tensor object**
     *
     * Performs maximum computation on the given {@link Tensor} object along the specified {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1 by taking the maximum along that axis;
     * all other dimensions are preserved.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the maximum operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisMax(String, String, int)
     */
    default CuBridge axisMax(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return axisMax(aName, out, axis);
    }

    /**
     * **AxisMaxI — Immediate maximum reduction along a specified axis**
     *
     * Immediately performs maximum computation on the most recent tensor in the queue
     * along the given {@code axis} and returns the result tensor directly.
     * <p>
     * Only the specified axis is reduced to size 1 by taking the maximum along that axis.
     * </p>
     *
     * @param axis The axis along which the maximum operation is performed.
     * @return A {@link Tensor} containing the axis-wise maximum result.
     * @see #axisMax(String, String, int)
     */
    default Tensor axisMaxI(int axis) {
        String oName = genRandomNameAxis();
        return axisMax("", oName, axis).get(oName);
    }

    /**
     * **AxisMaxI — Immediate maximum reduction on a named tensor**
     *
     * Performs immediate maximum computation on the tensor {@code a} along the specified {@code axis},
     * reducing that axis to size 1, and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which the maximum operation is performed.
     * @return A {@link Tensor} containing the axis-wise maximum result.
     * @see #axisMax(String, String, int)
     */
    default Tensor axisMaxI(String a, int axis) {
        String oName = genRandomNameAxis();
        return axisMax(a, oName, axis).get(oName);
    }

    /**
     * **AxisMaxI — Immediate maximum reduction using a Tensor object**
     *
     * Performs maximum computation on a {@link Tensor} object along the specified {@code axis}
     * and returns the result tensor directly.
     * <p>
     * The operation selects the maximum values along the chosen axis, reducing it to length 1,
     * while preserving all other dimensions.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which the maximum operation is performed.
     * @return A {@link Tensor} containing the axis-wise maximum result.
     * @see #axisMax(String, String, int)
     */
    default Tensor axisMaxI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return axisMax(aName, oName, axis).get(oName);
    }


    /**
     * **AxisMin — Basic minimum reduction along a specified axis**
     *
     * Performs minimum computation along the given {@code axis} of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1 by selecting the smallest values along that axis,
     * while all other dimensions remain unchanged.
     * </p>
     *
     * @param axis The axis along which the minimum operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisMin(String, String, int)
     */
    default CuBridge axisMin(int axis) {
        return axisMin("", genRandomNameAxis(), axis);
    }

    /**
     * **AxisMin — Minimum reduction on a named tensor**
     *
     * Performs minimum computation on the specified tensor {@code a} along the given {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The operation finds the smallest element along the selected axis,
     * reducing that axis to length 1 while preserving all other dimensions.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the minimum operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisMin(String, String, int)
     */
    default CuBridge axisMin(String a, String out, int axis) {
        if (CuBridgeJNI.axisMin(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | axisMin | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **AxisMin — Overload using a Tensor object**
     *
     * Performs minimum computation on the given {@link Tensor} object along the specified {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1 by taking the minimum along that axis;
     * all other dimensions are preserved.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the minimum operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisMin(String, String, int)
     */
    default CuBridge axisMin(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return axisMin(aName, out, axis);
    }

    /**
     * **AxisMinI — Immediate minimum reduction along a specified axis**
     *
     * Immediately performs minimum computation on the most recent tensor in the queue
     * along the given {@code axis} and returns the result tensor directly.
     * <p>
     * Only the specified axis is reduced to size 1 by taking the minimum along that axis.
     * </p>
     *
     * @param axis The axis along which the minimum operation is performed.
     * @return A {@link Tensor} containing the axis-wise minimum result.
     * @see #axisMin(String, String, int)
     */
    default Tensor axisMinI(int axis) {
        String oName = genRandomNameAxis();
        return axisMin("", oName, axis).get(oName);
    }

    /**
     * **AxisMinI — Immediate minimum reduction on a named tensor**
     *
     * Performs immediate minimum computation on the tensor {@code a} along the specified {@code axis},
     * reducing that axis to size 1, and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which the minimum operation is performed.
     * @return A {@link Tensor} containing the axis-wise minimum result.
     * @see #axisMin(String, String, int)
     */
    default Tensor axisMinI(String a, int axis) {
        String oName = genRandomNameAxis();
        return axisMin(a, oName, axis).get(oName);
    }

    /**
     * **AxisMinI — Immediate minimum reduction using a Tensor object**
     *
     * Performs minimum computation on a {@link Tensor} object along the specified {@code axis}
     * and returns the result tensor directly.
     * <p>
     * The operation selects the smallest values along the chosen axis, reducing it to length 1,
     * while preserving all other dimensions.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which the minimum operation is performed.
     * @return A {@link Tensor} containing the axis-wise minimum result.
     * @see #axisMin(String, String, int)
     */
    default Tensor axisMinI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return axisMin(aName, oName, axis).get(oName);
    }


    /**
     * **AxisVar — Basic variance reduction along a specified axis**
     *
     * Performs variance computation along the given {@code axis} of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1 by computing the variance of its elements,
     * while all other dimensions remain unchanged.
     * </p>
     *
     * @param axis The axis along which the variance operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisVar(String, String, int)
     */
    default CuBridge axisVar(int axis) {
        return axisVar("", genRandomNameAxis(), axis);
    }

    /**
     * **AxisVar — Variance reduction on a named tensor**
     *
     * Performs variance computation on the specified tensor {@code a} along the given {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The operation calculates the variance along the selected axis,
     * reducing that axis to length 1 while preserving all other dimensions.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the variance operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisVar(String, String, int)
     */
    default CuBridge axisVar(String a, String out, int axis) {
        if (CuBridgeJNI.axisVar(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | axisVar | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **AxisVar — Overload using a Tensor object**
     *
     * Performs variance computation on the given {@link Tensor} object along the specified {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1 by computing the variance of values along that axis;
     * all other dimensions are preserved.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the variance operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisVar(String, String, int)
     */
    default CuBridge axisVar(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return axisVar(aName, out, axis);
    }

    /**
     * **AxisVarI — Immediate variance reduction along a specified axis**
     *
     * Immediately performs variance computation on the most recent tensor in the queue
     * along the given {@code axis} and returns the result tensor directly.
     * <p>
     * Only the specified axis is reduced to size 1 by computing the variance along that axis.
     * </p>
     *
     * @param axis The axis along which the variance operation is performed.
     * @return A {@link Tensor} containing the axis-wise variance result.
     * @see #axisVar(String, String, int)
     */
    default Tensor axisVarI(int axis) {
        String oName = genRandomNameAxis();
        return axisVar("", oName, axis).get(oName);
    }

    /**
     * **AxisVarI — Immediate variance reduction on a named tensor**
     *
     * Performs immediate variance computation on the tensor {@code a} along the specified {@code axis},
     * reducing that axis to size 1, and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which the variance operation is performed.
     * @return A {@link Tensor} containing the axis-wise variance result.
     * @see #axisVar(String, String, int)
     */
    default Tensor axisVarI(String a, int axis) {
        String oName = genRandomNameAxis();
        return axisVar(a, oName, axis).get(oName);
    }

    /**
     * **AxisVarI — Immediate variance reduction using a Tensor object**
     *
     * Performs variance computation on a {@link Tensor} object along the specified {@code axis}
     * and returns the result tensor directly.
     * <p>
     * The operation computes the variance of values along the chosen axis, reducing it to length 1,
     * while preserving all other dimensions.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which the variance operation is performed.
     * @return A {@link Tensor} containing the axis-wise variance result.
     * @see #axisVar(String, String, int)
     */
    default Tensor axisVarI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return axisVar(aName, oName, axis).get(oName);
    }


    /**
     * **AxisStd — Basic standard deviation reduction along a specified axis**
     *
     * Performs standard deviation computation along the given {@code axis} of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1 by computing the standard deviation of its elements,
     * while all other dimensions remain unchanged.
     * </p>
     *
     * @param axis The axis along which the standard deviation operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisStd(String, String, int)
     */
    default CuBridge axisStd(int axis) {
        return axisStd("", genRandomNameAxis(), axis);
    }

    /**
     * **AxisStd — Standard deviation reduction on a named tensor**
     *
     * Performs standard deviation computation on the specified tensor {@code a} along the given {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The operation calculates the standard deviation along the selected axis,
     * reducing that axis to length 1 while preserving all other dimensions.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the standard deviation operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisStd(String, String, int)
     */
    default CuBridge axisStd(String a, String out, int axis) {
        if (CuBridgeJNI.axisStd(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | axisStd | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **AxisStd — Overload using a Tensor object**
     *
     * Performs standard deviation computation on the given {@link Tensor} object along the specified {@code axis}
     * and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1 by computing the standard deviation of values along that axis;
     * all other dimensions are preserved.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting tensor.
     * @param axis  The axis along which the standard deviation operation is performed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #axisStd(String, String, int)
     */
    default CuBridge axisStd(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return axisStd(aName, out, axis);
    }

    /**
     * **AxisStdI — Immediate standard deviation reduction along a specified axis**
     *
     * Immediately performs standard deviation computation on the most recent tensor in the queue
     * along the given {@code axis} and returns the result tensor directly.
     * <p>
     * Only the specified axis is reduced to size 1 by computing the standard deviation along that axis.
     * </p>
     *
     * @param axis The axis along which the standard deviation operation is performed.
     * @return A {@link Tensor} containing the axis-wise standard deviation result.
     * @see #axisStd(String, String, int)
     */
    default Tensor axisStdI(int axis) {
        String oName = genRandomNameAxis();
        return axisStd("", oName, axis).get(oName);
    }

    /**
     * **AxisStdI — Immediate standard deviation reduction on a named tensor**
     *
     * Performs immediate standard deviation computation on the tensor {@code a} along the specified {@code axis},
     * reducing that axis to size 1, and returns the result tensor.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which the standard deviation operation is performed.
     * @return A {@link Tensor} containing the axis-wise standard deviation result.
     * @see #axisStd(String, String, int)
     */
    default Tensor axisStdI(String a, int axis) {
        String oName = genRandomNameAxis();
        return axisStd(a, oName, axis).get(oName);
    }

    /**
     * **AxisStdI — Immediate standard deviation reduction using a Tensor object**
     *
     * Performs standard deviation computation on a {@link Tensor} object along the specified {@code axis}
     * and returns the result tensor directly.
     * <p>
     * The operation computes the standard deviation of values along the chosen axis, reducing it to length 1,
     * while preserving all other dimensions.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which the standard deviation operation is performed.
     * @return A {@link Tensor} containing the axis-wise standard deviation result.
     * @see #axisStd(String, String, int)
     */
    default Tensor axisStdI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return axisStd(aName, oName, axis).get(oName);
    }


    /**
     * **ArgMax — Basic index reduction along a specified axis**
     *
     * Finds the indices of the maximum values along the given {@code axis}
     * of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1, where each element represents
     * the index of the maximum value along that axis.
     * </p>
     *
     * @param axis The axis along which the maximum index is computed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #argMax(String, String, int)
     */
    default CuBridge argMax(int axis) {
        return argMax("", genRandomNameAxis(), axis);
    }

    /**
     * **ArgMax — Index reduction on a named tensor**
     *
     * Finds the indices of the maximum values in tensor {@code a}
     * along the specified {@code axis}, and stores the result in {@code out}.
     * <p>
     * The operation reduces the selected axis to size 1,
     * replacing each entry with the position of the maximum element along that axis.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting index tensor.
     * @param axis  The axis along which the maximum index is computed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #argMax(String, String, int)
     */
    default CuBridge argMax(String a, String out, int axis) {
        if (CuBridgeJNI.argMax(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | argMax | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **ArgMax — Overload using a Tensor object**
     *
     * Finds the indices of the maximum values in the given {@link Tensor} object
     * along the specified {@code axis}, and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1, and the resulting tensor contains
     * the positional indices of the maximum values along that axis.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting index tensor.
     * @param axis  The axis along which the maximum index is computed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #argMax(String, String, int)
     */
    default CuBridge argMax(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return argMax(aName, out, axis);
    }

    /**
     * **ArgMaxI — Immediate index reduction along a specified axis**
     *
     * Immediately finds the indices of the maximum values along the given {@code axis}
     * of the most recent tensor in the queue, and returns the resulting index tensor.
     * <p>
     * The specified axis is reduced to size 1, containing the indices of the maximum values.
     * </p>
     *
     * @param axis The axis along which the maximum index is computed.
     * @return A {@link Tensor} containing the index positions of maximum values.
     * @see #argMax(String, String, int)
     */
    default Tensor argMaxI(int axis) {
        String oName = genRandomNameAxis();
        return argMax("", oName, axis).get(oName);
    }

    /**
     * **ArgMaxI — Immediate index reduction on a named tensor**
     *
     * Immediately computes the indices of the maximum values in tensor {@code a}
     * along the specified {@code axis}, reducing that axis to size 1,
     * and returns the result as a {@link Tensor}.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which the maximum index is computed.
     * @return A {@link Tensor} containing the index positions of maximum values.
     * @see #argMax(String, String, int)
     */
    default Tensor argMaxI(String a, int axis) {
        String oName = genRandomNameAxis();
        return argMax(a, oName, axis).get(oName);
    }

    /**
     * **ArgMaxI — Immediate index reduction using a Tensor object**
     *
     * Immediately computes the indices of the maximum values in a {@link Tensor} object
     * along the specified {@code axis}, and returns the resulting index tensor directly.
     * <p>
     * The selected axis is reduced to size 1, with each element representing
     * the index of the maximum value along that axis.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which the maximum index is computed.
     * @return A {@link Tensor} containing the index positions of maximum values.
     * @see #argMax(String, String, int)
     */
    default Tensor argMaxI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return argMax(aName, oName, axis).get(oName);
    }


    /**
     * **ArgMin — Basic index reduction along a specified axis**
     *
     * Finds the indices of the minimum values along the given {@code axis}
     * of the most recent tensor stored in the internal queue.
     * <p>
     * Only the specified axis is reduced to size 1, where each element represents
     * the index of the minimum value along that axis.
     * </p>
     *
     * @param axis The axis along which the minimum index is computed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #argMin(String, String, int)
     */
    default CuBridge argMin(int axis) {
        return argMin("", genRandomNameAxis(), axis);
    }

    /**
     * **ArgMin — Index reduction on a named tensor**
     *
     * Finds the indices of the minimum values in tensor {@code a}
     * along the specified {@code axis}, and stores the result in {@code out}.
     * <p>
     * The operation reduces the selected axis to size 1,
     * replacing each entry with the position of the minimum element along that axis.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting index tensor.
     * @param axis  The axis along which the minimum index is computed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #argMin(String, String, int)
     */
    default CuBridge argMin(String a, String out, int axis) {
        if (CuBridgeJNI.argMin(a, out, axis)) return CuBridge.getInstance();
        else System.err.println("Error | argMin | " + a + " | " + out + " | " + axis);
        return null;
    }

    /**
     * **ArgMin — Overload using a Tensor object**
     *
     * Finds the indices of the minimum values in the given {@link Tensor} object
     * along the specified {@code axis}, and stores the result in {@code out}.
     * <p>
     * The selected axis is reduced to size 1, and the resulting tensor contains
     * the positional indices of the minimum values along that axis.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the resulting index tensor.
     * @param axis  The axis along which the minimum index is computed.
     * @return A {@link CuBridge} instance representing the result of the operation.
     * @see #argMin(String, String, int)
     */
    default CuBridge argMin(Tensor a, String out, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        return argMin(aName, out, axis);
    }

    /**
     * **ArgMinI — Immediate index reduction along a specified axis**
     *
     * Immediately finds the indices of the minimum values along the given {@code axis}
     * of the most recent tensor in the queue, and returns the resulting index tensor.
     * <p>
     * The specified axis is reduced to size 1, containing the indices of the minimum values.
     * </p>
     *
     * @param axis The axis along which the minimum index is computed.
     * @return A {@link Tensor} containing the index positions of minimum values.
     * @see #argMin(String, String, int)
     */
    default Tensor argMinI(int axis) {
        String oName = genRandomNameAxis();
        return argMin("", oName, axis).get(oName);
    }

    /**
     * **ArgMinI — Immediate index reduction on a named tensor**
     *
     * Immediately computes the indices of the minimum values in tensor {@code a}
     * along the specified {@code axis}, reducing that axis to size 1,
     * and returns the result as a {@link Tensor}.
     *
     * @param a    The name of the input tensor.
     * @param axis The axis along which the minimum index is computed.
     * @return A {@link Tensor} containing the index positions of minimum values.
     * @see #argMin(String, String, int)
     */
    default Tensor argMinI(String a, int axis) {
        String oName = genRandomNameAxis();
        return argMin(a, oName, axis).get(oName);
    }

    /**
     * **ArgMinI — Immediate index reduction using a Tensor object**
     *
     * Immediately computes the indices of the minimum values in a {@link Tensor} object
     * along the specified {@code axis}, and returns the resulting index tensor directly.
     * <p>
     * The selected axis is reduced to size 1, with each element representing
     * the index of the minimum value along that axis.
     * </p>
     *
     * @param a    The input {@link Tensor}.
     * @param axis The axis along which the minimum index is computed.
     * @return A {@link Tensor} containing the index positions of minimum values.
     * @see #argMin(String, String, int)
     */
    default Tensor argMinI(Tensor a, int axis) {
        String aName = genRandomNameAxis(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAxis();
        return argMin(aName, oName, axis).get(oName);
    }



}
