package CuBridge;

import java.util.UUID;

public interface UtilityOps{

    private String genRandomNameUtility() {
        return "UtilityOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /**
     * **Clip — Clamp tensor values within a range**
     *
     * Clamps all elements of the most recent tensor in the queue
     * between the specified {@code alpha} (minimum) and {@code beta} (maximum) values.
     * <p>
     * Typically used when a tensor is already stored in the internal queue.
     * </p>
     *
     * @param alpha The minimum clipping value.
     * @param beta  The maximum clipping value.
     * @return A {@link CuBridge} instance representing the clipping operation.
     * @see #clip(String, String, float, float)
     */
    default CuBridge clip(float alpha, float beta) {
        return clip("", genRandomNameUtility(), alpha, beta);
    }

    /**
     * **Clip — Clamp tensor values within a range**
     *
     * Clamps all elements of tensor {@code a} between {@code alpha} and {@code beta},
     * and stores the result in {@code out}.
     * <p>
     * Each value smaller than {@code alpha} becomes {@code alpha}, and each value
     * larger than {@code beta} becomes {@code beta}.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The minimum clipping value.
     * @param beta  The maximum clipping value.
     * @return A {@link CuBridge} instance representing the clipping operation.
     * @see #clip(Tensor, String, float, float)
     */
    default CuBridge clip(String a, String out, float alpha, float beta) {
        if (CuBridgeJNI.clip(a, out, alpha, beta)) return CuBridge.getInstance();
        else System.err.println("Error | clip | " + a + " | " + out + " | " + alpha + " | " + beta);
        return null;
    }

    /**
     * **Clip — Clamp tensor values within a range (Tensor input)**
     *
     * Clamps all elements of the given {@link Tensor} {@code a}
     * between {@code alpha} and {@code beta}, and stores the result in {@code out}.
     * <p>
     * Automatically assigns a random internal name to {@code a} before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The minimum clipping value.
     * @param beta  The maximum clipping value.
     * @return A {@link CuBridge} instance representing the clipping operation.
     * @see #clip(String, String, float, float)
     */
    default CuBridge clip(Tensor a, String out, float alpha, float beta) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        return clip(aName, out, alpha, beta);
    }

    /**
     * **ClipI — Immediate tensor clipping**
     *
     * Immediately clamps all elements of the most recent tensor in the queue
     * between {@code alpha} and {@code beta}, and directly returns the resulting {@link Tensor}.
     * <p>
     * This method performs the clipping operation without modifying the original tensor.
     * </p>
     *
     * @param alpha The minimum clipping value.
     * @param beta  The maximum clipping value.
     * @return A clipped {@link Tensor}.
     * @see #clip(float, float)
     */
    default Tensor clipI(float alpha, float beta) {
        String oName = genRandomNameUtility();
        return clip("", oName, alpha, beta).get(oName);
    }

    /**
     * **ClipI — Immediate tensor clipping (named tensor)**
     *
     * Immediately clamps all elements of the named tensor {@code a}
     * between {@code alpha} and {@code beta}, and directly returns the resulting {@link Tensor}.
     * <p>
     * Useful when clipping a tensor already registered in the queue.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param alpha The minimum clipping value.
     * @param beta  The maximum clipping value.
     * @return A clipped {@link Tensor}.
     * @see #clip(String, String, float, float)
     */
    default Tensor clipI(String a, float alpha, float beta) {
        String oName = genRandomNameUtility();
        return clip(a, oName, alpha, beta).get(oName);
    }

    /**
     * **ClipI — Immediate tensor clipping (Tensor input)**
     *
     * Immediately clamps all elements of the given {@link Tensor} {@code a}
     * between {@code alpha} and {@code beta}, and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param alpha The minimum clipping value.
     * @param beta  The maximum clipping value.
     * @return A clipped {@link Tensor}.
     * @see #clip(Tensor, String, float, float)
     */
    default Tensor clipI(Tensor a, float alpha, float beta) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUtility();
        return clip(aName, oName, alpha, beta).get(oName);
    }


    /**
     * **SoftClip — Smooth amplitude compression**
     *
     * Applies a smooth, non-linear clipping operation to the most recent tensor in the queue.
     * <p>
     * The soft clipping operation compresses the amplitude using a hyperbolic tangent–like curve,
     * preserving signal continuity while preventing sharp saturation.
     * </p>
     *
     * @param alpha The softness coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the soft clipping operation.
     * @see #softClip(String, String, float)
     */
    default CuBridge softClip(float alpha) {
        return softClip("", genRandomNameUtility(), alpha);
    }

    /**
     * **SoftClip — Smooth amplitude compression for a named tensor**
     *
     * Applies soft clipping to the specified tensor {@code a} using {@code alpha},
     * and stores the result in {@code out}.
     * <p>
     * This operation smoothly limits the amplitude while maintaining a continuous curve.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The softness coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the soft clipping operation.
     * @see #softClip(Tensor, String, float)
     */
    default CuBridge softClip(String a, String out, float alpha) {
        if (CuBridgeJNI.softClip(a, out, alpha)) return CuBridge.getInstance();
        else System.err.println("Error | softClip | " + a + " | " + out + " | " + alpha);
        return null;
    }

    /**
     * **SoftClip — Smooth amplitude compression for a tensor input**
     *
     * Applies soft clipping to the given {@link Tensor} {@code a} using {@code alpha},
     * and stores the result in {@code out}.
     * <p>
     * Automatically assigns a random internal name to {@code a} before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The softness coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the soft clipping operation.
     * @see #softClip(String, String, float)
     */
    default CuBridge softClip(Tensor a, String out, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        return softClip(aName, out, alpha);
    }

    /**
     * **SoftClipI — Immediate smooth amplitude compression**
     *
     * Immediately applies soft clipping to the most recent tensor in the queue
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * The operation uses {@code alpha} to control the degree of compression.
     * </p>
     *
     * @param alpha The softness coefficient controlling the compression strength.
     * @return A soft-clipped {@link Tensor}.
     * @see #softClip(String, String, float)
     */
    default Tensor softClipI(float alpha) {
        String oName = genRandomNameUtility();
        return softClip("", oName, alpha).get(oName);
    }

    /**
     * **SoftClipI — Immediate smooth amplitude compression of a named tensor**
     *
     * Immediately applies soft clipping to tensor {@code a}
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This function performs a smooth non-linear amplitude limiting based on {@code alpha}.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param alpha The softness coefficient controlling the compression strength.
     * @return A soft-clipped {@link Tensor}.
     * @see #softClip(String, String, float)
     */
    default Tensor softClipI(String a, float alpha) {
        String oName = genRandomNameUtility();
        return softClip(a, oName, alpha).get(oName);
    }

    /**
     * **SoftClipI — Immediate smooth amplitude compression of a tensor input**
     *
     * Immediately applies soft clipping to the given {@link Tensor} object
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param alpha The softness coefficient controlling the compression strength.
     * @return A soft-clipped {@link Tensor}.
     * @see #softClip(String, String, float)
     */
    default Tensor softClipI(Tensor a, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUtility();
        return softClip(aName, oName, alpha).get(oName);
    }


    /**
     * **SigClip — Sigmoid-based dynamic range compression**
     *
     * Applies a sigmoid-shaped amplitude compression to the most recent tensor in the queue.
     * <p>
     * The sigmoid clipping operation uses a logistic function scaled by {@code alpha}
     * to softly limit large magnitude values while maintaining differentiability.
     * </p>
     *
     * @param alpha The sigmoid gain factor controlling the curve steepness.
     * @return A {@link CuBridge} instance representing the sigmoid clipping operation.
     * @see #sigClip(String, String, float)
     */
    default CuBridge sigClip(float alpha) {
        return sigClip("", genRandomNameUtility(), alpha);
    }

    /**
     * **SigClip — Sigmoid-based dynamic range compression for a named tensor**
     *
     * Applies a sigmoid clipping operation to tensor {@code a} using {@code alpha},
     * and stores the result in {@code out}.
     * <p>
     * This operation compresses high-amplitude regions using a smooth logistic curve.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The sigmoid gain factor controlling the curve steepness.
     * @return A {@link CuBridge} instance representing the sigmoid clipping operation.
     * @see #sigClip(Tensor, String, float)
     */
    default CuBridge sigClip(String a, String out, float alpha) {
        if (CuBridgeJNI.sigClip(a, out, alpha)) return CuBridge.getInstance();
        else System.err.println("Error | sigClip | " + a + " | " + out + " | " + alpha);
        return null;
    }

    /**
     * **SigClip — Sigmoid-based dynamic range compression for a tensor input**
     *
     * Applies sigmoid clipping to the given {@link Tensor} {@code a} using {@code alpha},
     * and stores the result in {@code out}.
     * <p>
     * Automatically assigns a random internal name to {@code a} before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The sigmoid gain factor controlling the curve steepness.
     * @return A {@link CuBridge} instance representing the sigmoid clipping operation.
     * @see #sigClip(String, String, float)
     */
    default CuBridge sigClip(Tensor a, String out, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        return sigClip(aName, out, alpha);
    }

    /**
     * **SigClipI — Immediate sigmoid-based dynamic range compression**
     *
     * Immediately applies sigmoid clipping to the most recent tensor
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * The {@code alpha} parameter defines the slope of the logistic curve.
     * </p>
     *
     * @param alpha The sigmoid gain factor controlling the curve steepness.
     * @return A sigmoid-clipped {@link Tensor}.
     * @see #sigClip(String, String, float)
     */
    default Tensor sigClipI(float alpha) {
        String oName = genRandomNameUtility();
        return sigClip("", oName, alpha).get(oName);
    }

    /**
     * **SigClipI — Immediate sigmoid-based dynamic range compression of a named tensor**
     *
     * Immediately applies sigmoid clipping to tensor {@code a}
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This operation uses {@code alpha} to control the degree of non-linear compression.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param alpha The sigmoid gain factor controlling the curve steepness.
     * @return A sigmoid-clipped {@link Tensor}.
     * @see #sigClip(String, String, float)
     */
    default Tensor sigClipI(String a, float alpha) {
        String oName = genRandomNameUtility();
        return sigClip(a, oName, alpha).get(oName);
    }

    /**
     * **SigClipI — Immediate sigmoid-based dynamic range compression of a tensor input**
     *
     * Immediately applies sigmoid clipping to the given {@link Tensor} object
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param alpha The sigmoid gain factor controlling the curve steepness.
     * @return A sigmoid-clipped {@link Tensor}.
     * @see #sigClip(String, String, float)
     */
    default Tensor sigClipI(Tensor a, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUtility();
        return sigClip(aName, oName, alpha).get(oName);
    }


    /**
     * **TanhClip — Hyperbolic tangent–based amplitude compression**
     *
     * Applies a hyperbolic tangent–based amplitude compression to the most recent tensor in the queue.
     * <p>
     * Each element is transformed using {@code y = tanh(alpha * x)}, producing a smooth
     * saturation curve that limits values within [-1, 1] while maintaining continuity.
     * </p>
     *
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the tanh clipping operation.
     * @see #tanhClip(String, String, float)
     */
    default CuBridge tanhClip(float alpha) {
        return tanhClip("", genRandomNameUtility(), alpha);
    }

    /**
     * **TanhClip — Hyperbolic tangent–based amplitude compression (named tensor)**
     *
     * Applies a tanh-based amplitude compression to the named tensor {@code a}
     * using {@code alpha}, and stores the result in {@code out}.
     * <p>
     * The tanh function is widely used for non-linear dynamic range limiting,
     * providing smooth compression of large magnitudes.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the tanh clipping operation.
     * @see #tanhClip(Tensor, String, float)
     */
    default CuBridge tanhClip(String a, String out, float alpha) {
        if (CuBridgeJNI.tanhClip(a, out, alpha)) return CuBridge.getInstance();
        else System.err.println("Error | tanhClip | " + a + " | " + out + " | " + alpha);
        return null;
    }

    /**
     * **TanhClip — Hyperbolic tangent–based amplitude compression (Tensor input)**
     *
     * Applies tanh-based amplitude compression to the given {@link Tensor} object
     * using {@code alpha}, and stores the result in {@code out}.
     * <p>
     * Automatically assigns a random internal name to {@code a} before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the tanh clipping operation.
     * @see #tanhClip(String, String, float)
     */
    default CuBridge tanhClip(Tensor a, String out, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        return tanhClip(aName, out, alpha);
    }

    /**
     * **TanhClipI — Immediate hyperbolic tangent–based amplitude compression**
     *
     * Immediately applies tanh-based amplitude compression to the most recent tensor
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This function performs {@code y = tanh(alpha * x)} element-wise and returns
     * a smooth-limited tensor in range [-1, 1].
     * </p>
     *
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A tanh-clipped {@link Tensor}.
     * @see #tanhClip(String, String, float)
     */
    default Tensor tanhClipI(float alpha) {
        String oName = genRandomNameUtility();
        return tanhClip("", oName, alpha).get(oName);
    }

    /**
     * **TanhClipI — Immediate hyperbolic tangent–based compression (named tensor)**
     *
     * Immediately applies tanh-based amplitude compression to the named tensor {@code a}
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This operation is suitable for smooth non-linear limiting in real-time signal processing.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A tanh-clipped {@link Tensor}.
     * @see #tanhClip(String, String, float)
     */
    default Tensor tanhClipI(String a, float alpha) {
        String oName = genRandomNameUtility();
        return tanhClip(a, oName, alpha).get(oName);
    }

    /**
     * **TanhClipI — Immediate hyperbolic tangent–based compression (Tensor input)**
     *
     * Immediately applies tanh-based amplitude compression to the given {@link Tensor} object
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A tanh-clipped {@link Tensor}.
     * @see #tanhClip(Tensor, String, float)
     */
    default Tensor tanhClipI(Tensor a, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUtility();
        return tanhClip(aName, oName, alpha).get(oName);
    }


    /**
     * **LogClip — Logarithmic amplitude compression**
     *
     * Applies a logarithmic non-linear amplitude compression to the most recent tensor in the queue.
     * <p>
     * Each element is transformed using {@code y = sign(x) * log(1 + alpha * |x|)}, effectively
     * compressing large magnitudes while retaining the resolution of small signals.
     * </p>
     *
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the log clipping operation.
     * @see #logClip(String, String, float)
     */
    default CuBridge logClip(float alpha) {
        return logClip("", genRandomNameUtility(), alpha);
    }

    /**
     * **LogClip — Logarithmic amplitude compression (named tensor)**
     *
     * Applies logarithmic amplitude compression to the named tensor {@code a}
     * using {@code alpha}, and stores the result in {@code out}.
     * <p>
     * This non-linear transformation is often used in dynamic range control
     * and perceptual signal processing.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the log clipping operation.
     * @see #logClip(Tensor, String, float)
     */
    default CuBridge logClip(String a, String out, float alpha) {
        if (CuBridgeJNI.logClip(a, out, alpha)) return CuBridge.getInstance();
        else System.err.println("Error | logClip | " + a + " | " + out + " | " + alpha);
        return null;
    }

    /**
     * **LogClip — Logarithmic amplitude compression (Tensor input)**
     *
     * Applies logarithmic amplitude compression to the given {@link Tensor} object
     * using {@code alpha}, and stores the result in {@code out}.
     * <p>
     * Automatically assigns a random internal name to {@code a} before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the clipped output tensor.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A {@link CuBridge} instance representing the log clipping operation.
     * @see #logClip(String, String, float)
     */
    default CuBridge logClip(Tensor a, String out, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        return logClip(aName, out, alpha);
    }

    /**
     * **LogClipI — Immediate logarithmic amplitude compression**
     *
     * Immediately applies logarithmic amplitude compression to the most recent tensor
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Each element is transformed using {@code y = sign(x) * log(1 + alpha * |x|)}.
     * </p>
     *
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A log-compressed {@link Tensor}.
     * @see #logClip(String, String, float)
     */
    default Tensor logClipI(float alpha) {
        String oName = genRandomNameUtility();
        return logClip("", oName, alpha).get(oName);
    }

    /**
     * **LogClipI — Immediate logarithmic amplitude compression (named tensor)**
     *
     * Immediately applies logarithmic amplitude compression to the named tensor {@code a}
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Suitable for audio-style logarithmic dynamic range compression or signal scaling.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A log-compressed {@link Tensor}.
     * @see #logClip(String, String, float)
     */
    default Tensor logClipI(String a, float alpha) {
        String oName = genRandomNameUtility();
        return logClip(a, oName, alpha).get(oName);
    }

    /**
     * **LogClipI — Immediate logarithmic amplitude compression (Tensor input)**
     *
     * Immediately applies logarithmic amplitude compression to the given {@link Tensor} object
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names before execution.
     * </p>
     *
     * @param a     The input {@link Tensor}.
     * @param alpha The scaling coefficient controlling the compression strength.
     * @return A log-compressed {@link Tensor}.
     * @see #logClip(Tensor, String, float)
     */
    default Tensor logClipI(Tensor a, float alpha) {
        String aName = genRandomNameUtility(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUtility();
        return logClip(aName, oName, alpha).get(oName);
    }

}
