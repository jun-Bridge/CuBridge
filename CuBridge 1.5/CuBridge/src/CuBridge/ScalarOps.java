package CuBridge;

import java.util.UUID;

public interface ScalarOps {

    private String genRandomNameScalar() {
        return "ScalarOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }



    /**
     * **L1Norm — Core scalar L1 norm operation**
     *
     * Computes the L1 norm (sum of absolute values) of the specified tensor
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * L1 = sum(|x_i|)
     * </pre>
     * The resulting tensor has shape (1,) and represents a scalar value.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge L1Norm(String a, String out) {
        if (CuBridgeJNI.L1Norm(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | L1Norm | " + a + " | " + out);
        return null;
    }

    /**
     * **L1Norm — Overload using a Tensor object**
     *
     * Computes the L1 norm (sum of absolute values) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #L1Norm(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L1 norm computation.
     * @see #L1Norm(String, String)
     */
    default CuBridge L1Norm(Tensor a, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return L1Norm(aName, out);
    }

    /**
     * **L1NormI — Immediate L1 norm computation on a named tensor**
     *
     * Computes the L1 norm (sum of absolute values) of the specified tensor
     * and directly returns the resulting scalar {@link Tensor}.
     *
     * @param a   The name of the input tensor.
     * @return A {@link Tensor} representing the scalar L1 norm value.
     * @see #L1Norm(String, String)
     */
    default Tensor L1NormI(String a) {
        String oName = genRandomNameScalar();
        return L1Norm(a, oName).get(oName);
    }

    /**
     * **L1NormI — Immediate L1 norm computation on a Tensor object**
     *
     * Computes the L1 norm (sum of absolute values) of the given {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #L1Norm(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @return A {@link Tensor} representing the scalar L1 norm value.
     * @see #L1Norm(String, String)
     */
    default Tensor L1NormI(Tensor a) {
        String aName = genRandomNameScalar();
        String oName = genRandomNameScalar();
        CuBridge.getInstance().put(a, aName);
        return L1Norm(aName, oName).get(oName);
    }


    /**
     * **L2Norm — Core scalar L2 norm operation**
     *
     * Computes the L2 norm (Euclidean norm) of the specified tensor
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * L2 = sqrt(sum(x_i^2))
     * </pre>
     * The resulting tensor has shape (1,) and represents a scalar value.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge L2Norm(String a, String out) {
        if (CuBridgeJNI.L2Norm(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | L2Norm | " + a + " | " + out);
        return null;
    }

    /**
     * **L2Norm — Overload using a Tensor object**
     *
     * Computes the L2 norm (Euclidean norm) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #L2Norm(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L2 norm computation.
     * @see #L2Norm(String, String)
     */
    default CuBridge L2Norm(Tensor a, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return L2Norm(aName, out);
    }

    /**
     * **L2NormI — Immediate L2 norm computation on a named tensor**
     *
     * Computes the L2 norm (Euclidean norm) of the specified tensor
     * and directly returns the resulting scalar {@link Tensor}.
     *
     * @param a   The name of the input tensor.
     * @return A {@link Tensor} representing the scalar L2 norm value.
     * @see #L2Norm(String, String)
     */
    default Tensor L2NormI(String a)  {
        String oName = genRandomNameScalar();
        return L2Norm(a, oName).get(oName);
    }

    /**
     * **L2NormI — Immediate L2 norm computation on a Tensor object**
     *
     * Computes the L2 norm (Euclidean norm) of the given {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #L2Norm(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @return A {@link Tensor} representing the scalar L2 norm value.
     * @see #L2Norm(String, String)
     */
    default Tensor L2NormI(Tensor a) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return L2Norm(aName, oName).get(oName);
    }


    /**
     * **LinfNorm — Core scalar L∞ norm operation**
     *
     * Computes the L∞ norm (maximum absolute value) of the specified tensor
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * L∞ = max(|x_i|)
     * </pre>
     * The resulting tensor has shape (1,) and represents a scalar value.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge LinfNorm(String a, String out) {
        if (CuBridgeJNI.LinfNorm(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | LinfNorm | " + a + " | " + out);
        return null;
    }

    /**
     * **LinfNorm — Overload using a Tensor object**
     *
     * Computes the L∞ norm (maximum absolute value) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #LinfNorm(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Linf norm computation.
     * @see #LinfNorm(String, String)
     */
    default CuBridge LinfNorm(Tensor a, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return LinfNorm(aName, out);
    }

    /**
     * **LinfNormI — Immediate L∞ norm computation on a named tensor**
     *
     * Computes the L∞ norm (maximum absolute value) of the specified tensor
     * and directly returns the resulting scalar {@link Tensor}.
     *
     * @param a   The name of the input tensor.
     * @return A {@link Tensor} representing the scalar L∞ norm value.
     * @see #LinfNorm(String, String)
     */
    default Tensor LinfNormI(String a)  {
        String oName = genRandomNameScalar();
        return LinfNorm(a, oName).get(oName);
    }

    /**
     * **LinfNormI — Immediate L∞ norm computation on a Tensor object**
     *
     * Computes the L∞ norm (maximum absolute value) of the given {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #LinfNorm(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @return A {@link Tensor} representing the scalar L∞ norm value.
     * @see #LinfNorm(String, String)
     */
    default Tensor LinfNormI(Tensor a) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return LinfNorm(aName, oName).get(oName);
    }


    /**
     * **L1Dist — Core L1 distance operation**
     *
     * Computes the L1 distance (Manhattan distance) between two tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * L1Dist = sum(|a_i - b_i|)
     * </pre>
     * The resulting tensor has shape (1,).
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge L1Dist(String a, String b, String out) {
        if (CuBridgeJNI.L1Dist(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | L1Dist | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **L1Dist — Overload using Tensor b**
     *
     * Computes the L1 distance between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #L1Dist(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L1 distance operation.
     * @see #L1Dist(String, String, String)
     */
    default CuBridge L1Dist(String a, Tensor b, String out) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return L1Dist(a, bName, out);
    }

    /**
     * **L1Dist — Overload using Tensor a**
     *
     * Computes the L1 distance between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #L1Dist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L1 distance operation.
     * @see #L1Dist(String, String, String)
     */
    default CuBridge L1Dist(Tensor a, String b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return L1Dist(aName, b, out);
    }

    /**
     * **L1Dist — Overload using Tensor a and Tensor b**
     *
     * Computes the L1 distance between two {@link Tensor} objects.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #L1Dist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L1 distance operation.
     * @see #L1Dist(String, String, String)
     */
    default CuBridge L1Dist(Tensor a, Tensor b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return L1Dist(aName, bName, out);
    }

    /**
     * **L1DistI — Immediate L1 distance operation (String a, String b)**
     *
     * Computes the L1 distance between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar L1 distance.
     * @see #L1Dist(String, String, String)
     */
    default Tensor L1DistI(String a, String b) {
        String oName = genRandomNameScalar();
        return L1Dist(a, b, oName).get(oName);
    }

    /**
     * **L1DistI — Immediate L1 distance operation (String a, Tensor b)**
     *
     * Computes the L1 distance between a named tensor and a {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #L1Dist(String, String, String)}.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar L1 distance.
     * @see #L1Dist(String, String, String)
     */
    default Tensor L1DistI(String a, Tensor b) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return L1Dist(a, bName, oName).get(oName);
    }

    /**
     * **L1DistI — Immediate L1 distance operation (Tensor a, String b)**
     *
     * Computes the L1 distance between a {@link Tensor} object and a named tensor
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #L1Dist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar L1 distance.
     * @see #L1Dist(String, String, String)
     */
    default Tensor L1DistI(Tensor a, String b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return L1Dist(aName, b, oName).get(oName);
    }

    /**
     * **L1DistI — Immediate L1 distance operation (Tensor a, Tensor b)**
     *
     * Computes the L1 distance between two {@link Tensor} objects and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #L1Dist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar L1 distance.
     * @see #L1Dist(String, String, String)
     */
    default Tensor L1DistI(Tensor a, Tensor b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return L1Dist(aName, bName, oName).get(oName);
    }


    /**
     * **L2Dist — Core L2 (Euclidean) distance operation**
     *
     * Computes the L2 distance (Euclidean distance) between two tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * L2Dist = sqrt(sum((a_i - b_i)^2))
     * </pre>
     * The resulting tensor has shape (1,).
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge L2Dist(String a, String b, String out) {
        if (CuBridgeJNI.L2Dist(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | L2Dist | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **L2Dist — Overload using Tensor b**
     *
     * Computes the L2 distance between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #L2Dist(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L2 distance operation.
     * @see #L2Dist(String, String, String)
     */
    default CuBridge L2Dist(String a, Tensor b, String out) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return L2Dist(a, bName, out);
    }

    /**
     * **L2Dist — Overload using Tensor a**
     *
     * Computes the L2 distance between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #L2Dist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L2 distance operation.
     * @see #L2Dist(String, String, String)
     */
    default CuBridge L2Dist(Tensor a, String b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return L2Dist(aName, b, out);
    }

    /**
     * **L2Dist — Overload using Tensor a and Tensor b**
     *
     * Computes the L2 distance between two {@link Tensor} objects.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #L2Dist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L2 distance operation.
     * @see #L2Dist(String, String, String)
     */
    default CuBridge L2Dist(Tensor a, Tensor b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return L2Dist(aName, bName, out);
    }

    /**
     * **L2DistI — Immediate L2 distance operation (String a, String b)**
     *
     * Computes the L2 distance between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar L2 distance.
     * @see #L2Dist(String, String, String)
     */
    default Tensor L2DistI(String a, String b) {
        String oName = genRandomNameScalar();
        return L2Dist(a, b, oName).get(oName);
    }

    /**
     * **L2DistI — Immediate L2 distance operation (String a, Tensor b)**
     *
     * Computes the L2 distance between a named tensor and a {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #L2Dist(String, String, String)}.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar L2 distance.
     * @see #L2Dist(String, String, String)
     */
    default Tensor L2DistI(String a, Tensor b) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return L2Dist(a, bName, oName).get(oName);
    }

    /**
     * **L2DistI — Immediate L2 distance operation (Tensor a, String b)**
     *
     * Computes the L2 distance between a {@link Tensor} object and a named tensor
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #L2Dist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar L2 distance.
     * @see #L2Dist(String, String, String)
     */
    default Tensor L2DistI(Tensor a, String b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return L2Dist(aName, b, oName).get(oName);
    }

    /**
     * **L2DistI — Immediate L2 distance operation (Tensor a, Tensor b)**
     *
     * Computes the L2 distance between two {@link Tensor} objects and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #L2Dist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar L2 distance.
     * @see #L2Dist(String, String, String)
     */
    default Tensor L2DistI(Tensor a, Tensor b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return L2Dist(aName, bName, oName).get(oName);
    }


    /**
     * **LinfDist — Core L∞ (Chebyshev) distance operation**
     *
     * Computes the L∞ distance (Chebyshev distance, maximum absolute difference)
     * between two tensors and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * LinfDist = max(|a_i - b_i|)
     * </pre>
     * The resulting tensor has shape (1,).
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge LinfDist(String a, String b, String out) {
        if (CuBridgeJNI.LinfDist(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | LinfDist | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **LinfDist — Overload using Tensor b**
     *
     * Computes the L∞ distance between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #LinfDist(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L∞ distance operation.
     * @see #LinfDist(String, String, String)
     */
    default CuBridge LinfDist(String a, Tensor b, String out) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return LinfDist(a, bName, out);
    }

    /**
     * **LinfDist — Overload using Tensor a**
     *
     * Computes the L∞ distance between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #LinfDist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L∞ distance operation.
     * @see #LinfDist(String, String, String)
     */
    default CuBridge LinfDist(Tensor a, String b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return LinfDist(aName, b, out);
    }

    /**
     * **LinfDist — Overload using Tensor a and Tensor b**
     *
     * Computes the L∞ distance between two {@link Tensor} objects.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #LinfDist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the L∞ distance operation.
     * @see #LinfDist(String, String, String)
     */
    default CuBridge LinfDist(Tensor a, Tensor b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return LinfDist(aName, bName, out);
    }

    /**
     * **LinfDistI — Immediate L∞ distance operation (String a, String b)**
     *
     * Computes the L∞ distance between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar L∞ distance.
     * @see #LinfDist(String, String, String)
     */
    default Tensor LinfDistI(String a, String b) {
        String oName = genRandomNameScalar();
        return LinfDist(a, b, oName).get(oName);
    }

    /**
     * **LinfDistI — Immediate L∞ distance operation (String a, Tensor b)**
     *
     * Computes the L∞ distance between a named tensor and a {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #LinfDist(String, String, String)}.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar L∞ distance.
     * @see #LinfDist(String, String, String)
     */
    default Tensor LinfDistI(String a, Tensor b) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return LinfDist(a, bName, oName).get(oName);
    }

    /**
     * **LinfDistI — Immediate L∞ distance operation (Tensor a, String b)**
     *
     * Computes the L∞ distance between a {@link Tensor} object and a named tensor
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #LinfDist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar L∞ distance.
     * @see #LinfDist(String, String, String)
     */
    default Tensor LinfDistI(Tensor a, String b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return LinfDist(aName, b, oName).get(oName);
    }

    /**
     * **LinfDistI — Immediate L∞ distance operation (Tensor a, Tensor b)**
     *
     * Computes the L∞ distance between two {@link Tensor} objects and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #LinfDist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar L∞ distance.
     * @see #LinfDist(String, String, String)
     */
    default Tensor LinfDistI(Tensor a, Tensor b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return LinfDist(aName, bName, oName).get(oName);
    }


    /**
     * **CosSim — Core cosine similarity operation**
     *
     * Computes the cosine similarity between two tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * cosSim = (a · b) / (‖a‖₂ * ‖b‖₂)
     * </pre>
     * The result is a scalar value in the range [-1, 1].
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge cosSim(String a, String b, String out) {
        if (CuBridgeJNI.cosSim(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | cosSim | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **CosSim — Overload using Tensor b**
     *
     * Computes the cosine similarity between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #cosSim(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the cosine similarity operation.
     * @see #cosSim(String, String, String)
     */
    default CuBridge cosSim(String a, Tensor b, String out) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return cosSim(a, bName, out);
    }

    /**
     * **CosSim — Overload using Tensor a**
     *
     * Computes the cosine similarity between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #cosSim(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the cosine similarity operation.
     * @see #cosSim(String, String, String)
     */
    default CuBridge cosSim(Tensor a, String b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return cosSim(aName, b, out);
    }

    /**
     * **CosSim — Overload using Tensor a and Tensor b**
     *
     * Computes the cosine similarity between two {@link Tensor} objects.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #cosSim(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the cosine similarity operation.
     * @see #cosSim(String, String, String)
     */
    default CuBridge cosSim(Tensor a, Tensor b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return cosSim(aName, bName, out);
    }

    /**
     * **CosSimI — Immediate cosine similarity operation (String a, String b)**
     *
     * Computes the cosine similarity between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar cosine similarity value.
     * @see #cosSim(String, String, String)
     */
    default Tensor cosSimI(String a, String b) {
        String oName = genRandomNameScalar();
        return cosSim(a, b, oName).get(oName);
    }

    /**
     * **CosSimI — Immediate cosine similarity operation (String a, Tensor b)**
     *
     * Computes the cosine similarity between a named tensor and a {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #cosSim(String, String, String)}.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar cosine similarity value.
     * @see #cosSim(String, String, String)
     */
    default Tensor cosSimI(String a, Tensor b) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return cosSim(a, bName, oName).get(oName);
    }

    /**
     * **CosSimI — Immediate cosine similarity operation (Tensor a, String b)**
     *
     * Computes the cosine similarity between a {@link Tensor} object and a named tensor
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #cosSim(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar cosine similarity value.
     * @see #cosSim(String, String, String)
     */
    default Tensor cosSimI(Tensor a, String b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return cosSim(aName, b, oName).get(oName);
    }

    /**
     * **CosSimI — Immediate cosine similarity operation (Tensor a, Tensor b)**
     *
     * Computes the cosine similarity between two {@link Tensor} objects and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #cosSim(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar cosine similarity value.
     * @see #cosSim(String, String, String)
     */
    default Tensor cosSimI(Tensor a, Tensor b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return cosSim(aName, bName, oName).get(oName);
    }


    /**
     * **CosDist — Core cosine distance operation**
     *
     * Computes the cosine distance between two tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * cosDist = 1 - (a · b) / (‖a‖₂ * ‖b‖₂)
     * </pre>
     * The result is a scalar value in the range [0, 2].
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge cosDist(String a, String b, String out) {
        if (CuBridgeJNI.cosDist(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | cosDist | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **CosDist — Overload using Tensor b**
     *
     * Computes the cosine distance between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #cosDist(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first input tensor.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the cosine distance operation.
     * @see #cosDist(String, String, String)
     */
    default CuBridge cosDist(String a, Tensor b, String out) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return cosDist(a, bName, out);
    }

    /**
     * **CosDist — Overload using Tensor a**
     *
     * Computes the cosine distance between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #cosDist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The name of the second input tensor.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the cosine distance operation.
     * @see #cosDist(String, String, String)
     */
    default CuBridge cosDist(Tensor a, String b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        return cosDist(aName, b, out);
    }

    /**
     * **CosDist — Overload using Tensor a and Tensor b**
     *
     * Computes the cosine distance between two {@link Tensor} objects.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #cosDist(String, String, String)}.
     * </p>
     *
     * @param a   The first input tensor object.
     * @param b   The second input tensor object.
     * @param out The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the cosine distance operation.
     * @see #cosDist(String, String, String)
     */
    default CuBridge cosDist(Tensor a, Tensor b, String out) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        return cosDist(aName, bName, out);
    }

    /**
     * **CosDistI — Immediate cosine distance operation (String a, String b)**
     *
     * Computes the cosine distance between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar cosine distance value.
     * @see #cosDist(String, String, String)
     */
    default Tensor cosDistI(String a, String b) {
        String oName = genRandomNameScalar();
        return cosDist(a, b, oName).get(oName);
    }

    /**
     * **CosDistI — Immediate cosine distance operation (String a, Tensor b)**
     *
     * Computes the cosine distance between a named tensor and a {@link Tensor} object
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the second tensor before executing
     * {@link #cosDist(String, String, String)}.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar cosine distance value.
     * @see #cosDist(String, String, String)
     */
    default Tensor cosDistI(String a, Tensor b) {
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return cosDist(a, bName, oName).get(oName);
    }

    /**
     * **CosDistI — Immediate cosine distance operation (Tensor a, String b)**
     *
     * Computes the cosine distance between a {@link Tensor} object and a named tensor
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the first tensor before executing
     * {@link #cosDist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} representing the scalar cosine distance value.
     * @see #cosDist(String, String, String)
     */
    default Tensor cosDistI(Tensor a, String b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameScalar();
        return cosDist(aName, b, oName).get(oName);
    }

    /**
     * **CosDistI — Immediate cosine distance operation (Tensor a, Tensor b)**
     *
     * Computes the cosine distance between two {@link Tensor} objects and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #cosDist(String, String, String)}.
     * </p>
     *
     * @param a The first input tensor object.
     * @param b The second input tensor object.
     * @return A {@link Tensor} representing the scalar cosine distance value.
     * @see #cosDist(String, String, String)
     */
    default Tensor cosDistI(Tensor a, Tensor b) {
        String aName = genRandomNameScalar(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameScalar(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameScalar();
        return cosDist(aName, bName, oName).get(oName);
    }


    /**
     * **MSE — Core Mean Squared Error operation**
     *
     * Computes the Mean Squared Error (MSE) between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * MSE = mean((y_i - label_i)^2)
     * </pre>
     * The resulting tensor has shape (1,) representing a scalar value.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label (target) tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge mse(String y, String label, String out) {
        if (CuBridgeJNI.mse(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | mse | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **MSE — Overload using Tensor label**
     *
     * Computes the MSE between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #mse(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MSE operation.
     * @see #mse(String, String, String)
     */
    default CuBridge mse(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return mse(y, labelName, out);
    }

    /**
     * **MSE — Overload using Tensor y**
     *
     * Computes the MSE between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #mse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MSE operation.
     * @see #mse(String, String, String)
     */
    default CuBridge mse(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return mse(yName, label, out);
    }

    /**
     * **MSE — Overload using Tensor y and Tensor label**
     *
     * Computes the MSE between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #mse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MSE operation.
     * @see #mse(String, String, String)
     */
    default CuBridge mse(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return mse(yName, labelName, out);
    }

    /**
     * **MSEI — Immediate Mean Squared Error operation (String y, String label)**
     *
     * Computes the MSE between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar MSE value.
     * @see #mse(String, String, String)
     */
    default Tensor mseI(String y, String label) {
        String oName = genRandomNameScalar();
        return mse(y, label, oName).get(oName);
    }

    /**
     * **MSEI — Immediate Mean Squared Error operation (String y, Tensor label)**
     *
     * Computes the MSE between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #mse(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar MSE value.
     * @see #mse(String, String, String)
     */
    default Tensor mseI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return mse(y, labelName, oName).get(oName);
    }

    /**
     * **MSEI — Immediate Mean Squared Error operation (Tensor y, String label)**
     *
     * Computes the MSE between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #mse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar MSE value.
     * @see #mse(String, String, String)
     */
    default Tensor mseI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return mse(yName, label, oName).get(oName);
    }

    /**
     * **MSEI — Immediate Mean Squared Error operation (Tensor y, Tensor label)**
     *
     * Computes the MSE between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #mse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar MSE value.
     * @see #mse(String, String, String)
     */
    default Tensor mseI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return mse(yName, labelName, oName).get(oName);
    }


    /**
     * **BCE — Core Binary Cross Entropy operation**
     *
     * Computes the Binary Cross Entropy (BCE) loss between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * BCE = -mean(label * log(y) + (1 - label) * log(1 - y))
     * </pre>
     * The resulting tensor has shape (1,) representing a scalar value.
     * </p>
     *
     * @param y      The name of the prediction tensor (values in range [0, 1]).
     * @param label  The name of the label (target) tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge bce(String y, String label, String out) {
        if (CuBridgeJNI.bce(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | bce | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **BCE — Overload using Tensor label**
     *
     * Computes the BCE loss between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #bce(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the BCE operation.
     * @see #bce(String, String, String)
     */
    default CuBridge bce(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return bce(y, labelName, out);
    }

    /**
     * **BCE — Overload using Tensor y**
     *
     * Computes the BCE loss between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #bce(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the BCE operation.
     * @see #bce(String, String, String)
     */
    default CuBridge bce(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return bce(yName, label, out);
    }

    /**
     * **BCE — Overload using Tensor y and Tensor label**
     *
     * Computes the BCE loss between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #bce(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the BCE operation.
     * @see #bce(String, String, String)
     */
    default CuBridge bce(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return bce(yName, labelName, out);
    }

    /**
     * **BCEI — Immediate Binary Cross Entropy operation (String y, String label)**
     *
     * Computes the BCE loss between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar BCE loss value.
     * @see #bce(String, String, String)
     */
    default Tensor bceI(String y, String label) {
        String oName = genRandomNameScalar();
        return bce(y, label, oName).get(oName);
    }

    /**
     * **BCEI — Immediate Binary Cross Entropy operation (String y, Tensor label)**
     *
     * Computes the BCE loss between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #bce(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar BCE loss value.
     * @see #bce(String, String, String)
     */
    default Tensor bceI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return bce(y, labelName, oName).get(oName);
    }

    /**
     * **BCEI — Immediate Binary Cross Entropy operation (Tensor y, String label)**
     *
     * Computes the BCE loss between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #bce(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar BCE loss value.
     * @see #bce(String, String, String)
     */
    default Tensor bceI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return bce(yName, label, oName).get(oName);
    }

    /**
     * **BCEI — Immediate Binary Cross Entropy operation (Tensor y, Tensor label)**
     *
     * Computes the BCE loss between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #bce(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar BCE loss value.
     * @see #bce(String, String, String)
     */
    default Tensor bceI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return bce(yName, labelName, oName).get(oName);
    }


    /**
     * **CEE — Core Categorical Cross Entropy operation**
     *
     * Computes the Categorical Cross Entropy (CEE) loss between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * CEE = -mean(sum(label_i * log(y_i)))
     * </pre>
     * The resulting tensor has shape (1,) representing a scalar value.
     * </p>
     *
     * @param y      The name of the prediction tensor (each row representing class probabilities).
     * @param label  The name of the one-hot encoded label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge cee(String y, String label, String out) {
        if (CuBridgeJNI.cee(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | cee | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **CEE — Overload using Tensor label**
     *
     * Computes the CEE loss between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #cee(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the CEE operation.
     * @see #cee(String, String, String)
     */
    default CuBridge cee(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return cee(y, labelName, out);
    }

    /**
     * **CEE — Overload using Tensor y**
     *
     * Computes the CEE loss between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #cee(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the CEE operation.
     * @see #cee(String, String, String)
     */
    default CuBridge cee(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return cee(yName, label, out);
    }

    /**
     * **CEE — Overload using Tensor y and Tensor label**
     *
     * Computes the CEE loss between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #cee(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the CEE operation.
     * @see #cee(String, String, String)
     */
    default CuBridge cee(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return cee(yName, labelName, out);
    }

    /**
     * **CEEI — Immediate Categorical Cross Entropy operation (String y, String label)**
     *
     * Computes the CEE loss between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar CEE loss value.
     * @see #cee(String, String, String)
     */
    default Tensor ceeI(String y, String label) {
        String oName = genRandomNameScalar();
        return cee(y, label, oName).get(oName);
    }

    /**
     * **CEEI — Immediate Categorical Cross Entropy operation (String y, Tensor label)**
     *
     * Computes the CEE loss between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #cee(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar CEE loss value.
     * @see #cee(String, String, String)
     */
    default Tensor ceeI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return cee(y, labelName, oName).get(oName);
    }

    /**
     * **CEEI — Immediate Categorical Cross Entropy operation (Tensor y, String label)**
     *
     * Computes the CEE loss between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #cee(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar CEE loss value.
     * @see #cee(String, String, String)
     */
    default Tensor ceeI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return cee(yName, label, oName).get(oName);
    }

    /**
     * **CEEI — Immediate Categorical Cross Entropy operation (Tensor y, Tensor label)**
     *
     * Computes the CEE loss between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #cee(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar CEE loss value.
     * @see #cee(String, String, String)
     */
    default Tensor ceeI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return cee(yName, labelName, oName).get(oName);
    }


    /**
     * **MAE — Core Mean Absolute Error operation**
     *
     * Computes the Mean Absolute Error (MAE) between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * MAE = mean(|y_i - label_i|)
     * </pre>
     * The resulting tensor has shape (1,) representing a scalar value.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge mae(String y, String label, String out) {
        if (CuBridgeJNI.mae(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | mae | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **MAE — Overload using Tensor label**
     *
     * Computes the MAE between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #mae(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MAE operation.
     * @see #mae(String, String, String)
     */
    default CuBridge mae(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return mae(y, labelName, out);
    }

    /**
     * **MAE — Overload using Tensor y**
     *
     * Computes the MAE between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #mae(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MAE operation.
     * @see #mae(String, String, String)
     */
    default CuBridge mae(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return mae(yName, label, out);
    }

    /**
     * **MAE — Overload using Tensor y and Tensor label**
     *
     * Computes the MAE between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #mae(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MAE operation.
     * @see #mae(String, String, String)
     */
    default CuBridge mae(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return mae(yName, labelName, out);
    }

    /**
     * **MAEI — Immediate Mean Absolute Error operation (String y, String label)**
     *
     * Computes the MAE between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar MAE value.
     * @see #mae(String, String, String)
     */
    default Tensor maeI(String y, String label) {
        String oName = genRandomNameScalar();
        return mae(y, label, oName).get(oName);
    }

    /**
     * **MAEI — Immediate Mean Absolute Error operation (String y, Tensor label)**
     *
     * Computes the MAE between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #mae(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar MAE value.
     * @see #mae(String, String, String)
     */
    default Tensor maeI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return mae(y, labelName, oName).get(oName);
    }

    /**
     * **MAEI — Immediate Mean Absolute Error operation (Tensor y, String label)**
     *
     * Computes the MAE between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #mae(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar MAE value.
     * @see #mae(String, String, String)
     */
    default Tensor maeI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return mae(yName, label, oName).get(oName);
    }

    /**
     * **MAEI — Immediate Mean Absolute Error operation (Tensor y, Tensor label)**
     *
     * Computes the MAE between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #mae(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar MAE value.
     * @see #mae(String, String, String)
     */
    default Tensor maeI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return mae(yName, labelName, oName).get(oName);
    }


    /**
     * **RMSE — Core Root Mean Squared Error operation**
     *
     * Computes the Root Mean Squared Error (RMSE) between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * RMSE = sqrt(mean((y_i - label_i)^2))
     * </pre>
     * The resulting tensor has shape (1,) representing a scalar value.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge rmse(String y, String label, String out) {
        if (CuBridgeJNI.rmse(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | rmse | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **RMSE — Overload using Tensor label**
     *
     * Computes the Root Mean Squared Error (RMSE) between a named prediction tensor
     * and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #rmse(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the RMSE operation.
     * @see #rmse(String, String, String)
     */
    default CuBridge rmse(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return rmse(y, labelName, out);
    }

    /**
     * **RMSE — Overload using Tensor y**
     *
     * Computes the Root Mean Squared Error (RMSE) between a {@link Tensor} prediction
     * and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #rmse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the RMSE operation.
     * @see #rmse(String, String, String)
     */
    default CuBridge rmse(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return rmse(yName, label, out);
    }

    /**
     * **RMSE — Overload using Tensor y and Tensor label**
     *
     * Computes the Root Mean Squared Error (RMSE) between two {@link Tensor} objects
     * (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #rmse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the RMSE operation.
     * @see #rmse(String, String, String)
     */
    default CuBridge rmse(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return rmse(yName, labelName, out);
    }

    /**
     * **RMSEI — Immediate Root Mean Squared Error operation (String y, String label)**
     *
     * Computes the RMSE between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar RMSE value.
     * @see #rmse(String, String, String)
     */
    default Tensor rmseI(String y, String label) {
        String oName = genRandomNameScalar();
        return rmse(y, label, oName).get(oName);
    }

    /**
     * **RMSEI — Immediate Root Mean Squared Error operation (String y, Tensor label)**
     *
     * Computes the RMSE between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #rmse(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar RMSE value.
     * @see #rmse(String, String, String)
     */
    default Tensor rmseI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return rmse(y, labelName, oName).get(oName);
    }

    /**
     * **RMSEI — Immediate Root Mean Squared Error operation (Tensor y, String label)**
     *
     * Computes the RMSE between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #rmse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar RMSE value.
     * @see #rmse(String, String, String)
     */
    default Tensor rmseI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return rmse(yName, label, oName).get(oName);
    }

    /**
     * **RMSEI — Immediate Root Mean Squared Error operation (Tensor y, Tensor label)**
     *
     * Computes the RMSE between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #rmse(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar RMSE value.
     * @see #rmse(String, String, String)
     */
    default Tensor rmseI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return rmse(yName, labelName, oName).get(oName);
    }


    /**
     * **MAPE — Core Mean Absolute Percentage Error operation**
     *
     * Computes the Mean Absolute Percentage Error (MAPE) between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Formula:
     * <pre>
     * MAPE = 100 * mean(|(y_i - label_i) / (label_i + ε)|)
     * </pre>
     * The result represents the mean relative error in percentage form.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge mape(String y, String label, String out) {
        if (CuBridgeJNI.mape(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | mape | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **MAPE — Overload using Tensor label**
     *
     * Computes the MAPE between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #mape(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MAPE operation.
     * @see #mape(String, String, String)
     */
    default CuBridge mape(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return mape(y, labelName, out);
    }

    /**
     * **MAPE — Overload using Tensor y**
     *
     * Computes the MAPE between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #mape(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MAPE operation.
     * @see #mape(String, String, String)
     */
    default CuBridge mape(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return mape(yName, label, out);
    }

    /**
     * **MAPE — Overload using Tensor y and Tensor label**
     *
     * Computes the MAPE between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #mape(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the MAPE operation.
     * @see #mape(String, String, String)
     */
    default CuBridge mape(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return mape(yName, labelName, out);
    }

    /**
     * **MAPEI — Immediate Mean Absolute Percentage Error operation (String y, String label)**
     *
     * Computes the MAPE between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar MAPE value.
     * @see #mape(String, String, String)
     */
    default Tensor mapeI(String y, String label) {
        String oName = genRandomNameScalar();
        return mape(y, label, oName).get(oName);
    }

    /**
     * **MAPEI — Immediate Mean Absolute Percentage Error operation (String y, Tensor label)**
     *
     * Computes the MAPE between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #mape(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar MAPE value.
     * @see #mape(String, String, String)
     */
    default Tensor mapeI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return mape(y, labelName, oName).get(oName);
    }

    /**
     * **MAPEI — Immediate Mean Absolute Percentage Error operation (Tensor y, String label)**
     *
     * Computes the MAPE between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #mape(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar MAPE value.
     * @see #mape(String, String, String)
     */
    default Tensor mapeI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return mape(yName, label, oName).get(oName);
    }

    /**
     * **MAPEI — Immediate Mean Absolute Percentage Error operation (Tensor y, Tensor label)**
     *
     * Computes the MAPE between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #mape(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar MAPE value.
     * @see #mape(String, String, String)
     */
    default Tensor mapeI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return mape(yName, labelName, oName).get(oName);
    }


    /**
     * **Focal — Core Focal Loss operation**
     *
     * Computes the Focal Loss between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Focal Loss is a variant of cross-entropy designed to handle class imbalance by
     * reducing the relative loss for well-classified examples and focusing training on hard samples.
     * </p>
     * <p>
     * Formula:
     * <pre>
     * FL = -α * (1 - p_t)^γ * log(p_t)
     * where:
     *   p_t = y_i       if label_i = 1
     *         1 - y_i   if label_i = 0
     * </pre>
     * Common defaults: α = 0.25, γ = 2.0.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The name of the prediction tensor (values in range [0, 1]).
     * @param label  The name of the label tensor (binary 0 or 1 values).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge focal(String y, String label, String out) {
        if (CuBridgeJNI.focal(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | focal | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **Focal — Overload using Tensor label**
     *
     * Computes the Focal Loss between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #focal(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor (values in range [0, 1]).
     * @param label  The label tensor object (binary 0 or 1 values).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Focal Loss operation.
     * @see #focal(String, String, String)
     */
    default CuBridge focal(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return focal(y, labelName, out);
    }

    /**
     * **Focal — Overload using Tensor y**
     *
     * Computes the Focal Loss between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #focal(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object (values in range [0, 1]).
     * @param label  The name of the label tensor (binary 0 or 1 values).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Focal Loss operation.
     * @see #focal(String, String, String)
     */
    default CuBridge focal(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return focal(yName, label, out);
    }

    /**
     * **Focal — Overload using Tensor y and Tensor label**
     *
     * Computes the Focal Loss between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #focal(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object (values in range [0, 1]).
     * @param label  The label tensor object (binary 0 or 1 values).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Focal Loss operation.
     * @see #focal(String, String, String)
     */
    default CuBridge focal(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return focal(yName, labelName, out);
    }

    /**
     * **FocalI — Immediate Focal Loss operation (String y, String label)**
     *
     * Computes the Focal Loss between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar Focal Loss value.
     * @see #focal(String, String, String)
     */
    default Tensor focalI(String y, String label) {
        String oName = genRandomNameScalar();
        return focal(y, label, oName).get(oName);
    }

    /**
     * **FocalI — Immediate Focal Loss operation (String y, Tensor label)**
     *
     * Computes the Focal Loss between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #focal(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar Focal Loss value.
     * @see #focal(String, String, String)
     */
    default Tensor focalI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return focal(y, labelName, oName).get(oName);
    }

    /**
     * **FocalI — Immediate Focal Loss operation (Tensor y, String label)**
     *
     * Computes the Focal Loss between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #focal(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar Focal Loss value.
     * @see #focal(String, String, String)
     */
    default Tensor focalI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return focal(yName, label, oName).get(oName);
    }

    /**
     * **FocalI — Immediate Focal Loss operation (Tensor y, Tensor label)**
     *
     * Computes the Focal Loss between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #focal(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar Focal Loss value.
     * @see #focal(String, String, String)
     */
    default Tensor focalI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return focal(yName, labelName, oName).get(oName);
    }


    /**
     * **Perplexity — Core Perplexity computation**
     *
     * Computes the Perplexity score between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * Perplexity is the exponential of the categorical cross entropy,
     * commonly used to evaluate probabilistic language models.
     * Lower values indicate better predictive performance.
     * </p>
     * <p>
     * Formula:
     * <pre>
     * Perplexity = exp(mean(-Σ label_i * log(y_i)))
     * </pre>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The name of the prediction tensor (class probabilities).
     * @param label  The name of the label tensor (one-hot encoded).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge perplexity(String y, String label, String out) {
        if (CuBridgeJNI.perplexity(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | perplexity | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **Perplexity — Overload using Tensor label**
     *
     * Computes the Perplexity between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #perplexity(String, String, String)}.
     * </p>
     *
     * @param y      The name of the prediction tensor (class probabilities).
     * @param label  The label tensor object (one-hot encoded).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Perplexity operation.
     * @see #perplexity(String, String, String)
     */
    default CuBridge perplexity(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return perplexity(y, labelName, out);
    }

    /**
     * **Perplexity — Overload using Tensor y**
     *
     * Computes the Perplexity between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #perplexity(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object (class probabilities).
     * @param label  The name of the label tensor (one-hot encoded).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Perplexity operation.
     * @see #perplexity(String, String, String)
     */
    default CuBridge perplexity(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return perplexity(yName, label, out);
    }

    /**
     * **Perplexity — Overload using Tensor y and Tensor label**
     *
     * Computes the Perplexity between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #perplexity(String, String, String)}.
     * </p>
     *
     * @param y      The prediction tensor object (class probabilities).
     * @param label  The label tensor object (one-hot encoded).
     * @param out    The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Perplexity operation.
     * @see #perplexity(String, String, String)
     */
    default CuBridge perplexity(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return perplexity(yName, labelName, out);
    }

    /**
     * **PerplexityI — Immediate Perplexity operation (String y, String label)**
     *
     * Computes the Perplexity between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar Perplexity value.
     * @see #perplexity(String, String, String)
     */
    default Tensor perplexityI(String y, String label) {
        String oName = genRandomNameScalar();
        return perplexity(y, label, oName).get(oName);
    }

    /**
     * **PerplexityI — Immediate Perplexity operation (String y, Tensor label)**
     *
     * Computes the Perplexity between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #perplexity(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The name of the prediction tensor.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar Perplexity value.
     * @see #perplexity(String, String, String)
     */
    default Tensor perplexityI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return perplexity(y, labelName, oName).get(oName);
    }

    /**
     * **PerplexityI — Immediate Perplexity operation (Tensor y, String label)**
     *
     * Computes the Perplexity between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #perplexity(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The name of the label tensor.
     * @return A {@link Tensor} representing the scalar Perplexity value.
     * @see #perplexity(String, String, String)
     */
    default Tensor perplexityI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return perplexity(yName, label, oName).get(oName);
    }

    /**
     * **PerplexityI — Immediate Perplexity operation (Tensor y, Tensor label)**
     *
     * Computes the Perplexity between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #perplexity(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y      The prediction tensor object.
     * @param label  The label tensor object.
     * @return A {@link Tensor} representing the scalar Perplexity value.
     * @see #perplexity(String, String, String)
     */
    default Tensor perplexityI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return perplexity(yName, labelName, oName).get(oName);
    }


    /**
     * **Dice — Core Dice Coefficient operation**
     *
     * Calculates the Dice Coefficient between prediction and label tensors
     * and stores the result as a scalar tensor.
     * <p>
     * The Dice coefficient measures the overlap between two binary or probabilistic masks,
     * and is widely used as a segmentation similarity metric.
     * </p>
     * <p>
     * Formula:
     * <pre>
     * Dice = (2 * Σ(y_i * label_i)) / (Σy_i + Σlabel_i + ε)
     * </pre>
     * where ε is a small constant to prevent division by zero.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The name of the prediction tensor (values in [0, 1]).
     * @param label The name of the ground truth label tensor (values in [0, 1]).
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #dice(String, String, String)
     */
    default CuBridge dice(String y, String label, String out) {
        if (CuBridgeJNI.dice(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | dice | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **Dice — Overload using Tensor label**
     *
     * Calculates the Dice coefficient between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #dice(String, String, String)}.
     * </p>
     *
     * @param y     The name of the prediction tensor.
     * @param label The label tensor object.
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Dice operation.
     * @see #dice(String, String, String)
     */
    default CuBridge dice(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return dice(y, labelName, out);
    }

    /**
     * **Dice — Overload using Tensor y**
     *
     * Calculates the Dice coefficient between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #dice(String, String, String)}.
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The name of the label tensor.
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Dice operation.
     * @see #dice(String, String, String)
     */
    default CuBridge dice(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return dice(yName, label, out);
    }

    /**
     * **Dice — Overload using Tensor y and Tensor label**
     *
     * Calculates the Dice coefficient between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #dice(String, String, String)}.
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The label tensor object.
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the Dice operation.
     * @see #dice(String, String, String)
     */
    default CuBridge dice(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return dice(yName, labelName, out);
    }

    /**
     * **DiceI — Immediate Dice Coefficient operation (String y, String label)**
     *
     * Calculates the Dice coefficient between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The name of the prediction tensor.
     * @param label The name of the label tensor.
     * @return A {@link Tensor} representing the scalar Dice coefficient.
     * @see #dice(String, String, String)
     */
    default Tensor diceI(String y, String label) {
        String oName = genRandomNameScalar();
        return dice(y, label, oName).get(oName);
    }

    /**
     * **DiceI — Immediate Dice Coefficient operation (String y, Tensor label)**
     *
     * Calculates the Dice coefficient between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #dice(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The name of the prediction tensor.
     * @param label The label tensor object.
     * @return A {@link Tensor} representing the scalar Dice coefficient.
     * @see #dice(String, String, String)
     */
    default Tensor diceI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return dice(y, labelName, oName).get(oName);
    }

    /**
     * **DiceI — Immediate Dice Coefficient operation (Tensor y, String label)**
     *
     * Calculates the Dice coefficient between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #dice(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The name of the label tensor.
     * @return A {@link Tensor} representing the scalar Dice coefficient.
     * @see #dice(String, String, String)
     */
    default Tensor diceI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return dice(yName, label, oName).get(oName);
    }

    /**
     * **DiceI — Immediate Dice Coefficient operation (Tensor y, Tensor label)**
     *
     * Calculates the Dice coefficient between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #dice(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The label tensor object.
     * @return A {@link Tensor} representing the scalar Dice coefficient.
     * @see #dice(String, String, String)
     */
    default Tensor diceI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return dice(yName, labelName, oName).get(oName);
    }


    /**
     * **IoU — Core Intersection over Union operation**
     *
     * Calculates the Intersection over Union (IoU, or Jaccard Index)
     * between prediction and label tensors, and stores the result as a scalar tensor.
     * <p>
     * IoU measures the ratio of overlap between two sets to their union,
     * commonly used as a segmentation accuracy metric.
     * </p>
     * <p>
     * Formula:
     * <pre>
     * IoU = Σ(y_i * label_i) / (Σy_i + Σlabel_i - Σ(y_i * label_i) + ε)
     * </pre>
     * where ε is a small constant to avoid division by zero.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The name of the prediction tensor (values in [0, 1]).
     * @param label The name of the label tensor (values in [0, 1]).
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #iou(String, String, String)
     */
    default CuBridge iou(String y, String label, String out) {
        if (CuBridgeJNI.iou(y, label, out)) return CuBridge.getInstance();
        else System.err.println("Error | iou | " + y + " | " + label + " | " + out);
        return null;
    }

    /**
     * **IoU — Overload using Tensor label**
     *
     * Calculates the Intersection over Union between a named prediction tensor and a {@link Tensor} label object.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #iou(String, String, String)}.
     * </p>
     *
     * @param y     The name of the prediction tensor.
     * @param label The label tensor object.
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the IoU operation.
     * @see #iou(String, String, String)
     */
    default CuBridge iou(String y, Tensor label, String out) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return iou(y, labelName, out);
    }

    /**
     * **IoU — Overload using Tensor y**
     *
     * Calculates the Intersection over Union between a {@link Tensor} prediction and a named label tensor.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #iou(String, String, String)}.
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The name of the label tensor.
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the IoU operation.
     * @see #iou(String, String, String)
     */
    default CuBridge iou(Tensor y, String label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        return iou(yName, label, out);
    }

    /**
     * **IoU — Overload using Tensor y and Tensor label**
     *
     * Calculates the Intersection over Union between two {@link Tensor} objects (prediction and label).
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #iou(String, String, String)}.
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The label tensor object.
     * @param out   The name to store the resulting scalar tensor.
     * @return A {@link CuBridge} instance representing the IoU operation.
     * @see #iou(String, String, String)
     */
    default CuBridge iou(Tensor y, Tensor label, String out) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        return iou(yName, labelName, out);
    }

    /**
     * **IoUI — Immediate Intersection over Union operation (String y, String label)**
     *
     * Calculates the IoU between two named tensors and directly returns
     * the resulting scalar {@link Tensor}.
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The name of the prediction tensor.
     * @param label The name of the label tensor.
     * @return A {@link Tensor} representing the scalar IoU value.
     * @see #iou(String, String, String)
     */
    default Tensor iouI(String y, String label) {
        String oName = genRandomNameScalar();
        return iou(y, label, oName).get(oName);
    }

    /**
     * **IoUI — Immediate Intersection over Union operation (String y, Tensor label)**
     *
     * Calculates the IoU between a named prediction tensor and a {@link Tensor} label object,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the label tensor before executing
     * {@link #iou(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The name of the prediction tensor.
     * @param label The label tensor object.
     * @return A {@link Tensor} representing the scalar IoU value.
     * @see #iou(String, String, String)
     */
    default Tensor iouI(String y, Tensor label) {
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return iou(y, labelName, oName).get(oName);
    }

    /**
     * **IoUI — Immediate Intersection over Union operation (Tensor y, String label)**
     *
     * Calculates the IoU between a {@link Tensor} prediction and a named label tensor,
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the prediction tensor before executing
     * {@link #iou(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The name of the label tensor.
     * @return A {@link Tensor} representing the scalar IoU value.
     * @see #iou(String, String, String)
     */
    default Tensor iouI(Tensor y, String label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String oName = genRandomNameScalar();
        return iou(yName, label, oName).get(oName);
    }

    /**
     * **IoUI — Immediate Intersection over Union operation (Tensor y, Tensor label)**
     *
     * Calculates the IoU between two {@link Tensor} objects (prediction and label)
     * and directly returns the resulting scalar {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to both tensors before executing
     * {@link #iou(String, String, String)}.
     * </p>
     * <p>
     * Output: Scalar tensor (shape = [1])
     * </p>
     *
     * @param y     The prediction tensor object.
     * @param label The label tensor object.
     * @return A {@link Tensor} representing the scalar IoU value.
     * @see #iou(String, String, String)
     */
    default Tensor iouI(Tensor y, Tensor label) {
        String yName = genRandomNameScalar(); CuBridge.getInstance().put(y, yName);
        String labelName = genRandomNameScalar(); CuBridge.getInstance().put(label, labelName);
        String oName = genRandomNameScalar();
        return iou(yName, labelName, oName).get(oName);
    }

}
