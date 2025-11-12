package CuBridge;

import java.util.UUID;

public interface UnaryOps {

    private String genRandomNameUnary() {
        return "UnaryOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }


    /**
     * **Abs — Basic absolute value operation with empty tensor reference**
     *
     * Computes the element-wise absolute value of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the absolute value operation.
     * @see #abs(String, String)
     */
    default CuBridge abs() {
        return abs("", genRandomNameUnary());
    }

    /**
     * **Abs — Absolute value operation on a named tensor**
     *
     * Computes the element-wise absolute value of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the absolute value operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge abs(String a, String out) {
        if (CuBridgeJNI.abs(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | abs | " + a + " | " + out);
        return null;
    }

    /**
     * **Abs — Absolute value operation using a Tensor object**
     *
     * Computes the element-wise absolute value of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #abs(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the absolute value operation.
     * @see #abs(String, String)
     */
    default CuBridge abs(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return abs(aName, out);
    }

    /**
     * **AbsI — Immediate absolute value operation**
     *
     * Computes the element-wise absolute value of a tensor already stored in the queue,
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * When no tensor name is provided, this function operates on the topmost tensor in the queue.
     * </p>
     *
     * @return A new {@link Tensor} representing the absolute value of the input tensor.
     * @see #abs(String, String)
     */
    default Tensor absI() {
        String oName = genRandomNameUnary();
        return abs("", oName).get(oName);
    }

    /**
     * **AbsI — Immediate absolute value of a named tensor**
     *
     * Computes the element-wise absolute value of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the absolute value of the input tensor.
     * @see #abs(String, String)
     */
    default Tensor absI(String a) {
        String oName = genRandomNameUnary();
        return abs(a, oName).get(oName);
    }

    /**
     * **AbsI — Immediate absolute value of a Tensor object**
     *
     * Computes the element-wise absolute value of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #abs(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the absolute value of the input tensor.
     * @see #abs(String, String)
     */
    default Tensor absI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return abs(aName, oName).get(oName);
    }


    /**
     * **Neg — Basic negation with empty tensor reference**
     *
     * Computes the element-wise negation of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the negation operation.
     * @see #neg(String, String)
     */
    default CuBridge neg() {
        return neg("", genRandomNameUnary());
    }

    /**
     * **Neg — Negation of a named tensor**
     *
     * Computes the element-wise negation of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the negation operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge neg(String a, String out) {
        if (CuBridgeJNI.neg(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | neg | " + a + " | " + out);
        return null;
    }

    /**
     * **Neg — Negation of a Tensor object**
     *
     * Computes the element-wise negation of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #neg(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the negation operation.
     * @see #neg(String, String)
     */
    default CuBridge neg(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return neg(aName, out);
    }

    /**
     * **NegI — Immediate negation with default tensor reference**
     *
     * Computes the element-wise negation of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * When no tensor name is provided, this function operates on the topmost tensor in the queue.
     * </p>
     *
     * @return A new {@link Tensor} representing the negated tensor.
     * @see #neg(String, String)
     */
    default Tensor negI() {
        String oName = genRandomNameUnary();
        return neg("", oName).get(oName);
    }

    /**
     * **NegI — Immediate negation of a named tensor**
     *
     * Computes the element-wise negation of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the negated tensor.
     * @see #neg(String, String)
     */
    default Tensor negI(String a) {
        String oName = genRandomNameUnary();
        return neg(a, oName).get(oName);
    }

    /**
     * **NegI — Immediate negation of a Tensor object**
     *
     * Computes the element-wise negation of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #neg(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the negated tensor.
     * @see #neg(String, String)
     */
    default Tensor negI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return neg(aName, oName).get(oName);
    }


    /**
     * **Square — Basic squaring with empty tensor reference**
     *
     * Computes the element-wise square of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the square operation.
     * @see #square(String, String)
     */
    default CuBridge square() {
        return square("", genRandomNameUnary());
    }

    /**
     * **Square — Squaring a named tensor**
     *
     * Computes the element-wise square of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the square operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge square(String a, String out) {
        if (CuBridgeJNI.square(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | square | " + a + " | " + out);
        return null;
    }

    /**
     * **Square — Squaring a Tensor object**
     *
     * Computes the element-wise square of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #square(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the square operation.
     * @see #square(String, String)
     */
    default CuBridge square(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return square(aName, out);
    }

    /**
     * **SquareI — Immediate squaring with default tensor reference**
     *
     * Computes the element-wise square of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * When no tensor name is provided, this function operates on the topmost tensor in the queue.
     * </p>
     *
     * @return A new {@link Tensor} representing the squared tensor.
     * @see #square(String, String)
     */
    default Tensor squareI() {
        String oName = genRandomNameUnary();
        return square("", oName).get(oName);
    }

    /**
     * **SquareI — Immediate squaring of a named tensor**
     *
     * Computes the element-wise square of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the squared tensor.
     * @see #square(String, String)
     */
    default Tensor squareI(String a) {
        String oName = genRandomNameUnary();
        return square(a, oName).get(oName);
    }

    /**
     * **SquareI — Immediate squaring of a Tensor object**
     *
     * Computes the element-wise square of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #square(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the squared tensor.
     * @see #square(String, String)
     */
    default Tensor squareI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return square(aName, oName).get(oName);
    }


    /**
     * **Sqrt — Basic square root with empty tensor reference**
     *
     * Computes the element-wise square root of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the square root operation.
     * @see #sqrt(String, String)
     */
    default CuBridge sqrt() {
        return sqrt("", genRandomNameUnary());
    }

    /**
     * **Sqrt — Square root of a named tensor**
     *
     * Computes the element-wise square root of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the square root operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge sqrt(String a, String out) {
        if (CuBridgeJNI.sqrt(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | sqrt | " + a + " | " + out);
        return null;
    }

    /**
     * **Sqrt — Square root of a Tensor object**
     *
     * Computes the element-wise square root of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #sqrt(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the square root operation.
     * @see #sqrt(String, String)
     */
    default CuBridge sqrt(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return sqrt(aName, out);
    }

    /**
     * **SqrtI — Immediate square root with default tensor reference**
     *
     * Computes the element-wise square root of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * When no tensor name is provided, this function operates on the topmost tensor in the queue.
     * </p>
     *
     * @return A new {@link Tensor} representing the square-rooted tensor.
     * @see #sqrt(String, String)
     */
    default Tensor sqrtI() {
        String oName = genRandomNameUnary();
        return sqrt("", oName).get(oName);
    }

    /**
     * **SqrtI — Immediate square root of a named tensor**
     *
     * Computes the element-wise square root of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the square-rooted tensor.
     * @see #sqrt(String, String)
     */
    default Tensor sqrtI(String a) {
        String oName = genRandomNameUnary();
        return sqrt(a, oName).get(oName);
    }

    /**
     * **SqrtI — Immediate square root of a Tensor object**
     *
     * Computes the element-wise square root of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #sqrt(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the square-rooted tensor.
     * @see #sqrt(String, String)
     */
    default Tensor sqrtI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return sqrt(aName, oName).get(oName);
    }


    /**
     * **Log — Base-10 logarithm with empty tensor reference**
     *
     * Computes the element-wise base-10 logarithm of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the base-10 logarithm operation.
     * @see #log(String, String)
     */
    default CuBridge log() {
        return log("", genRandomNameUnary());
    }

    /**
     * **Log — Base-10 logarithm of a named tensor**
     *
     * Computes the element-wise base-10 logarithm of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the base-10 logarithm operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge log(String a, String out) {
        if (CuBridgeJNI.log(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | log | " + a + " | " + out);
        return null;
    }

    /**
     * **Log — Base-10 logarithm of a Tensor object**
     *
     * Computes the element-wise base-10 logarithm of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #log(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the base-10 logarithm operation.
     * @see #log(String, String)
     */
    default CuBridge log(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return log(aName, out);
    }

    /**
     * **LogI — Immediate base-10 logarithm with default tensor reference**
     *
     * Computes the element-wise base-10 logarithm of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the log-transformed tensor.
     * @see #log(String, String)
     */
    default Tensor logI() {
        String oName = genRandomNameUnary();
        return log("", oName).get(oName);
    }

    /**
     * **LogI — Immediate base-10 logarithm of a named tensor**
     *
     * Computes the element-wise base-10 logarithm of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the log-transformed tensor.
     * @see #log(String, String)
     */
    default Tensor logI(String a) {
        String oName = genRandomNameUnary();
        return log(a, oName).get(oName);
    }

    /**
     * **LogI — Immediate base-10 logarithm of a Tensor object**
     *
     * Computes the element-wise base-10 logarithm of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #log(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the log-transformed tensor.
     * @see #log(String, String)
     */
    default Tensor logI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return log(aName, oName).get(oName);
    }


    /**
     * **Log2 — Base-2 logarithm with empty tensor reference**
     *
     * Computes the element-wise base-2 logarithm of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the base-2 logarithm operation.
     * @see #log2(String, String)
     */
    default CuBridge log2() {
        return log2("", genRandomNameUnary());
    }

    /**
     * **Log2 — Base-2 logarithm of a named tensor**
     *
     * Computes the element-wise base-2 logarithm of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the base-2 logarithm operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge log2(String a, String out) {
        if (CuBridgeJNI.log2(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | log2 | " + a + " | " + out);
        return null;
    }

    /**
     * **Log2 — Base-2 logarithm of a Tensor object**
     *
     * Computes the element-wise base-2 logarithm of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #log2(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the base-2 logarithm operation.
     * @see #log2(String, String)
     */
    default CuBridge log2(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return log2(aName, out);
    }

    /**
     * **Log2I — Immediate base-2 logarithm with default tensor reference**
     *
     * Computes the element-wise base-2 logarithm of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the log2-transformed tensor.
     * @see #log2(String, String)
     */
    default Tensor log2I() {
        String oName = genRandomNameUnary();
        return log2("", oName).get(oName);
    }

    /**
     * **Log2I — Immediate base-2 logarithm of a named tensor**
     *
     * Computes the element-wise base-2 logarithm of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the log2-transformed tensor.
     * @see #log2(String, String)
     */
    default Tensor log2I(String a) {
        String oName = genRandomNameUnary();
        return log2(a, oName).get(oName);
    }

    /**
     * **Log2I — Immediate base-2 logarithm of a Tensor object**
     *
     * Computes the element-wise base-2 logarithm of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #log2(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the log2-transformed tensor.
     * @see #log2(String, String)
     */
    default Tensor log2I(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return log2(aName, oName).get(oName);
    }


    /**
     * **Ln — Natural logarithm with empty tensor reference**
     *
     * Computes the element-wise natural logarithm (base *e*) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the natural logarithm operation.
     * @see #ln(String, String)
     */
    default CuBridge ln() {
        return ln("", genRandomNameUnary());
    }

    /**
     * **Ln — Natural logarithm of a named tensor**
     *
     * Computes the element-wise natural logarithm (base *e*) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the natural logarithm operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge ln(String a, String out) {
        if (CuBridgeJNI.ln(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | ln | " + a + " | " + out);
        return null;
    }

    /**
     * **Ln — Natural logarithm of a Tensor object**
     *
     * Computes the element-wise natural logarithm (base *e*) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #ln(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the natural logarithm operation.
     * @see #ln(String, String)
     */
    default CuBridge ln(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return ln(aName, out);
    }

    /**
     * **LnI — Immediate natural logarithm with default tensor reference**
     *
     * Computes the element-wise natural logarithm (base *e*) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the ln-transformed tensor.
     * @see #ln(String, String)
     */
    default Tensor lnI() {
        String oName = genRandomNameUnary();
        return ln("", oName).get(oName);
    }

    /**
     * **LnI — Immediate natural logarithm of a named tensor**
     *
     * Computes the element-wise natural logarithm (base *e*) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the ln-transformed tensor.
     * @see #ln(String, String)
     */
    default Tensor lnI(String a) {
        String oName = genRandomNameUnary();
        return ln(a, oName).get(oName);
    }

    /**
     * **LnI — Immediate natural logarithm of a Tensor object**
     *
     * Computes the element-wise natural logarithm (base *e*) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #ln(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the ln-transformed tensor.
     * @see #ln(String, String)
     */
    default Tensor lnI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return ln(aName, oName).get(oName);
    }

    /**
     * **Reciprocal — Basic reciprocal with empty tensor reference**
     *
     * Computes the element-wise reciprocal (1/x) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the reciprocal operation.
     * @see #reciprocal(String, String)
     */
    default CuBridge reciprocal() {
        return reciprocal("", genRandomNameUnary());
    }

    /**
     * **Reciprocal — Reciprocal of a named tensor**
     *
     * Computes the element-wise reciprocal (1/x) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the reciprocal operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge reciprocal(String a, String out) {
        if (CuBridgeJNI.reciprocal(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | reciprocal | " + a + " | " + out);
        return null;
    }

    /**
     * **Reciprocal — Reciprocal of a Tensor object**
     *
     * Computes the element-wise reciprocal (1/x) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #reciprocal(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the reciprocal operation.
     * @see #reciprocal(String, String)
     */
    default CuBridge reciprocal(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return reciprocal(aName, out);
    }

    /**
     * **ReciprocalI — Immediate reciprocal with default tensor reference**
     *
     * Computes the element-wise reciprocal (1/x) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the reciprocal values of the input tensor.
     * @see #reciprocal(String, String)
     */
    default Tensor reciprocalI() {
        String oName = genRandomNameUnary();
        return reciprocal("", oName).get(oName);
    }

    /**
     * **ReciprocalI — Immediate reciprocal of a named tensor**
     *
     * Computes the element-wise reciprocal (1/x) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the reciprocal values of the input tensor.
     * @see #reciprocal(String, String)
     */
    default Tensor reciprocalI(String a) {
        String oName = genRandomNameUnary();
        return reciprocal(a, oName).get(oName);
    }

    /**
     * **ReciprocalI — Immediate reciprocal of a Tensor object**
     *
     * Computes the element-wise reciprocal (1/x) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #reciprocal(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the reciprocal values of the input tensor.
     * @see #reciprocal(String, String)
     */
    default Tensor reciprocalI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return reciprocal(aName, oName).get(oName);
    }


    /**
     * **Rsqrt — Basic reciprocal square root with empty tensor reference**
     *
     * Computes the element-wise reciprocal square root (1/√x) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the reciprocal square root operation.
     * @see #rsqrt(String, String)
     */
    default CuBridge rsqrt() {
        return rsqrt("", genRandomNameUnary());
    }

    /**
     * **Rsqrt — Reciprocal square root of a named tensor**
     *
     * Computes the element-wise reciprocal square root (1/√x) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the reciprocal square root operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge rsqrt(String a, String out) {
        if (CuBridgeJNI.rsqrt(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | rsqrt | " + a + " | " + out);
        return null;
    }

    /**
     * **Rsqrt — Reciprocal square root of a Tensor object**
     *
     * Computes the element-wise reciprocal square root (1/√x) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #rsqrt(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the reciprocal square root operation.
     * @see #rsqrt(String, String)
     */
    default CuBridge rsqrt(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return rsqrt(aName, out);
    }

    /**
     * **RsqrtI — Immediate reciprocal square root with default tensor reference**
     *
     * Computes the element-wise reciprocal square root (1/√x) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the reciprocal square root of the input tensor.
     * @see #rsqrt(String, String)
     */
    default Tensor rsqrtI() {
        String oName = genRandomNameUnary();
        return rsqrt("", oName).get(oName);
    }

    /**
     * **RsqrtI — Immediate reciprocal square root of a named tensor**
     *
     * Computes the element-wise reciprocal square root (1/√x) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the reciprocal square root of the input tensor.
     * @see #rsqrt(String, String)
     */
    default Tensor rsqrtI(String a) {
        String oName = genRandomNameUnary();
        return rsqrt(a, oName).get(oName);
    }

    /**
     * **RsqrtI — Immediate reciprocal square root of a Tensor object**
     *
     * Computes the element-wise reciprocal square root (1/√x) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #rsqrt(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the reciprocal square root of the input tensor.
     * @see #rsqrt(String, String)
     */
    default Tensor rsqrtI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return rsqrt(aName, oName).get(oName);
    }


    /**
     * **Exp — Basic exponential with empty tensor reference**
     *
     * Computes the element-wise exponential (eˣ) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the exponential operation.
     * @see #exp(String, String)
     */
    default CuBridge exp() {
        return exp("", genRandomNameUnary());
    }

    /**
     * **Exp — Exponential of a named tensor**
     *
     * Computes the element-wise exponential (eˣ) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the exponential operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge exp(String a, String out) {
        if (CuBridgeJNI.exp(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | exp | " + a + " | " + out);
        return null;
    }

    /**
     * **Exp — Exponential of a Tensor object**
     *
     * Computes the element-wise exponential (eˣ) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #exp(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the exponential operation.
     * @see #exp(String, String)
     */
    default CuBridge exp(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return exp(aName, out);
    }

    /**
     * **ExpI — Immediate exponential with default tensor reference**
     *
     * Computes the element-wise exponential (eˣ) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the exponential of the input tensor.
     * @see #exp(String, String)
     */
    default Tensor expI() {
        String oName = genRandomNameUnary();
        return exp("", oName).get(oName);
    }

    /**
     * **ExpI — Immediate exponential of a named tensor**
     *
     * Computes the element-wise exponential (eˣ) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the exponential of the input tensor.
     * @see #exp(String, String)
     */
    default Tensor expI(String a) {
        String oName = genRandomNameUnary();
        return exp(a, oName).get(oName);
    }

    /**
     * **ExpI — Immediate exponential of a Tensor object**
     *
     * Computes the element-wise exponential (eˣ) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #exp(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the exponential of the input tensor.
     * @see #exp(String, String)
     */
    default Tensor expI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return exp(aName, oName).get(oName);
    }


    /**
     * **Sin — Basic sine with empty tensor reference**
     *
     * Computes the element-wise sine of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the sine operation.
     * @see #sin(String, String)
     */
    default CuBridge sin() {
        return sin("", genRandomNameUnary());
    }

    /**
     * **Sin — Sine of a named tensor**
     *
     * Computes the element-wise sine of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the sine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge sin(String a, String out) {
        if (CuBridgeJNI.sin(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | sin | " + a + " | " + out);
        return null;
    }

    /**
     * **Sin — Sine of a Tensor object**
     *
     * Computes the element-wise sine of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #sin(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the sine operation.
     * @see #sin(String, String)
     */
    default CuBridge sin(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return sin(aName, out);
    }

    /**
     * **SinI — Immediate sine with default tensor reference**
     *
     * Computes the element-wise sine of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the sine of the input tensor.
     * @see #sin(String, String)
     */
    default Tensor sinI() {
        String oName = genRandomNameUnary();
        return sin("", oName).get(oName);
    }

    /**
     * **SinI — Immediate sine of a named tensor**
     *
     * Computes the element-wise sine of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the sine of the input tensor.
     * @see #sin(String, String)
     */
    default Tensor sinI(String a) {
        String oName = genRandomNameUnary();
        return sin(a, oName).get(oName);
    }

    /**
     * **SinI — Immediate sine of a Tensor object**
     *
     * Computes the element-wise sine of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #sin(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the sine of the input tensor.
     * @see #sin(String, String)
     */
    default Tensor sinI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return sin(aName, oName).get(oName);
    }


    /**
     * **Cos — Basic cosine with empty tensor reference**
     *
     * Computes the element-wise cosine of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the cosine operation.
     * @see #cos(String, String)
     */
    default CuBridge cos() {
        return cos("", genRandomNameUnary());
    }

    /**
     * **Cos — Cosine of a named tensor**
     *
     * Computes the element-wise cosine of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the cosine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge cos(String a, String out) {
        if (CuBridgeJNI.cos(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | cos | " + a + " | " + out);
        return null;
    }

    /**
     * **Cos — Cosine of a Tensor object**
     *
     * Computes the element-wise cosine of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #cos(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the cosine operation.
     * @see #cos(String, String)
     */
    default CuBridge cos(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return cos(aName, out);
    }

    /**
     * **CosI — Immediate cosine with default tensor reference**
     *
     * Computes the element-wise cosine of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the cosine of the input tensor.
     * @see #cos(String, String)
     */
    default Tensor cosI() {
        String oName = genRandomNameUnary();
        return cos("", oName).get(oName);
    }

    /**
     * **CosI — Immediate cosine of a named tensor**
     *
     * Computes the element-wise cosine of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the cosine of the input tensor.
     * @see #cos(String, String)
     */
    default Tensor cosI(String a) {
        String oName = genRandomNameUnary();
        return cos(a, oName).get(oName);
    }

    /**
     * **CosI — Immediate cosine of a Tensor object**
     *
     * Computes the element-wise cosine of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #cos(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the cosine of the input tensor.
     * @see #cos(String, String)
     */
    default Tensor cosI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return cos(aName, oName).get(oName);
    }


    /**
     * **Tan — Basic tangent with empty tensor reference**
     *
     * Computes the element-wise tangent of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the tangent operation.
     * @see #tan(String, String)
     */
    default CuBridge tan() {
        return tan("", genRandomNameUnary());
    }

    /**
     * **Tan — Tangent of a named tensor**
     *
     * Computes the element-wise tangent of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the tangent operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge tan(String a, String out) {
        if (CuBridgeJNI.tan(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | tan | " + a + " | " + out);
        return null;
    }

    /**
     * **Tan — Tangent of a Tensor object**
     *
     * Computes the element-wise tangent of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #tan(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the tangent operation.
     * @see #tan(String, String)
     */
    default CuBridge tan(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return tan(aName, out);
    }

    /**
     * **TanI — Immediate tangent with default tensor reference**
     *
     * Computes the element-wise tangent of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the tangent of the input tensor.
     * @see #tan(String, String)
     */
    default Tensor tanI() {
        String oName = genRandomNameUnary();
        return tan("", oName).get(oName);
    }

    /**
     * **TanI — Immediate tangent of a named tensor**
     *
     * Computes the element-wise tangent of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the tangent of the input tensor.
     * @see #tan(String, String)
     */
    default Tensor tanI(String a) {
        String oName = genRandomNameUnary();
        return tan(a, oName).get(oName);
    }

    /**
     * **TanI — Immediate tangent of a Tensor object**
     *
     * Computes the element-wise tangent of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #tan(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the tangent of the input tensor.
     * @see #tan(String, String)
     */
    default Tensor tanI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return tan(aName, oName).get(oName);
    }


    /**
     * **Sinh — Basic hyperbolic sine with empty tensor reference**
     *
     * Computes the element-wise hyperbolic sine (sinh(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the hyperbolic sine operation.
     * @see #sinh(String, String)
     */
    default CuBridge sinh() {
        return sinh("", genRandomNameUnary());
    }

    /**
     * **Sinh — Hyperbolic sine of a named tensor**
     *
     * Computes the element-wise hyperbolic sine (sinh(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the hyperbolic sine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge sinh(String a, String out) {
        if (CuBridgeJNI.sinh(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | sinh | " + a + " | " + out);
        return null;
    }

    /**
     * **Sinh — Hyperbolic sine of a Tensor object**
     *
     * Computes the element-wise hyperbolic sine (sinh(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #sinh(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the hyperbolic sine operation.
     * @see #sinh(String, String)
     */
    default CuBridge sinh(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return sinh(aName, out);
    }

    /**
     * **SinhI — Immediate hyperbolic sine with default tensor reference**
     *
     * Computes the element-wise hyperbolic sine (sinh(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing sinh(x) of the input tensor.
     * @see #sinh(String, String)
     */
    default Tensor sinhI() {
        String oName = genRandomNameUnary();
        return sinh("", oName).get(oName);
    }

    /**
     * **SinhI — Immediate hyperbolic sine of a named tensor**
     *
     * Computes the element-wise hyperbolic sine (sinh(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing sinh(x) of the input tensor.
     * @see #sinh(String, String)
     */
    default Tensor sinhI(String a) {
        String oName = genRandomNameUnary();
        return sinh(a, oName).get(oName);
    }

    /**
     * **SinhI — Immediate hyperbolic sine of a Tensor object**
     *
     * Computes the element-wise hyperbolic sine (sinh(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #sinh(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing sinh(x) of the input tensor.
     * @see #sinh(String, String)
     */
    default Tensor sinhI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return sinh(aName, oName).get(oName);
    }


    /**
     * **Cosh — Basic hyperbolic cosine with empty tensor reference**
     *
     * Computes the element-wise hyperbolic cosine (cosh(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the hyperbolic cosine operation.
     * @see #cosh(String, String)
     */
    default CuBridge cosh() {
        return cosh("", genRandomNameUnary());
    }

    /**
     * **Cosh — Hyperbolic cosine of a named tensor**
     *
     * Computes the element-wise hyperbolic cosine (cosh(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the hyperbolic cosine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge cosh(String a, String out) {
        if (CuBridgeJNI.cosh(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | cosh | " + a + " | " + out);
        return null;
    }

    /**
     * **Cosh — Hyperbolic cosine of a Tensor object**
     *
     * Computes the element-wise hyperbolic cosine (cosh(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #cosh(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the hyperbolic cosine operation.
     * @see #cosh(String, String)
     */
    default CuBridge cosh(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return cosh(aName, out);
    }

    /**
     * **CoshI — Immediate hyperbolic cosine with default tensor reference**
     *
     * Computes the element-wise hyperbolic cosine (cosh(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing cosh(x) of the input tensor.
     * @see #cosh(String, String)
     */
    default Tensor coshI() {
        String oName = genRandomNameUnary();
        return cosh("", oName).get(oName);
    }

    /**
     * **CoshI — Immediate hyperbolic cosine of a named tensor**
     *
     * Computes the element-wise hyperbolic cosine (cosh(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing cosh(x) of the input tensor.
     * @see #cosh(String, String)
     */
    default Tensor coshI(String a) {
        String oName = genRandomNameUnary();
        return cosh(a, oName).get(oName);
    }

    /**
     * **CoshI — Immediate hyperbolic cosine of a Tensor object**
     *
     * Computes the element-wise hyperbolic cosine (cosh(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #cosh(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing cosh(x) of the input tensor.
     * @see #cosh(String, String)
     */
    default Tensor coshI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return cosh(aName, oName).get(oName);
    }


    /**
     * **Tanh — Basic hyperbolic tangent with empty tensor reference**
     *
     * Computes the element-wise hyperbolic tangent (tanh(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the hyperbolic tangent operation.
     * @see #tanh(String, String)
     */
    default CuBridge tanh() {
        return tanh("", genRandomNameUnary());
    }

    /**
     * **Tanh — Hyperbolic tangent of a named tensor**
     *
     * Computes the element-wise hyperbolic tangent (tanh(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the hyperbolic tangent operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge tanh(String a, String out) {
        if (CuBridgeJNI.tanh(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | tanh | " + a + " | " + out);
        return null;
    }

    /**
     * **Tanh — Hyperbolic tangent of a Tensor object**
     *
     * Computes the element-wise hyperbolic tangent (tanh(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #tanh(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the hyperbolic tangent operation.
     * @see #tanh(String, String)
     */
    default CuBridge tanh(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return tanh(aName, out);
    }

    /**
     * **TanhI — Immediate hyperbolic tangent with default tensor reference**
     *
     * Computes the element-wise hyperbolic tangent (tanh(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing tanh(x) of the input tensor.
     * @see #tanh(String, String)
     */
    default Tensor tanhI() {
        String oName = genRandomNameUnary();
        return tanh("", oName).get(oName);
    }

    /**
     * **TanhI — Immediate hyperbolic tangent of a named tensor**
     *
     * Computes the element-wise hyperbolic tangent (tanh(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing tanh(x) of the input tensor.
     * @see #tanh(String, String)
     */
    default Tensor tanhI(String a) {
        String oName = genRandomNameUnary();
        return tanh(a, oName).get(oName);
    }

    /**
     * **TanhI — Immediate hyperbolic tangent of a Tensor object**
     *
     * Computes the element-wise hyperbolic tangent (tanh(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #tanh(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing tanh(x) of the input tensor.
     * @see #tanh(String, String)
     */
    default Tensor tanhI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return tanh(aName, oName).get(oName);
    }


    /**
     * **Asin — Basic arcsine with empty tensor reference**
     *
     * Computes the element-wise arcsine (inverse sine, asin(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the arcsine operation.
     * @see #asin(String, String)
     */
    default CuBridge asin() {
        return asin("", genRandomNameUnary());
    }

    /**
     * **Asin — Arcsine of a named tensor**
     *
     * Computes the element-wise arcsine (inverse sine, asin(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the arcsine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge asin(String a, String out) {
        if (CuBridgeJNI.asin(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | asin | " + a + " | " + out);
        return null;
    }

    /**
     * **Asin — Arcsine of a Tensor object**
     *
     * Computes the element-wise arcsine (inverse sine, asin(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #asin(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the arcsine operation.
     * @see #asin(String, String)
     */
    default CuBridge asin(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return asin(aName, out);
    }

    /**
     * **AsinI — Immediate arcsine with default tensor reference**
     *
     * Computes the element-wise arcsine (inverse sine, asin(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing asin(x) of the input tensor.
     * @see #asin(String, String)
     */
    default Tensor asinI() {
        String oName = genRandomNameUnary();
        return asin("", oName).get(oName);
    }

    /**
     * **AsinI — Immediate arcsine of a named tensor**
     *
     * Computes the element-wise arcsine (inverse sine, asin(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing asin(x) of the input tensor.
     * @see #asin(String, String)
     */
    default Tensor asinI(String a) {
        String oName = genRandomNameUnary();
        return asin(a, oName).get(oName);
    }

    /**
     * **AsinI — Immediate arcsine of a Tensor object**
     *
     * Computes the element-wise arcsine (inverse sine, asin(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #asin(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing asin(x) of the input tensor.
     * @see #asin(String, String)
     */
    default Tensor asinI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return asin(aName, oName).get(oName);
    }


    /**
     * **Acos — Basic arccosine with empty tensor reference**
     *
     * Computes the element-wise arccosine (inverse cosine, acos(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the arccosine operation.
     * @see #acos(String, String)
     */
    default CuBridge acos() {
        return acos("", genRandomNameUnary());
    }

    /**
     * **Acos — Arccosine of a named tensor**
     *
     * Computes the element-wise arccosine (inverse cosine, acos(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the arccosine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge acos(String a, String out) {
        if (CuBridgeJNI.acos(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | acos | " + a + " | " + out);
        return null;
    }

    /**
     * **Acos — Arccosine of a Tensor object**
     *
     * Computes the element-wise arccosine (inverse cosine, acos(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #acos(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the arccosine operation.
     * @see #acos(String, String)
     */
    default CuBridge acos(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return acos(aName, out);
    }

    /**
     * **AcosI — Immediate arccosine with default tensor reference**
     *
     * Computes the element-wise arccosine (inverse cosine, acos(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing acos(x) of the input tensor.
     * @see #acos(String, String)
     */
    default Tensor acosI() {
        String oName = genRandomNameUnary();
        return acos("", oName).get(oName);
    }

    /**
     * **AcosI — Immediate arccosine of a named tensor**
     *
     * Computes the element-wise arccosine (inverse cosine, acos(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing acos(x) of the input tensor.
     * @see #acos(String, String)
     */
    default Tensor acosI(String a) {
        String oName = genRandomNameUnary();
        return acos(a, oName).get(oName);
    }

    /**
     * **AcosI — Immediate arccosine of a Tensor object**
     *
     * Computes the element-wise arccosine (inverse cosine, acos(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #acos(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing acos(x) of the input tensor.
     * @see #acos(String, String)
     */
    default Tensor acosI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return acos(aName, oName).get(oName);
    }


    /**
     * **Atan — Basic arctangent with empty tensor reference**
     *
     * Computes the element-wise arctangent (inverse tangent, atan(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the arctangent operation.
     * @see #atan(String, String)
     */
    default CuBridge atan() {
        return atan("", genRandomNameUnary());
    }

    /**
     * **Atan — Arctangent of a named tensor**
     *
     * Computes the element-wise arctangent (inverse tangent, atan(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the arctangent operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge atan(String a, String out) {
        if (CuBridgeJNI.atan(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | atan | " + a + " | " + out);
        return null;
    }

    /**
     * **Atan — Arctangent of a Tensor object**
     *
     * Computes the element-wise arctangent (inverse tangent, atan(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #atan(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the arctangent operation.
     * @see #atan(String, String)
     */
    default CuBridge atan(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return atan(aName, out);
    }

    /**
     * **AtanI — Immediate arctangent with default tensor reference**
     *
     * Computes the element-wise arctangent (inverse tangent, atan(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing atan(x) of the input tensor.
     * @see #atan(String, String)
     */
    default Tensor atanI() {
        String oName = genRandomNameUnary();
        return atan("", oName).get(oName);
    }

    /**
     * **AtanI — Immediate arctangent of a named tensor**
     *
     * Computes the element-wise arctangent (inverse tangent, atan(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing atan(x) of the input tensor.
     * @see #atan(String, String)
     */
    default Tensor atanI(String a) {
        String oName = genRandomNameUnary();
        return atan(a, oName).get(oName);
    }

    /**
     * **AtanI — Immediate arctangent of a Tensor object**
     *
     * Computes the element-wise arctangent (inverse tangent, atan(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #atan(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing atan(x) of the input tensor.
     * @see #atan(String, String)
     */
    default Tensor atanI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return atan(aName, oName).get(oName);
    }


    /**
     * **Asinh — Basic inverse hyperbolic sine with empty tensor reference**
     *
     * Computes the element-wise inverse hyperbolic sine (asinh(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the inverse hyperbolic sine operation.
     * @see #asinh(String, String)
     */
    default CuBridge asinh() {
        return asinh("", genRandomNameUnary());
    }

    /**
     * **Asinh — Inverse hyperbolic sine of a named tensor**
     *
     * Computes the element-wise inverse hyperbolic sine (asinh(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the inverse hyperbolic sine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge asinh(String a, String out) {
        if (CuBridgeJNI.asinh(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | asinh | " + a + " | " + out);
        return null;
    }

    /**
     * **Asinh — Inverse hyperbolic sine of a Tensor object**
     *
     * Computes the element-wise inverse hyperbolic sine (asinh(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #asinh(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the inverse hyperbolic sine operation.
     * @see #asinh(String, String)
     */
    default CuBridge asinh(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return asinh(aName, out);
    }

    /**
     * **AsinhI — Immediate inverse hyperbolic sine with default tensor reference**
     *
     * Computes the element-wise inverse hyperbolic sine (asinh(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing asinh(x) of the input tensor.
     * @see #asinh(String, String)
     */
    default Tensor asinhI() {
        String oName = genRandomNameUnary();
        return asinh("", oName).get(oName);
    }

    /**
     * **AsinhI — Immediate inverse hyperbolic sine of a named tensor**
     *
     * Computes the element-wise inverse hyperbolic sine (asinh(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing asinh(x) of the input tensor.
     * @see #asinh(String, String)
     */
    default Tensor asinhI(String a) {
        String oName = genRandomNameUnary();
        return asinh(a, oName).get(oName);
    }

    /**
     * **AsinhI — Immediate inverse hyperbolic sine of a Tensor object**
     *
     * Computes the element-wise inverse hyperbolic sine (asinh(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #asinh(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing asinh(x) of the input tensor.
     * @see #asinh(String, String)
     */
    default Tensor asinhI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return asinh(aName, oName).get(oName);
    }


    /**
     * **Acosh — Basic inverse hyperbolic cosine with empty tensor reference**
     *
     * Computes the element-wise inverse hyperbolic cosine (acosh(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the inverse hyperbolic cosine operation.
     * @see #acosh(String, String)
     */
    default CuBridge acosh() {
        return acosh("", genRandomNameUnary());
    }

    /**
     * **Acosh — Inverse hyperbolic cosine of a named tensor**
     *
     * Computes the element-wise inverse hyperbolic cosine (acosh(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the inverse hyperbolic cosine operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge acosh(String a, String out) {
        if (CuBridgeJNI.acosh(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | acosh | " + a + " | " + out);
        return null;
    }

    /**
     * **Acosh — Inverse hyperbolic cosine of a Tensor object**
     *
     * Computes the element-wise inverse hyperbolic cosine (acosh(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #acosh(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the inverse hyperbolic cosine operation.
     * @see #acosh(String, String)
     */
    default CuBridge acosh(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return acosh(aName, out);
    }

    /**
     * **AcoshI — Immediate inverse hyperbolic cosine with default tensor reference**
     *
     * Computes the element-wise inverse hyperbolic cosine (acosh(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing acosh(x) of the input tensor.
     * @see #acosh(String, String)
     */
    default Tensor acoshI() {
        String oName = genRandomNameUnary();
        return acosh("", oName).get(oName);
    }

    /**
     * **AcoshI — Immediate inverse hyperbolic cosine of a named tensor**
     *
     * Computes the element-wise inverse hyperbolic cosine (acosh(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing acosh(x) of the input tensor.
     * @see #acosh(String, String)
     */
    default Tensor acoshI(String a) {
        String oName = genRandomNameUnary();
        return acosh(a, oName).get(oName);
    }

    /**
     * **AcoshI — Immediate inverse hyperbolic cosine of a Tensor object**
     *
     * Computes the element-wise inverse hyperbolic cosine (acosh(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #acosh(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing acosh(x) of the input tensor.
     * @see #acosh(String, String)
     */
    default Tensor acoshI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return acosh(aName, oName).get(oName);
    }


    /**
     * **Atanh — Basic inverse hyperbolic tangent with empty tensor reference**
     *
     * Computes the element-wise inverse hyperbolic tangent (atanh(x)) of a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the inverse hyperbolic tangent operation.
     * @see #atanh(String, String)
     */
    default CuBridge atanh() {
        return atanh("", genRandomNameUnary());
    }

    /**
     * **Atanh — Inverse hyperbolic tangent of a named tensor**
     *
     * Computes the element-wise inverse hyperbolic tangent (atanh(x)) of the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the inverse hyperbolic tangent operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge atanh(String a, String out) {
        if (CuBridgeJNI.atanh(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | atanh | " + a + " | " + out);
        return null;
    }

    /**
     * **Atanh — Inverse hyperbolic tangent of a Tensor object**
     *
     * Computes the element-wise inverse hyperbolic tangent (atanh(x)) of the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #atanh(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the inverse hyperbolic tangent operation.
     * @see #atanh(String, String)
     */
    default CuBridge atanh(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return atanh(aName, out);
    }

    /**
     * **AtanhI — Immediate inverse hyperbolic tangent with default tensor reference**
     *
     * Computes the element-wise inverse hyperbolic tangent (atanh(x)) of a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing atanh(x) of the input tensor.
     * @see #atanh(String, String)
     */
    default Tensor atanhI() {
        String oName = genRandomNameUnary();
        return atanh("", oName).get(oName);
    }

    /**
     * **AtanhI — Immediate inverse hyperbolic tangent of a named tensor**
     *
     * Computes the element-wise inverse hyperbolic tangent (atanh(x)) of the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing atanh(x) of the input tensor.
     * @see #atanh(String, String)
     */
    default Tensor atanhI(String a) {
        String oName = genRandomNameUnary();
        return atanh(a, oName).get(oName);
    }

    /**
     * **AtanhI — Immediate inverse hyperbolic tangent of a Tensor object**
     *
     * Computes the element-wise inverse hyperbolic tangent (atanh(x)) of the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #atanh(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing atanh(x) of the input tensor.
     * @see #atanh(String, String)
     */
    default Tensor atanhI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return atanh(aName, oName).get(oName);
    }


    /**
     * **Step — Basic step function with empty tensor reference**
     *
     * Applies the element-wise step function to a tensor already stored in the internal queue.
     * <p>
     * The step function outputs 1 for all elements greater than or equal to 0, and 0 otherwise.
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the step operation.
     * @see #step(String, String)
     */
    default CuBridge step() {
        return step("", genRandomNameUnary());
    }

    /**
     * **Step — Step function on a named tensor**
     *
     * Applies the element-wise step function to the specified tensor.
     * <p>
     * Each element x is transformed to 1 if x ≥ 0, and 0 otherwise.
     * The result is stored under the name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the step operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge step(String a, String out) {
        if (CuBridgeJNI.step(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | step | " + a + " | " + out);
        return null;
    }

    /**
     * **Step — Step function on a Tensor object**
     *
     * Applies the element-wise step function to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #step(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the step operation.
     * @see #step(String, String)
     */
    default CuBridge step(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return step(aName, out);
    }

    /**
     * **StepI — Immediate step function with default tensor reference**
     *
     * Applies the element-wise step function to a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Each element x is transformed to 1 if x ≥ 0, and 0 otherwise.
     * </p>
     *
     * @return A new {@link Tensor} representing the step-transformed tensor.
     * @see #step(String, String)
     */
    default Tensor stepI() {
        String oName = genRandomNameUnary();
        return step("", oName).get(oName);
    }

    /**
     * **StepI — Immediate step function of a named tensor**
     *
     * Applies the element-wise step function to the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the step-transformed tensor.
     * @see #step(String, String)
     */
    default Tensor stepI(String a) {
        String oName = genRandomNameUnary();
        return step(a, oName).get(oName);
    }

    /**
     * **StepI — Immediate step function of a Tensor object**
     *
     * Applies the element-wise step function to the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #step(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the step-transformed tensor.
     * @see #step(String, String)
     */
    default Tensor stepI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return step(aName, oName).get(oName);
    }


    /**
     * **Sigmoid — Basic sigmoid activation with empty tensor reference**
     *
     * Applies the element-wise sigmoid function σ(x) = 1 / (1 + e^-x)
     * to a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the sigmoid operation.
     * @see #sigmoid(String, String)
     */
    default CuBridge sigmoid() {
        return sigmoid("", genRandomNameUnary());
    }

    /**
     * **Sigmoid — Sigmoid activation on a named tensor**
     *
     * Applies the element-wise sigmoid activation function σ(x) = 1 / (1 + e^-x)
     * to the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the sigmoid operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge sigmoid(String a, String out) {
        if (CuBridgeJNI.sigmoid(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | sigmoid | " + a + " | " + out);
        return null;
    }

    /**
     * **Sigmoid — Sigmoid activation on a Tensor object**
     *
     * Applies the element-wise sigmoid activation σ(x) = 1 / (1 + e^-x)
     * to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #sigmoid(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the sigmoid operation.
     * @see #sigmoid(String, String)
     */
    default CuBridge sigmoid(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return sigmoid(aName, out);
    }

    /**
     * **SigmoidI — Immediate sigmoid activation with default tensor reference**
     *
     * Applies the element-wise sigmoid activation σ(x) = 1 / (1 + e^-x)
     * to a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the sigmoid-transformed tensor.
     * @see #sigmoid(String, String)
     */
    default Tensor sigmoidI() {
        String oName = genRandomNameUnary();
        return sigmoid("", oName).get(oName);
    }

    /**
     * **SigmoidI — Immediate sigmoid activation of a named tensor**
     *
     * Applies the element-wise sigmoid activation σ(x) = 1 / (1 + e^-x)
     * to the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the sigmoid-transformed tensor.
     * @see #sigmoid(String, String)
     */
    default Tensor sigmoidI(String a) {
        String oName = genRandomNameUnary();
        return sigmoid(a, oName).get(oName);
    }

    /**
     * **SigmoidI — Immediate sigmoid activation of a Tensor object**
     *
     * Applies the element-wise sigmoid activation σ(x) = 1 / (1 + e^-x)
     * to the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #sigmoid(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the sigmoid-transformed tensor.
     * @see #sigmoid(String, String)
     */
    default Tensor sigmoidI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return sigmoid(aName, oName).get(oName);
    }


    /**
     * **Relu — Basic Rectified Linear Unit (ReLU) activation with empty tensor reference**
     *
     * Applies the element-wise ReLU activation function ReLU(x) = max(0, x)
     * to a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the ReLU operation.
     * @see #relu(String, String)
     */
    default CuBridge relu() {
        return relu("", genRandomNameUnary());
    }

    /**
     * **Relu — ReLU activation on a named tensor**
     *
     * Applies the element-wise ReLU activation function ReLU(x) = max(0, x)
     * to the specified tensor.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the ReLU operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge relu(String a, String out) {
        if (CuBridgeJNI.relu(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | relu | " + a + " | " + out);
        return null;
    }

    /**
     * **Relu — ReLU activation on a Tensor object**
     *
     * Applies the element-wise ReLU activation function ReLU(x) = max(0, x)
     * to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #relu(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the ReLU operation.
     * @see #relu(String, String)
     */
    default CuBridge relu(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return relu(aName, out);
    }

    /**
     * **ReluI — Immediate ReLU with default tensor reference**
     *
     * Applies the element-wise ReLU activation function ReLU(x) = max(0, x)
     * to a tensor already stored in the queue and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the ReLU-transformed tensor.
     * @see #relu(String, String)
     */
    default Tensor reluI() {
        String oName = genRandomNameUnary();
        return relu("", oName).get(oName);
    }

    /**
     * **ReluI — Immediate ReLU of a named tensor**
     *
     * Applies the element-wise ReLU activation function ReLU(x) = max(0, x)
     * to the specified named tensor and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the ReLU-transformed tensor.
     * @see #relu(String, String)
     */
    default Tensor reluI(String a) {
        String oName = genRandomNameUnary();
        return relu(a, oName).get(oName);
    }

    /**
     * **ReluI — Immediate ReLU of a Tensor object**
     *
     * Applies the element-wise ReLU activation function ReLU(x) = max(0, x)
     * to the given {@link Tensor} object and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #relu(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the ReLU-transformed tensor.
     * @see #relu(String, String)
     */
    default Tensor reluI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return relu(aName, oName).get(oName);
    }


    /**
     * **LeakRelu — Basic Leaky ReLU activation with empty tensor reference**
     *
     * Applies the element-wise Leaky ReLU activation function:
     * <p>
     * LeakyReLU(x) = x if x > 0, otherwise α·x (where α is a small constant, typically 0.01)
     * </p>
     * to a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the Leaky ReLU operation.
     * @see #leakRelu(String, String)
     */
    default CuBridge leakRelu() {
        return leakRelu("", genRandomNameUnary());
    }

    /**
     * **LeakRelu — Leaky ReLU activation on a named tensor**
     *
     * Applies the element-wise Leaky ReLU activation function:
     * LeakyReLU(x) = x if x > 0, otherwise α·x.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the Leaky ReLU operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge leakRelu(String a, String out) {
        if (CuBridgeJNI.leakRelu(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | leakRelu | " + a + " | " + out);
        return null;
    }

    /**
     * **LeakRelu — Leaky ReLU activation on a Tensor object**
     *
     * Applies the element-wise Leaky ReLU activation function:
     * LeakyReLU(x) = x if x > 0, otherwise α·x.
     * <p>
     * Automatically assigns a random internal name to the input tensor before executing
     * {@link #leakRelu(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the Leaky ReLU operation.
     * @see #leakRelu(String, String)
     */
    default CuBridge leakRelu(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return leakRelu(aName, out);
    }

    /**
     * **LeakReluI — Immediate Leaky ReLU with default tensor reference**
     *
     * Applies the element-wise Leaky ReLU activation function:
     * LeakyReLU(x) = x if x > 0, otherwise α·x.
     * <p>
     * Computes on the topmost tensor in the queue and returns the resulting {@link Tensor}.
     * </p>
     *
     * @return A new {@link Tensor} representing the Leaky ReLU-transformed tensor.
     * @see #leakRelu(String, String)
     */
    default Tensor leakReluI() {
        String oName = genRandomNameUnary();
        return leakRelu("", oName).get(oName);
    }

    /**
     * **LeakReluI — Immediate Leaky ReLU of a named tensor**
     *
     * Applies the element-wise Leaky ReLU activation function:
     * LeakyReLU(x) = x if x > 0, otherwise α·x.
     * <p>
     * Returns the resulting {@link Tensor} directly.
     * </p>
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the Leaky ReLU-transformed tensor.
     * @see #leakRelu(String, String)
     */
    default Tensor leakReluI(String a) {
        String oName = genRandomNameUnary();
        return leakRelu(a, oName).get(oName);
    }

    /**
     * **LeakReluI — Immediate Leaky ReLU of a Tensor object**
     *
     * Applies the element-wise Leaky ReLU activation function:
     * LeakyReLU(x) = x if x > 0, otherwise α·x.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #leakRelu(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the Leaky ReLU-transformed tensor.
     * @see #leakRelu(String, String)
     */
    default Tensor leakReluI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return leakRelu(aName, oName).get(oName);
    }


    /**
     * **Softplus — Basic Softplus activation with empty tensor reference**
     *
     * Applies the element-wise Softplus activation function:
     * <p>
     * Softplus(x) = log(1 + eˣ)
     * </p>
     * to a tensor already stored in the internal queue.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the Softplus operation.
     * @see #softplus(String, String)
     */
    default CuBridge softplus() {
        return softplus("", genRandomNameUnary());
    }

    /**
     * **Softplus — Softplus activation on a named tensor**
     *
     * Applies the element-wise Softplus activation function:
     * <p>
     * Softplus(x) = log(1 + eˣ)
     * </p>
     * to the specified tensor and stores the result under the name {@code out}.
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the Softplus operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge softplus(String a, String out) {
        if (CuBridgeJNI.softplus(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | softplus | " + a + " | " + out);
        return null;
    }

    /**
     * **Softplus — Softplus activation on a Tensor object**
     *
     * Applies the element-wise Softplus activation function:
     * <p>
     * Softplus(x) = log(1 + eˣ)
     * </p>
     * to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #softplus(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the Softplus operation.
     * @see #softplus(String, String)
     */
    default CuBridge softplus(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return softplus(aName, out);
    }

    /**
     * **SoftplusI — Immediate Softplus activation with default tensor reference**
     *
     * Applies the element-wise Softplus activation function:
     * <p>
     * Softplus(x) = log(1 + eˣ)
     * </p>
     * to a tensor already stored in the queue and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the Softplus-transformed tensor.
     * @see #softplus(String, String)
     */
    default Tensor softplusI() {
        String oName = genRandomNameUnary();
        return softplus("", oName).get(oName);
    }

    /**
     * **SoftplusI — Immediate Softplus activation of a named tensor**
     *
     * Applies the element-wise Softplus activation function:
     * <p>
     * Softplus(x) = log(1 + eˣ)
     * </p>
     * to the specified named tensor and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the Softplus-transformed tensor.
     * @see #softplus(String, String)
     */
    default Tensor softplusI(String a) {
        String oName = genRandomNameUnary();
        return softplus(a, oName).get(oName);
    }

    /**
     * **SoftplusI — Immediate Softplus activation of a Tensor object**
     *
     * Applies the element-wise Softplus activation function:
     * <p>
     * Softplus(x) = log(1 + eˣ)
     * </p>
     * to the given {@link Tensor} object and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #softplus(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the Softplus-transformed tensor.
     * @see #softplus(String, String)
     */
    default Tensor softplusI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return softplus(aName, oName).get(oName);
    }


    /**
     * **Round — Basic rounding operation with empty tensor reference**
     *
     * Rounds each element of a tensor to the nearest integer value.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * The result is stored under an automatically generated internal name.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the rounding operation.
     * @see #round(String, String)
     */
    default CuBridge round() {
        return round("", genRandomNameUnary());
    }

    /**
     * **Round — Rounding on a named tensor**
     *
     * Rounds each element of the specified tensor to the nearest integer value.
     * <p>
     * The result is stored in the tensor name specified by {@code out}.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the rounding operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge round(String a, String out) {
        if (CuBridgeJNI.round(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | round | " + a + " | " + out);
        return null;
    }

    /**
     * **Round — Rounding on a Tensor object**
     *
     * Rounds each element of the given {@link Tensor} object to the nearest integer value.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #round(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the rounding operation.
     * @see #round(String, String)
     */
    default CuBridge round(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return round(aName, out);
    }

    /**
     * **RoundI — Immediate rounding with default tensor reference**
     *
     * Rounds each element of a tensor already stored in the queue to the nearest integer value,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the rounded tensor.
     * @see #round(String, String)
     */
    default Tensor roundI() {
        String oName = genRandomNameUnary();
        return round("", oName).get(oName);
    }

    /**
     * **RoundI — Immediate rounding of a named tensor**
     *
     * Rounds each element of the specified named tensor to the nearest integer value,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the rounded tensor.
     * @see #round(String, String)
     */
    default Tensor roundI(String a) {
        String oName = genRandomNameUnary();
        return round(a, oName).get(oName);
    }

    /**
     * **RoundI — Immediate rounding of a Tensor object**
     *
     * Rounds each element of the given {@link Tensor} object to the nearest integer value,
     * and immediately returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #round(String, String)}.
     * </p>
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the rounded tensor.
     * @see #round(String, String)
     */
    default Tensor roundI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return round(aName, oName).get(oName);
    }


    /**
     * **Ceil — Basic ceiling operation with empty tensor reference**
     *
     * Applies the element-wise ceiling function to a tensor, rounding each element up
     * to the nearest integer greater than or equal to that element.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the ceiling operation.
     * @see #ceil(String, String)
     */
    default CuBridge ceil() {
        return ceil("", genRandomNameUnary());
    }

    /**
     * **Ceil — Ceiling on a named tensor**
     *
     * Applies the element-wise ceiling function to the specified tensor, rounding each element up
     * to the nearest integer greater than or equal to that element.
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the ceiling operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge ceil(String a, String out) {
        if (CuBridgeJNI.ceil(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | ceil | " + a + " | " + out);
        return null;
    }

    /**
     * **Ceil — Ceiling on a Tensor object**
     *
     * Applies the element-wise ceiling function to the given {@link Tensor} object,
     * rounding each element up to the nearest integer greater than or equal to that element.
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the ceiling operation.
     * @see #ceil(String, String)
     */
    default CuBridge ceil(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return ceil(aName, out);
    }

    /**
     * **CeilI — Immediate ceiling operation with default tensor reference**
     *
     * Applies the element-wise ceiling function to a tensor already stored in the queue,
     * rounding each element up to the nearest integer greater than or equal to that element,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the ceiling-transformed tensor.
     * @see #ceil(String, String)
     */
    default Tensor ceilI() {
        String oName = genRandomNameUnary();
        return ceil("", oName).get(oName);
    }

    /**
     * **CeilI — Immediate ceiling of a named tensor**
     *
     * Applies the element-wise ceiling function to the specified named tensor,
     * rounding each element up to the nearest integer greater than or equal to that element,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the ceiling-transformed tensor.
     * @see #ceil(String, String)
     */
    default Tensor ceilI(String a) {
        String oName = genRandomNameUnary();
        return ceil(a, oName).get(oName);
    }

    /**
     * **CeilI — Immediate ceiling of a Tensor object**
     *
     * Applies the element-wise ceiling function to the given {@link Tensor} object,
     * rounding each element up to the nearest integer greater than or equal to that element,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the ceiling-transformed tensor.
     * @see #ceil(String, String)
     */
    default Tensor ceilI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return ceil(aName, oName).get(oName);
    }


    /**
     * **Floor — Basic floor operation with empty tensor reference**
     *
     * Applies the element-wise floor function to a tensor, rounding each element down
     * to the nearest integer less than or equal to that element.
     * <p>
     * When no tensor name is provided, this function assumes the topmost tensor in the queue.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the floor operation.
     * @see #floor(String, String)
     */
    default CuBridge floor() {
        return floor("", genRandomNameUnary());
    }

    /**
     * **Floor — Floor operation on a named tensor**
     *
     * Applies the element-wise floor function to the specified tensor, rounding each element down
     * to the nearest integer less than or equal to that element.
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the floor operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge floor(String a, String out) {
        if (CuBridgeJNI.floor(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | floor | " + a + " | " + out);
        return null;
    }

    /**
     * **Floor — Floor operation on a Tensor object**
     *
     * Applies the element-wise floor function to the given {@link Tensor} object,
     * rounding each element down to the nearest integer less than or equal to that element.
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the floor operation.
     * @see #floor(String, String)
     */
    default CuBridge floor(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return floor(aName, out);
    }

    /**
     * **FloorI — Immediate floor operation with default tensor reference**
     *
     * Applies the element-wise floor function to a tensor already stored in the queue,
     * rounding each element down to the nearest integer less than or equal to that element,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the floor-transformed tensor.
     * @see #floor(String, String)
     */
    default Tensor floorI() {
        String oName = genRandomNameUnary();
        return floor("", oName).get(oName);
    }

    /**
     * **FloorI — Immediate floor of a named tensor**
     *
     * Applies the element-wise floor function to the specified named tensor,
     * rounding each element down to the nearest integer less than or equal to that element,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the floor-transformed tensor.
     * @see #floor(String, String)
     */
    default Tensor floorI(String a) {
        String oName = genRandomNameUnary();
        return floor(a, oName).get(oName);
    }

    /**
     * **FloorI — Immediate floor of a Tensor object**
     *
     * Applies the element-wise floor function to the given {@link Tensor} object,
     * rounding each element down to the nearest integer less than or equal to that element,
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the floor-transformed tensor.
     * @see #floor(String, String)
     */
    default Tensor floorI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return floor(aName, oName).get(oName);
    }


    /**
     * **Not — Basic logical negation with empty tensor reference**
     *
     * Performs element-wise logical negation on a tensor already stored in the internal queue.
     * <p>
     * Each element is inverted such that 0 becomes 1, and nonzero values become 0.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the logical NOT operation.
     * @see #not(String, String)
     */
    default CuBridge not() {
        return not("", genRandomNameUnary());
    }

    /**
     * **Not — Logical negation on a named tensor**
     *
     * Performs element-wise logical negation on the specified tensor.
     * <p>
     * Each element is inverted such that 0 becomes 1, and nonzero values become 0.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the logical NOT operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge not(String a, String out) {
        if (CuBridgeJNI.not(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | not | " + a + " | " + out);
        return null;
    }

    /**
     * **Not — Logical negation on a Tensor object**
     *
     * Performs element-wise logical negation on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #not(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the logical NOT operation.
     * @see #not(String, String)
     */
    default CuBridge not(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return not(aName, out);
    }

    /**
     * **NotI — Immediate logical negation with default tensor reference**
     *
     * Performs element-wise logical negation on a tensor already stored in the queue
     * and immediately returns the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} representing the logically inverted tensor.
     * @see #not(String, String)
     */
    default Tensor notI() {
        String oName = genRandomNameUnary();
        return not("", oName).get(oName);
    }

    /**
     * **NotI — Immediate logical negation of a named tensor**
     *
     * Performs element-wise logical negation on the specified named tensor
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} representing the logically inverted tensor.
     * @see #not(String, String)
     */
    default Tensor notI(String a) {
        String oName = genRandomNameUnary();
        return not(a, oName).get(oName);
    }

    /**
     * **NotI — Immediate logical negation of a Tensor object**
     *
     * Performs element-wise logical negation on the given {@link Tensor} object
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} representing the logically inverted tensor.
     * @see #not(String, String)
     */
    default Tensor notI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return not(aName, oName).get(oName);
    }


    /**
     * **Deg2Rad — Basic degree-to-radian conversion with empty tensor reference**
     *
     * Converts each element of a tensor from degrees to radians.
     * <p>
     * rad = deg × π / 180
     * </p>
     *
     * @return A {@link CuBridge} instance representing the degree-to-radian conversion operation.
     * @see #deg2rad(String, String)
     */
    default CuBridge deg2rad() {
        return deg2rad("", genRandomNameUnary());
    }

    /**
     * **Deg2Rad — Conversion of a named tensor from degrees to radians**
     *
     * Converts each element of the specified tensor from degrees to radians.
     * <p>
     * rad = deg × π / 180
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the conversion operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge deg2rad(String a, String out) {
        if (CuBridgeJNI.deg2rad(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | deg2rad | " + a + " | " + out);
        return null;
    }

    /**
     * **Deg2Rad — Conversion of a Tensor object from degrees to radians**
     *
     * Converts each element of the given {@link Tensor} from degrees to radians.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #deg2rad(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the conversion operation.
     * @see #deg2rad(String, String)
     */
    default CuBridge deg2rad(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return deg2rad(aName, out);
    }

    /**
     * **Deg2RadI — Immediate degree-to-radian conversion with default tensor reference**
     *
     * Converts each element of a tensor from degrees to radians and immediately returns
     * the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} with values converted to radians.
     * @see #deg2rad(String, String)
     */
    default Tensor deg2radI() {
        String oName = genRandomNameUnary();
        return deg2rad("", oName).get(oName);
    }

    /**
     * **Deg2RadI — Immediate conversion of a named tensor from degrees to radians**
     *
     * Converts each element of the specified named tensor from degrees to radians
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} with values converted to radians.
     * @see #deg2rad(String, String)
     */
    default Tensor deg2radI(String a) {
        String oName = genRandomNameUnary();
        return deg2rad(a, oName).get(oName);
    }

    /**
     * **Deg2RadI — Immediate conversion of a Tensor object from degrees to radians**
     *
     * Converts each element of the given {@link Tensor} from degrees to radians
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} with values converted to radians.
     * @see #deg2rad(String, String)
     */
    default Tensor deg2radI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return deg2rad(aName, oName).get(oName);
    }


    /**
     * **Rad2Deg — Basic radian-to-degree conversion with empty tensor reference**
     *
     * Converts each element of a tensor from radians to degrees.
     * <p>
     * deg = rad × 180 / π
     * </p>
     *
     * @return A {@link CuBridge} instance representing the radian-to-degree conversion operation.
     * @see #rad2deg(String, String)
     */
    default CuBridge rad2deg() {
        return rad2deg("", genRandomNameUnary());
    }

    /**
     * **Rad2Deg — Conversion of a named tensor from radians to degrees**
     *
     * Converts each element of the specified tensor from radians to degrees.
     * <p>
     * deg = rad × 180 / π
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name under which the result will be stored.
     * @return A {@link CuBridge} instance representing the conversion operation,
     *         or {@code null} if the operation failed.
     */
    default CuBridge rad2deg(String a, String out) {
        if (CuBridgeJNI.rad2deg(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | rad2deg | " + a + " | " + out);
        return null;
    }

    /**
     * **Rad2Deg — Conversion of a Tensor object from radians to degrees**
     *
     * Converts each element of the given {@link Tensor} from radians to degrees.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #rad2deg(String, String)}.
     * </p>
     *
     * @param a   The input tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the conversion operation.
     * @see #rad2deg(String, String)
     */
    default CuBridge rad2deg(Tensor a, String out) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        return rad2deg(aName, out);
    }

    /**
     * **Rad2DegI — Immediate radian-to-degree conversion with default tensor reference**
     *
     * Converts each element of a tensor from radians to degrees and immediately returns
     * the resulting {@link Tensor}.
     *
     * @return A new {@link Tensor} with values converted to degrees.
     * @see #rad2deg(String, String)
     */
    default Tensor rad2degI() {
        String oName = genRandomNameUnary();
        return rad2deg("", oName).get(oName);
    }

    /**
     * **Rad2DegI — Immediate conversion of a named tensor from radians to degrees**
     *
     * Converts each element of the specified named tensor from radians to degrees
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A new {@link Tensor} with values converted to degrees.
     * @see #rad2deg(String, String)
     */
    default Tensor rad2degI(String a) {
        String oName = genRandomNameUnary();
        return rad2deg(a, oName).get(oName);
    }

    /**
     * **Rad2DegI — Immediate conversion of a Tensor object from radians to degrees**
     *
     * Converts each element of the given {@link Tensor} from radians to degrees
     * and immediately returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A new {@link Tensor} with values converted to degrees.
     * @see #rad2deg(String, String)
     */
    default Tensor rad2degI(Tensor a) {
        String aName = genRandomNameUnary();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameUnary();
        return rad2deg(aName, oName).get(oName);
    }

}
