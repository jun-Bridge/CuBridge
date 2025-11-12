package CuBridge;

import java.util.UUID;

public interface BinaryOps {

    private String genRandomNameBinary() {
        return "BinaryOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }


    /**
     * **Add — Basic element-wise addition**
     *
     * Performs an element-wise addition between the two most recent tensors stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name to store the resulting tensor.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the addition.
     * @see #add(String, String, String)
     */
    default CuBridge add() {
        return add("", "", genRandomNameBinary());
    }

    /**
     * **Add — Element-wise addition between two named tensors**
     *
     * Performs an element-wise addition between tensors {@code a} and {@code b},
     * storing the result in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor.
     * @param b   The name of the second tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the addition.
     */
    default CuBridge add(String a, String b, String out) {
        if (CuBridgeJNI.add(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | add | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Add — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise addition between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #add(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the addition.
     * @see #add(String, String, String)
     */
    default CuBridge add(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return add(aName, b, out);
    }

    /**
     * **Add — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise addition between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #add(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the addition.
     * @see #add(String, String, String)
     */
    default CuBridge add(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return add(a, bName, out);
    }

    /**
     * **Add — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise addition between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #add(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the addition.
     * @see #add(String, String, String)
     */
    default CuBridge add(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return add(aName, bName, out);
    }

    /**
     * **AddI — Immediate element-wise addition**
     *
     * Immediately performs an element-wise addition between the two most recent tensors
     * in the internal queue, and returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name for the output tensor.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the addition.
     * @see #add(String, String, String)
     */
    default Tensor addI() {
        String oName = genRandomNameBinary();
        return add("", "", oName).get(oName);
    }

    /**
     * **AddI — Immediate element-wise addition between two named tensors**
     *
     * Performs an element-wise addition between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #add(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the addition.
     * @see #add(String, String, String)
     */
    default Tensor addI(String a, String b) {
        String oName = genRandomNameBinary();
        return add(a, b, oName).get(oName);
    }

    /**
     * **AddI — Immediate element-wise addition with a Tensor and a named operand**
     *
     * Performs an element-wise addition between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #add(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the addition.
     * @see #add(String, String, String)
     */
    default Tensor addI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return add(aName, b, oName).get(oName);
    }

    /**
     * **AddI — Immediate element-wise addition with a named and a Tensor operand**
     *
     * Performs an element-wise addition between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #add(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the addition.
     * @see #add(String, String, String)
     */
    default Tensor addI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return add(a, bName, oName).get(oName);
    }

    /**
     * **AddI — Immediate element-wise addition between two Tensor objects**
     *
     * Performs an element-wise addition between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #add(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing the element-wise sum.
     * @see #add(String, String, String)
     */
    default Tensor addI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return add(aName, bName, oName).get(oName);
    }


    /**
     * **Sub — Basic element-wise subtraction**
     *
     * Performs an element-wise subtraction between the two most recent tensors stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name to store the resulting tensor.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default CuBridge sub() {
        return sub("", "", genRandomNameBinary());
    }

    /**
     * **Sub — Element-wise subtraction between two named tensors**
     *
     * Performs an element-wise subtraction between tensors {@code a} and {@code b},
     * storing the result in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (minuend).
     * @param b   The name of the second tensor (subtrahend).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the subtraction.
     */
    default CuBridge sub(String a, String b, String out) {
        if (CuBridgeJNI.sub(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | sub | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Sub — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise subtraction between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #sub(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object (minuend).
     * @param b   The name of the right operand tensor (already stored in the queue, subtrahend).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default CuBridge sub(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return sub(aName, b, out);
    }

    /**
     * **Sub — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise subtraction between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #sub(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue, minuend).
     * @param b   The right operand tensor object (subtrahend).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default CuBridge sub(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return sub(a, bName, out);
    }

    /**
     * **Sub — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise subtraction between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #sub(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object (minuend).
     * @param b   The right operand tensor object (subtrahend).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default CuBridge sub(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return sub(aName, bName, out);
    }

    /**
     * **SubI — Immediate element-wise subtraction**
     *
     * Immediately performs an element-wise subtraction between the two most recent tensors
     * in the internal queue, and returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name for the output tensor.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default Tensor subI() {
        String oName = genRandomNameBinary();
        return sub("", "", oName).get(oName);
    }

    /**
     * **SubI — Immediate element-wise subtraction between two named tensors**
     *
     * Performs an element-wise subtraction between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #sub(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor (minuend).
     * @param b The name of the second input tensor (subtrahend).
     * @return A {@link Tensor} containing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default Tensor subI(String a, String b) {
        String oName = genRandomNameBinary();
        return sub(a, b, oName).get(oName);
    }

    /**
     * **SubI — Immediate element-wise subtraction with a Tensor and a named operand**
     *
     * Performs an element-wise subtraction between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #sub(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object (minuend).
     * @param b The name of the right operand tensor (already stored in the queue, subtrahend).
     * @return A {@link Tensor} containing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default Tensor subI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return sub(aName, b, oName).get(oName);
    }

    /**
     * **SubI — Immediate element-wise subtraction with a named and a Tensor operand**
     *
     * Performs an element-wise subtraction between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #sub(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue, minuend).
     * @param b The right operand tensor object (subtrahend).
     * @return A {@link Tensor} containing the result of the subtraction.
     * @see #sub(String, String, String)
     */
    default Tensor subI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return sub(a, bName, oName).get(oName);
    }

    /**
     * **SubI — Immediate element-wise subtraction between two Tensor objects**
     *
     * Performs an element-wise subtraction between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #sub(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor (minuend).
     * @param b The right input tensor (subtrahend).
     * @return A new {@link Tensor} containing the element-wise difference.
     * @see #sub(String, String, String)
     */
    default Tensor subI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return sub(aName, bName, oName).get(oName);
    }


    /**
     * **Mul — Basic element-wise multiplication**
     *
     * Performs an element-wise multiplication between the two most recent tensors stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name to store the resulting tensor.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default CuBridge mul() {
        return mul("", "", genRandomNameBinary());
    }

    /**
     * **Mul — Element-wise multiplication between two named tensors**
     *
     * Performs an element-wise multiplication between tensors {@code a} and {@code b},
     * storing the result in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor.
     * @param b   The name of the second tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the multiplication.
     */
    default CuBridge mul(String a, String b, String out) {
        if (CuBridgeJNI.mul(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | mul | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Mul — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise multiplication between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #mul(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default CuBridge mul(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return mul(aName, b, out);
    }

    /**
     * **Mul — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise multiplication between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #mul(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default CuBridge mul(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return mul(a, bName, out);
    }

    /**
     * **Mul — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise multiplication between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #mul(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default CuBridge mul(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return mul(aName, bName, out);
    }

    /**
     * **MulI — Immediate element-wise multiplication**
     *
     * Immediately performs an element-wise multiplication between the two most recent tensors
     * in the internal queue, and returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name for the output tensor.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default Tensor mulI() {
        String oName = genRandomNameBinary();
        return mul("", "", oName).get(oName);
    }

    /**
     * **MulI — Immediate element-wise multiplication between two named tensors**
     *
     * Performs an element-wise multiplication between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #mul(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default Tensor mulI(String a, String b) {
        String oName = genRandomNameBinary();
        return mul(a, b, oName).get(oName);
    }

    /**
     * **MulI — Immediate element-wise multiplication with a Tensor and a named operand**
     *
     * Performs an element-wise multiplication between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #mul(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default Tensor mulI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return mul(aName, b, oName).get(oName);
    }

    /**
     * **MulI — Immediate element-wise multiplication with a named and a Tensor operand**
     *
     * Performs an element-wise multiplication between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #mul(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the multiplication.
     * @see #mul(String, String, String)
     */
    default Tensor mulI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return mul(a, bName, oName).get(oName);
    }

    /**
     * **MulI — Immediate element-wise multiplication between two Tensor objects**
     *
     * Performs an element-wise multiplication between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #mul(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing the element-wise product.
     * @see #mul(String, String, String)
     */
    default Tensor mulI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return mul(aName, bName, oName).get(oName);
    }


    /**
     * **Div — Basic element-wise division**
     *
     * Performs an element-wise division between the two most recent tensors stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name to store the resulting tensor.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the division.
     * @see #div(String, String, String)
     */
    default CuBridge div() {
        return div("", "", genRandomNameBinary());
    }

    /**
     * **Div — Element-wise division between two named tensors**
     *
     * Performs an element-wise division between tensors {@code a} and {@code b},
     * storing the result in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the numerator tensor.
     * @param b   The name of the denominator tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the division.
     */
    default CuBridge div(String a, String b, String out) {
        if (CuBridgeJNI.div(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | div | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Div — Overload using a Tensor object as the numerator**
     *
     * Performs an element-wise division between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #div(String, String, String)}.
     * </p>
     *
     * @param a   The numerator tensor object.
     * @param b   The name of the denominator tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the division.
     * @see #div(String, String, String)
     */
    default CuBridge div(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return div(aName, b, out);
    }

    /**
     * **Div — Overload using a Tensor object as the denominator**
     *
     * Performs an element-wise division between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #div(String, String, String)}.
     * </p>
     *
     * @param a   The name of the numerator tensor.
     * @param b   The denominator tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the division.
     * @see #div(String, String, String)
     */
    default CuBridge div(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return div(a, bName, out);
    }

    /**
     * **Div — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise division between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #div(String, String, String)} for execution.
     * </p>
     *
     * @param a   The numerator tensor object.
     * @param b   The denominator tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the division.
     * @see #div(String, String, String)
     */
    default CuBridge div(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return div(aName, bName, out);
    }

    /**
     * **DivI — Immediate element-wise division**
     *
     * Immediately performs an element-wise division between the two most recent tensors
     * in the internal queue, and returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name for the output tensor.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the division.
     * @see #div(String, String, String)
     */
    default Tensor divI() {
        String oName = genRandomNameBinary();
        return div("", "", oName).get(oName);
    }

    /**
     * **DivI — Immediate element-wise division between two named tensors**
     *
     * Performs an element-wise division between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #div(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the numerator tensor.
     * @param b The name of the denominator tensor.
     * @return A {@link Tensor} containing the result of the division.
     * @see #div(String, String, String)
     */
    default Tensor divI(String a, String b) {
        String oName = genRandomNameBinary();
        return div(a, b, oName).get(oName);
    }

    /**
     * **DivI — Immediate element-wise division with a Tensor and a named operand**
     *
     * Performs an element-wise division between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #div(String, String, String)}.
     * </p>
     *
     * @param a The numerator tensor object.
     * @param b The name of the denominator tensor.
     * @return A {@link Tensor} containing the result of the division.
     * @see #div(String, String, String)
     */
    default Tensor divI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return div(aName, b, oName).get(oName);
    }

    /**
     * **DivI — Immediate element-wise division with a named and a Tensor operand**
     *
     * Performs an element-wise division between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #div(String, String, String)}.
     * </p>
     *
     * @param a The name of the numerator tensor.
     * @param b The denominator tensor object.
     * @return A {@link Tensor} containing the result of the division.
     * @see #div(String, String, String)
     */
    default Tensor divI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return div(a, bName, oName).get(oName);
    }

    /**
     * **DivI — Immediate element-wise division between two Tensor objects**
     *
     * Performs an element-wise division between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #div(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The numerator tensor.
     * @param b The denominator tensor.
     * @return A new {@link Tensor} containing the element-wise quotient.
     * @see #div(String, String, String)
     */
    default Tensor divI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return div(aName, bName, oName).get(oName);
    }


    /**
     * **Pow — Basic element-wise exponentiation**
     *
     * Performs an element-wise power operation between the two most recent tensors stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name to store the resulting tensor.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default CuBridge pow() {
        return pow("", "", genRandomNameBinary());
    }

    /**
     * **Pow — Element-wise exponentiation between two named tensors**
     *
     * Performs an element-wise exponentiation between tensors {@code a} and {@code b},
     * computing {@code a^b} element by element and storing the result in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the base tensor.
     * @param b   The name of the exponent tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the exponentiation.
     */
    default CuBridge pow(String a, String b, String out) {
        if (CuBridgeJNI.pow(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | pow | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Pow — Overload using a Tensor object as the base**
     *
     * Performs an element-wise exponentiation between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #pow(String, String, String)}.
     * </p>
     *
     * @param a   The base tensor object.
     * @param b   The name of the exponent tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default CuBridge pow(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return pow(aName, b, out);
    }

    /**
     * **Pow — Overload using a Tensor object as the exponent**
     *
     * Performs an element-wise exponentiation between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #pow(String, String, String)}.
     * </p>
     *
     * @param a   The name of the base tensor.
     * @param b   The exponent tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default CuBridge pow(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return pow(a, bName, out);
    }

    /**
     * **Pow — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise exponentiation between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #pow(String, String, String)} for execution.
     * </p>
     *
     * @param a   The base tensor object.
     * @param b   The exponent tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default CuBridge pow(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return pow(aName, bName, out);
    }

    /**
     * **PowI — Immediate element-wise exponentiation**
     *
     * Immediately performs an element-wise exponentiation between the two most recent tensors
     * in the internal queue, and returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name for the output tensor.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default Tensor powI() {
        String oName = genRandomNameBinary();
        return pow("", "", oName).get(oName);
    }

    /**
     * **PowI — Immediate element-wise exponentiation between two named tensors**
     *
     * Performs an element-wise exponentiation between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #pow(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the base tensor.
     * @param b The name of the exponent tensor.
     * @return A {@link Tensor} containing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default Tensor powI(String a, String b) {
        String oName = genRandomNameBinary();
        return pow(a, b, oName).get(oName);
    }

    /**
     * **PowI — Immediate element-wise exponentiation with a Tensor and a named operand**
     *
     * Performs an element-wise exponentiation between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the base tensor
     * before executing {@link #pow(String, String, String)}.
     * </p>
     *
     * @param a The base tensor object.
     * @param b The name of the exponent tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default Tensor powI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return pow(aName, b, oName).get(oName);
    }

    /**
     * **PowI — Immediate element-wise exponentiation with a named and a Tensor operand**
     *
     * Performs an element-wise exponentiation between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the exponent tensor
     * before executing {@link #pow(String, String, String)}.
     * </p>
     *
     * @param a The name of the base tensor.
     * @param b The exponent tensor object.
     * @return A {@link Tensor} containing the result of the exponentiation.
     * @see #pow(String, String, String)
     */
    default Tensor powI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return pow(a, bName, oName).get(oName);
    }

    /**
     * **PowI — Immediate element-wise exponentiation between two Tensor objects**
     *
     * Performs an element-wise exponentiation between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #pow(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The base tensor object.
     * @param b The exponent tensor object.
     * @return A new {@link Tensor} containing the element-wise power results.
     * @see #pow(String, String, String)
     */
    default Tensor powI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return pow(aName, bName, oName).get(oName);
    }


    /**
     * **Mod — Basic element-wise modulo**
     *
     * Performs an element-wise modulo (remainder) operation between the two most recent tensors
     * stored in the internal queue.
     * <p>
     * Each element of the result is computed as {@code a[i] % b[i]}.
     * Automatically assigns a random internal name to store the resulting tensor.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the modulo operation.
     * @see #mod(String, String, String)
     */
    default CuBridge mod() {
        return mod("", "", genRandomNameBinary());
    }

    /**
     * **Mod — Element-wise modulo between two named tensors**
     *
     * Performs an element-wise modulo between tensors {@code a} and {@code b},
     * computing {@code a % b} element by element and storing the result in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the dividend tensor.
     * @param b   The name of the divisor tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the modulo operation.
     */
    default CuBridge mod(String a, String b, String out) {
        if (CuBridgeJNI.mod(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | mod | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Mod — Overload using a Tensor object as the dividend**
     *
     * Performs an element-wise modulo between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #mod(String, String, String)}.
     * </p>
     *
     * @param a   The dividend tensor object.
     * @param b   The name of the divisor tensor.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the modulo operation.
     */
    default CuBridge mod(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return mod(aName, b, out);
    }

    /**
     * **Mod — Overload using a Tensor object as the divisor**
     *
     * Performs an element-wise modulo between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #mod(String, String, String)}.
     * </p>
     *
     * @param a   The name of the dividend tensor.
     * @param b   The divisor tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the modulo operation.
     */
    default CuBridge mod(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return mod(a, bName, out);
    }

    /**
     * **Mod — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise modulo between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #mod(String, String, String)} for execution.
     * </p>
     *
     * @param a   The dividend tensor object.
     * @param b   The divisor tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the modulo operation.
     */
    default CuBridge mod(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return mod(aName, bName, out);
    }

    /**
     * **ModI — Immediate element-wise modulo**
     *
     * Immediately performs an element-wise modulo between the two most recent tensors
     * in the internal queue, and returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name for the output tensor.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the modulo operation.
     */
    default Tensor modI() {
        String oName = genRandomNameBinary();
        return mod("", "", oName).get(oName);
    }

    /**
     * **ModI — Immediate element-wise modulo between two named tensors**
     *
     * Performs an element-wise modulo between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #mod(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the dividend tensor.
     * @param b The name of the divisor tensor.
     * @return A {@link Tensor} containing the result of the modulo operation.
     */
    default Tensor modI(String a, String b) {
        String oName = genRandomNameBinary();
        return mod(a, b, oName).get(oName);
    }

    /**
     * **ModI — Immediate element-wise modulo with a Tensor and a named operand**
     *
     * Performs an element-wise modulo between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #mod(String, String, String)}.
     * </p>
     *
     * @param a The dividend tensor object.
     * @param b The name of the divisor tensor.
     * @return A {@link Tensor} containing the result of the modulo operation.
     */
    default Tensor modI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return mod(aName, b, oName).get(oName);
    }

    /**
     * **ModI — Immediate element-wise modulo with a named and a Tensor operand**
     *
     * Performs an element-wise modulo between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #mod(String, String, String)}.
     * </p>
     *
     * @param a The name of the dividend tensor.
     * @param b The divisor tensor object.
     * @return A {@link Tensor} containing the result of the modulo operation.
     */
    default Tensor modI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return mod(a, bName, oName).get(oName);
    }

    /**
     * **ModI — Immediate element-wise modulo between two Tensor objects**
     *
     * Performs an element-wise modulo between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #mod(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The dividend tensor.
     * @param b The divisor tensor.
     * @return A new {@link Tensor} containing the element-wise remainder.
     */
    default Tensor modI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return mod(aName, bName, oName).get(oName);
    }


    /**
     * **Gt — Basic element-wise greater-than comparison**
     *
     * Performs an element-wise comparison between the two most recent tensors in the queue,
     * evaluating {@code a > b} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where the condition is true,
     * and {@code 0.0} where it is false.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #gt(String, String, String)
     */
    default CuBridge gt() {
        return gt("", "", genRandomNameBinary());
    }

    /**
     * **Gt — Element-wise greater-than comparison between two named tensors**
     *
     * Performs an element-wise greater-than comparison between tensors {@code a} and {@code b},
     * computing {@code (a > b)} for each element and storing the boolean results (1 or 0) in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     */
    default CuBridge gt(String a, String b, String out) {
        if (CuBridgeJNI.gt(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | gt | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Gt — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise greater-than comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #gt(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     */
    default CuBridge gt(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return gt(aName, b, out);
    }

    /**
     * **Gt — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise greater-than comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #gt(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     */
    default CuBridge gt(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return gt(a, bName, out);
    }

    /**
     * **Gt — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise greater-than comparison between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #gt(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     */
    default CuBridge gt(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return gt(aName, bName, out);
    }

    /**
     * **GtI — Immediate element-wise greater-than comparison**
     *
     * Immediately performs an element-wise greater-than comparison between the two most recent tensors
     * in the internal queue and returns the resulting {@link Tensor}.
     * <p>
     * Each element in the output tensor is {@code 1.0} if {@code a > b}, otherwise {@code 0.0}.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the comparison.
     */
    default Tensor gtI() {
        String oName = genRandomNameBinary();
        return gt("", "", oName).get(oName);
    }

    /**
     * **GtI — Immediate element-wise greater-than comparison between two named tensors**
     *
     * Performs an element-wise greater-than comparison between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #gt(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the comparison.
     */
    default Tensor gtI(String a, String b) {
        String oName = genRandomNameBinary();
        return gt(a, b, oName).get(oName);
    }

    /**
     * **GtI — Immediate element-wise greater-than comparison with a Tensor and a named operand**
     *
     * Performs an element-wise greater-than comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #gt(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the comparison.
     */
    default Tensor gtI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return gt(aName, b, oName).get(oName);
    }

    /**
     * **GtI — Immediate element-wise greater-than comparison with a named and a Tensor operand**
     *
     * Performs an element-wise greater-than comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #gt(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the comparison.
     */
    default Tensor gtI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return gt(a, bName, oName).get(oName);
    }

    /**
     * **GtI — Immediate element-wise greater-than comparison between two Tensor objects**
     *
     * Performs an element-wise greater-than comparison between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #gt(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where {@code a > b}, otherwise 0.0.
     */
    default Tensor gtI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return gt(aName, bName, oName).get(oName);
    }


    /**
     * **Lt — Basic element-wise less-than comparison**
     *
     * Performs an element-wise comparison between the two most recent tensors stored in the internal queue,
     * evaluating {@code a < b} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where the condition is true
     * and {@code 0.0} where it is false.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default CuBridge lt() {
        return lt("", "", genRandomNameBinary());
    }

    /**
     * **Lt — Element-wise less-than comparison between two named tensors**
     *
     * Performs an element-wise less-than comparison between tensors {@code a} and {@code b},
     * computing {@code (a < b)} for each element and storing the boolean results (1 or 0) in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default CuBridge lt(String a, String b, String out) {
        if (CuBridgeJNI.lt(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | lt | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Lt — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise less-than comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #lt(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default CuBridge lt(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return lt(aName, b, out);
    }

    /**
     * **Lt — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise less-than comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #lt(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default CuBridge lt(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return lt(a, bName, out);
    }

    /**
     * **Lt — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise less-than comparison between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #lt(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default CuBridge lt(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return lt(aName, bName, out);
    }

    /**
     * **LtI — Immediate element-wise less-than comparison with empty tensor references**
     *
     * Performs an element-wise less-than comparison using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default Tensor ltI() {
        String oName = genRandomNameBinary();
        return lt("", "", oName).get(oName);
    }

    /**
     * **LtI — Immediate element-wise less-than comparison between two named tensors**
     *
     * Performs an element-wise less-than comparison between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #lt(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default Tensor ltI(String a, String b) {
        String oName = genRandomNameBinary();
        return lt(a, b, oName).get(oName);
    }

    /**
     * **LtI — Immediate element-wise less-than comparison with a Tensor and a named operand**
     *
     * Performs an element-wise less-than comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #lt(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default Tensor ltI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return lt(aName, b, oName).get(oName);
    }

    /**
     * **LtI — Immediate element-wise less-than comparison with a named and a Tensor operand**
     *
     * Performs an element-wise less-than comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #lt(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #lt(String, String, String)
     */
    default Tensor ltI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return lt(a, bName, oName).get(oName);
    }

    /**
     * **LtI — Immediate element-wise less-than comparison between two Tensor objects**
     *
     * Performs an element-wise less-than comparison between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #lt(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where {@code a < b}, otherwise 0.0.
     * @see #lt(String, String, String)
     */
    default Tensor ltI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return lt(aName, bName, oName).get(oName);
    }


    /**
     * **Ge — Basic element-wise greater-than-or-equal comparison**
     *
     * Performs an element-wise comparison between the two most recent tensors stored in the internal queue,
     * evaluating {@code a >= b} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where the condition is true
     * and {@code 0.0} where it is false.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default CuBridge ge() {
        return ge("", "", genRandomNameBinary());
    }

    /**
     * **Ge — Element-wise greater-than-or-equal comparison between two named tensors**
     *
     * Performs an element-wise greater-than-or-equal comparison between tensors {@code a} and {@code b},
     * computing {@code (a >= b)} for each element and storing the boolean results (1 or 0) in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default CuBridge ge(String a, String b, String out) {
        if (CuBridgeJNI.ge(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | ge | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Ge — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise greater-than-or-equal comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #ge(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default CuBridge ge(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return ge(aName, b, out);
    }

    /**
     * **Ge — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise greater-than-or-equal comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #ge(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default CuBridge ge(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return ge(a, bName, out);
    }

    /**
     * **Ge — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise greater-than-or-equal comparison between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #ge(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default CuBridge ge(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return ge(aName, bName, out);
    }

    /**
     * **GeI — Immediate element-wise greater-than-or-equal comparison with empty tensor references**
     *
     * Performs an element-wise greater-than-or-equal comparison using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default Tensor geI() {
        String oName = genRandomNameBinary();
        return ge("", "", oName).get(oName);
    }

    /**
     * **GeI — Immediate element-wise greater-than-or-equal comparison between two named tensors**
     *
     * Performs an element-wise greater-than-or-equal comparison between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #ge(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default Tensor geI(String a, String b) {
        String oName = genRandomNameBinary();
        return ge(a, b, oName).get(oName);
    }

    /**
     * **GeI — Immediate element-wise greater-than-or-equal comparison with a Tensor and a named operand**
     *
     * Performs an element-wise greater-than-or-equal comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #ge(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default Tensor geI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return ge(aName, b, oName).get(oName);
    }

    /**
     * **GeI — Immediate element-wise greater-than-or-equal comparison with a named and a Tensor operand**
     *
     * Performs an element-wise greater-than-or-equal comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #ge(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #ge(String, String, String)
     */
    default Tensor geI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return ge(a, bName, oName).get(oName);
    }

    /**
     * **GeI — Immediate element-wise greater-than-or-equal comparison between two Tensor objects**
     *
     * Performs an element-wise greater-than-or-equal comparison between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #ge(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where {@code a >= b}, otherwise 0.0.
     * @see #ge(String, String, String)
     */
    default Tensor geI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return ge(aName, bName, oName).get(oName);
    }


    /**
     * **Le — Basic element-wise less-than-or-equal comparison**
     *
     * Performs an element-wise comparison between the two most recent tensors stored in the internal queue,
     * evaluating {@code a <= b} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where the condition is true
     * and {@code 0.0} where it is false.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #le(String, String, String)
     */
    default CuBridge le() {
        return le("", "", genRandomNameBinary());
    }

    /**
     * **Le — Element-wise less-than-or-equal comparison between two named tensors**
     *
     * Performs an element-wise less-than-or-equal comparison between tensors {@code a} and {@code b},
     * computing {@code (a <= b)} for each element and storing the boolean results (1 or 0) in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #le(String, String, String)
     */
    default CuBridge le(String a, String b, String out) {
        if (CuBridgeJNI.le(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | le | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Le — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise less-than-or-equal comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #le(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #le(String, String, String)
     */
    default CuBridge le(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return le(aName, b, out);
    }

    /**
     * **Le — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise less-than-or-equal comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #le(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #le(String, String, String)
     */
    default CuBridge le(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return le(a, bName, out);
    }

    /**
     * **Le — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise less-than-or-equal comparison between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #le(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #le(String, String, String)
     */
    default CuBridge le(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return le(aName, bName, out);
    }

    /**
     * **LeI — Immediate element-wise less-than-or-equal comparison with empty tensor references**
     *
     * Performs an element-wise less-than-or-equal comparison using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the comparison.
     * @see #le(String, String, String)
     */
    default Tensor leI() {
        String oName = genRandomNameBinary();
        return le("", "", oName).get(oName);
    }

    /**
     * **LeI — Immediate element-wise less-than-or-equal comparison between two named tensors**
     *
     * Performs an element-wise less-than-or-equal comparison between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #le(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #le(String, String, String)
     */
    default Tensor leI(String a, String b) {
        String oName = genRandomNameBinary();
        return le(a, b, oName).get(oName);
    }

    /**
     * **LeI — Immediate element-wise less-than-or-equal comparison with a Tensor and a named operand**
     *
     * Performs an element-wise less-than-or-equal comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #le(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #le(String, String, String)
     */
    default Tensor leI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return le(aName, b, oName).get(oName);
    }

    /**
     * **LeI — Immediate element-wise less-than-or-equal comparison with a named and a Tensor operand**
     *
     * Performs an element-wise less-than-or-equal comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #le(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #le(String, String, String)
     */
    default Tensor leI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return le(a, bName, oName).get(oName);
    }

    /**
     * **LeI — Immediate element-wise less-than-or-equal comparison between two Tensor objects**
     *
     * Performs an element-wise less-than-or-equal comparison between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #le(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where {@code a <= b}, otherwise 0.0.
     * @see #le(String, String, String)
     */
    default Tensor leI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return le(aName, bName, oName).get(oName);
    }


    /**
     * **Eq — Basic element-wise equality comparison**
     *
     * Performs an element-wise comparison between the two most recent tensors stored in the internal queue,
     * evaluating {@code a == b} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where the condition is true
     * and {@code 0.0} where it is false.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default CuBridge eq() {
        return eq("", "", genRandomNameBinary());
    }

    /**
     * **Eq — Element-wise equality comparison between two named tensors**
     *
     * Performs an element-wise equality comparison between tensors {@code a} and {@code b},
     * computing {@code (a == b)} for each element and storing the boolean results (1 or 0) in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default CuBridge eq(String a, String b, String out) {
        if (CuBridgeJNI.eq(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | eq | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Eq — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise equality comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #eq(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default CuBridge eq(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return eq(aName, b, out);
    }

    /**
     * **Eq — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise equality comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #eq(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default CuBridge eq(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return eq(a, bName, out);
    }

    /**
     * **Eq — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise equality comparison between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #eq(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default CuBridge eq(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return eq(aName, bName, out);
    }

    /**
     * **EqI — Immediate element-wise equality comparison with empty tensor references**
     *
     * Performs an element-wise equality comparison using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default Tensor eqI() {
        String oName = genRandomNameBinary();
        return eq("", "", oName).get(oName);
    }

    /**
     * **EqI — Immediate element-wise equality comparison between two named tensors**
     *
     * Performs an element-wise equality comparison between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #eq(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default Tensor eqI(String a, String b) {
        String oName = genRandomNameBinary();
        return eq(a, b, oName).get(oName);
    }

    /**
     * **EqI — Immediate element-wise equality comparison with a Tensor and a named operand**
     *
     * Performs an element-wise equality comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #eq(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default Tensor eqI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return eq(aName, b, oName).get(oName);
    }

    /**
     * **EqI — Immediate element-wise equality comparison with a named and a Tensor operand**
     *
     * Performs an element-wise equality comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #eq(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #eq(String, String, String)
     */
    default Tensor eqI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return eq(a, bName, oName).get(oName);
    }

    /**
     * **EqI — Immediate element-wise equality comparison between two Tensor objects**
     *
     * Performs an element-wise equality comparison between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #eq(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where {@code a == b}, otherwise 0.0.
     * @see #eq(String, String, String)
     */
    default Tensor eqI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return eq(aName, bName, oName).get(oName);
    }


    /**
     * **Ne — Basic element-wise not-equal comparison**
     *
     * Performs an element-wise comparison between the two most recent tensors stored in the internal queue,
     * evaluating {@code a != b} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where the condition is true
     * and {@code 0.0} where it is false.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default CuBridge ne() {
        return ne("", "", genRandomNameBinary());
    }

    /**
     * **Ne — Element-wise not-equal comparison between two named tensors**
     *
     * Performs an element-wise not-equal comparison between tensors {@code a} and {@code b},
     * computing {@code (a != b)} for each element and storing the boolean results (1 or 0) in {@code out}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default CuBridge ne(String a, String b, String out) {
        if (CuBridgeJNI.ne(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | ne | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Ne — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise not-equal comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #ne(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default CuBridge ne(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return ne(aName, b, out);
    }

    /**
     * **Ne — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise not-equal comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #ne(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default CuBridge ne(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return ne(a, bName, out);
    }

    /**
     * **Ne — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise not-equal comparison between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #ne(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default CuBridge ne(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return ne(aName, bName, out);
    }

    /**
     * **NeI — Immediate element-wise not-equal comparison with empty tensor references**
     *
     * Performs an element-wise not-equal comparison using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default Tensor neI() {
        String oName = genRandomNameBinary();
        return ne("", "", oName).get(oName);
    }

    /**
     * **NeI — Immediate element-wise not-equal comparison between two named tensors**
     *
     * Performs an element-wise not-equal comparison between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #ne(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default Tensor neI(String a, String b) {
        String oName = genRandomNameBinary();
        return ne(a, b, oName).get(oName);
    }

    /**
     * **NeI — Immediate element-wise not-equal comparison with a Tensor and a named operand**
     *
     * Performs an element-wise not-equal comparison between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #ne(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default Tensor neI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return ne(aName, b, oName).get(oName);
    }

    /**
     * **NeI — Immediate element-wise not-equal comparison with a named and a Tensor operand**
     *
     * Performs an element-wise not-equal comparison between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #ne(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the comparison.
     * @see #ne(String, String, String)
     */
    default Tensor neI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return ne(a, bName, oName).get(oName);
    }

    /**
     * **NeI — Immediate element-wise not-equal comparison between two Tensor objects**
     *
     * Performs an element-wise not-equal comparison between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #ne(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where {@code a != b}, otherwise 0.0.
     * @see #ne(String, String, String)
     */
    default Tensor neI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return ne(aName, bName, oName).get(oName);
    }


    /**
     * **And — Basic element-wise logical AND operation**
     *
     * Performs an element-wise logical AND operation between the two most recent tensors stored in the internal queue,
     * evaluating {@code (a != 0 && b != 0)} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where both inputs are nonzero,
     * and {@code 0.0} otherwise.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default CuBridge and() {
        return and("", "", genRandomNameBinary());
    }

    /**
     * **And — Element-wise logical AND operation between two named tensors**
     *
     * Performs an element-wise logical AND operation between tensors {@code a} and {@code b},
     * computing {@code (a && b)} element by element and storing {@code 1.0} when both inputs are nonzero, otherwise {@code 0.0}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default CuBridge and(String a, String b, String out) {
        if (CuBridgeJNI.and(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | and | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **And — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise logical AND operation between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #and(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default CuBridge and(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return and(aName, b, out);
    }

    /**
     * **And — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise logical AND operation between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #and(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default CuBridge and(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return and(a, bName, out);
    }

    /**
     * **And — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise logical AND operation between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #and(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default CuBridge and(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return and(aName, bName, out);
    }

    /**
     * **AndI — Immediate element-wise logical AND operation with empty tensor references**
     *
     * Performs an element-wise logical AND operation using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default Tensor andI() {
        String oName = genRandomNameBinary();
        return and("", "", oName).get(oName);
    }

    /**
     * **AndI — Immediate element-wise logical AND operation between two named tensors**
     *
     * Performs an element-wise logical AND operation between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #and(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default Tensor andI(String a, String b) {
        String oName = genRandomNameBinary();
        return and(a, b, oName).get(oName);
    }

    /**
     * **AndI — Immediate element-wise logical AND operation with a Tensor and a named operand**
     *
     * Performs an element-wise logical AND operation between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #and(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default Tensor andI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return and(aName, b, oName).get(oName);
    }

    /**
     * **AndI — Immediate element-wise logical AND operation with a named and a Tensor operand**
     *
     * Performs an element-wise logical AND operation between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #and(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the logical AND operation.
     * @see #and(String, String, String)
     */
    default Tensor andI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return and(a, bName, oName).get(oName);
    }

    /**
     * **AndI — Immediate element-wise logical AND operation between two Tensor objects**
     *
     * Performs an element-wise logical AND operation between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #and(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where both {@code a} and {@code b} are nonzero, otherwise 0.0.
     * @see #and(String, String, String)
     */
    default Tensor andI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return and(aName, bName, oName).get(oName);
    }


    /**
     * **Or — Basic element-wise logical OR operation**
     *
     * Performs an element-wise logical OR operation between the two most recent tensors stored in the internal queue,
     * evaluating {@code (a != 0 || b != 0)} for each corresponding element.
     * <p>
     * The resulting tensor contains {@code 1.0} where at least one of the inputs is nonzero,
     * and {@code 0.0} where both inputs are zero.
     * </p>
     *
     * @return A {@link CuBridge} instance representing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default CuBridge or() {
        return or("", "", genRandomNameBinary());
    }

    /**
     * **Or — Element-wise logical OR operation between two named tensors**
     *
     * Performs an element-wise logical OR operation between tensors {@code a} and {@code b},
     * computing {@code (a || b)} element by element and storing {@code 1.0} when either input is nonzero, otherwise {@code 0.0}.
     * <p>
     * Broadcasting is automatically applied when shapes are compatible.
     * </p>
     *
     * @param a   The name of the first tensor (left operand).
     * @param b   The name of the second tensor (right operand).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default CuBridge or(String a, String b, String out) {
        if (CuBridgeJNI.or(a, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | or | " + a + " | " + b + " | " + out);
        return null;
    }

    /**
     * **Or — Overload using a Tensor object as the first operand**
     *
     * Performs an element-wise logical OR operation between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #or(String, String, String)}.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The name of the right operand tensor (already stored in the queue).
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default CuBridge or(Tensor a, String b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        return or(aName, b, out);
    }

    /**
     * **Or — Overload using a Tensor object as the second operand**
     *
     * Performs an element-wise logical OR operation between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #or(String, String, String)}.
     * </p>
     *
     * @param a   The name of the first operand tensor (already stored in the queue).
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default CuBridge or(String a, Tensor b, String out) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return or(a, bName, out);
    }

    /**
     * **Or — Overload using two Tensor objects as operands**
     *
     * Performs an element-wise logical OR operation between two {@link Tensor} objects directly.
     * <p>
     * Random internal names are automatically generated for both input tensors,
     * which are then passed to {@link #or(String, String, String)} for execution.
     * </p>
     *
     * @param a   The left operand tensor object.
     * @param b   The right operand tensor object.
     * @param out The name to store the resulting tensor.
     * @return A {@link CuBridge} instance representing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default CuBridge or(Tensor a, Tensor b, String out) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        return or(aName, bName, out);
    }

    /**
     * **OrI — Immediate element-wise logical OR operation with empty tensor references**
     *
     * Performs an element-wise logical OR operation using an automatically assigned output name
     * when both input tensors are unspecified (empty names).
     * <p>
     * Typically used when operands already exist in the internal queue.
     * </p>
     *
     * @return A {@link Tensor} representing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default Tensor orI() {
        String oName = genRandomNameBinary();
        return or("", "", oName).get(oName);
    }

    /**
     * **OrI — Immediate element-wise logical OR operation between two named tensors**
     *
     * Performs an element-wise logical OR operation between two tensors that are already stored in the internal queue.
     * <p>
     * Automatically assigns a random internal name for the output tensor,
     * executes the {@link #or(String, String, String)} operation,
     * and retrieves the computed result directly.
     * </p>
     *
     * @param a The name of the first input tensor.
     * @param b The name of the second input tensor.
     * @return A {@link Tensor} containing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default Tensor orI(String a, String b) {
        String oName = genRandomNameBinary();
        return or(a, b, oName).get(oName);
    }

    /**
     * **OrI — Immediate element-wise logical OR operation with a Tensor and a named operand**
     *
     * Performs an element-wise logical OR operation between a {@link Tensor} object and a named tensor.
     * <p>
     * Automatically assigns a random internal name to the first tensor
     * before executing {@link #or(String, String, String)}.
     * </p>
     *
     * @param a The left operand tensor object.
     * @param b The name of the right operand tensor (already stored in the queue).
     * @return A {@link Tensor} containing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default Tensor orI(Tensor a, String b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameBinary();
        return or(aName, b, oName).get(oName);
    }

    /**
     * **OrI — Immediate element-wise logical OR operation with a named and a Tensor operand**
     *
     * Performs an element-wise logical OR operation between a named tensor and a {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the second tensor
     * before executing {@link #or(String, String, String)}.
     * </p>
     *
     * @param a The name of the first operand tensor (already stored in the queue).
     * @param b The right operand tensor object.
     * @return A {@link Tensor} containing the result of the logical OR operation.
     * @see #or(String, String, String)
     */
    default Tensor orI(String a, Tensor b) {
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return or(a, bName, oName).get(oName);
    }

    /**
     * **OrI — Immediate element-wise logical OR operation between two Tensor objects**
     *
     * Performs an element-wise logical OR operation between two input {@link Tensor} objects
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * This method automatically assigns random internal names to the input tensors,
     * executes the {@link #or(String, String, String)} operation,
     * and retrieves the computed output tensor from the bridge queue.
     * </p>
     *
     * @param a The left input tensor.
     * @param b The right input tensor.
     * @return A new {@link Tensor} containing 1.0 where either {@code a} or {@code b} is nonzero, otherwise 0.0.
     * @see #or(String, String, String)
     */
    default Tensor orI(Tensor a, Tensor b) {
        String aName = genRandomNameBinary(); CuBridge.getInstance().put(a, aName);
        String bName = genRandomNameBinary(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameBinary();
        return or(aName, bName, oName).get(oName);
    }

}
