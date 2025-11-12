package CuBridge;

import java.util.UUID;

public interface ImageOps {

    private String genRandomNameImage() {
        return "ImageOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /**
     * **Rotate — Core rotation operation**
     *
     * Rotates the specified tensor (image or matrix) by the given angle (in degrees).
     * <p>
     * The output tensor retains the same dimensions as the input tensor.
     * This function performs a spatial rotation using bilinear interpolation internally.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting rotated tensor.
     * @param angle The rotation angle in degrees. Positive values rotate counterclockwise.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge rotate(String a, String out, float angle) {
        if (CuBridgeJNI.rotate(a, out, angle)) return CuBridge.getInstance();
        else System.err.println("Error | rotate | " + a + " | " + out + " | angle=" + angle);
        return null;
    }

    /**
     * **Rotate — Overload using a Tensor object**
     *
     * Rotates the given {@link Tensor} object by the specified angle (in degrees).
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #rotate(String, String, float)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param out   The name to store the resulting rotated tensor.
     * @param angle The rotation angle in degrees.
     * @return A {@link CuBridge} instance representing the rotation operation.
     * @see #rotate(String, String, float)
     */
    default CuBridge rotate(Tensor a, String out, float angle) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return rotate(aName, out, angle);
    }

    /**
     * **RotateI — Immediate rotation operation on a named tensor**
     *
     * Rotates the named tensor by the specified angle
     * and directly returns the resulting rotated {@link Tensor}.
     *
     * @param a     The name of the input tensor.
     * @param angle The rotation angle in degrees.
     * @return A {@link Tensor} representing the rotated output tensor.
     * @see #rotate(String, String, float)
     */
    default Tensor rotateI(String a, float angle) {
        String oName = genRandomNameImage();
        return rotate(a, oName, angle).get(oName);
    }

    /**
     * **RotateI — Immediate rotation operation on a Tensor object**
     *
     * Rotates the given {@link Tensor} object by the specified angle (in degrees)
     * and directly returns the resulting rotated {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #rotate(String, String, float)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param angle The rotation angle in degrees.
     * @return A {@link Tensor} representing the rotated output tensor.
     * @see #rotate(String, String, float)
     */
    default Tensor rotateI(Tensor a, float angle) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return rotate(aName, oName, angle).get(oName);
    }


    /**
     * **Shift — Core spatial shift operation**
     *
     * Shifts the specified tensor (image or matrix) horizontally and vertically
     * by the given pixel offsets.
     * <p>
     * Positive {@code sW} values move the image to the right,
     * and positive {@code sH} values move the image downward.
     * Areas that move outside the boundary are filled with zeros.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting shifted tensor.
     * @param sW  Horizontal shift (in pixels). Positive → right, Negative → left.
     * @param sH  Vertical shift (in pixels). Positive → down, Negative → up.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge shift(String a, String out, int sW, int sH) {
        if (CuBridgeJNI.shift(a, out, sW, sH)) return CuBridge.getInstance();
        else System.err.println("Error | shift | " + a + " | " + out + " | sW=" + sW + " | sH=" + sH);
        return null;
    }

    /**
     * **Shift — Overload using a Tensor object**
     *
     * Shifts the given {@link Tensor} object by the specified horizontal and vertical offsets.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #shift(String, String, int, int)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting shifted tensor.
     * @param sW  Horizontal shift (in pixels). Positive → right.
     * @param sH  Vertical shift (in pixels). Positive → down.
     * @return A {@link CuBridge} instance representing the shift operation.
     * @see #shift(String, String, int, int)
     */
    default CuBridge shift(Tensor a, String out, int sW, int sH) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return shift(aName, out, sW, sH);
    }

    /**
     * **ShiftI — Immediate spatial shift on a named tensor**
     *
     * Shifts the named tensor by the specified pixel offsets
     * and directly returns the resulting shifted {@link Tensor}.
     *
     * @param a  The name of the input tensor.
     * @param sW Horizontal shift (in pixels). Positive → right.
     * @param sH Vertical shift (in pixels). Positive → down.
     * @return A {@link Tensor} representing the shifted output tensor.
     * @see #shift(String, String, int, int)
     */
    default Tensor shiftI(String a, int sW, int sH) {
        String oName = genRandomNameImage();
        return shift(a, oName, sW, sH).get(oName);
    }

    /**
     * **ShiftI — Immediate spatial shift on a Tensor object**
     *
     * Shifts the given {@link Tensor} object by the specified pixel offsets
     * and directly returns the resulting shifted {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #shift(String, String, int, int)}.
     * </p>
     *
     * @param a  The input tensor object.
     * @param sW Horizontal shift (in pixels). Positive → right.
     * @param sH Vertical shift (in pixels). Positive → down.
     * @return A {@link Tensor} representing the shifted output tensor.
     * @see #shift(String, String, int, int)
     */
    default Tensor shiftI(Tensor a, int sW, int sH) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return shift(aName, oName, sW, sH).get(oName);
    }


    /**
     * **Translate — Core modular shift operation**
     *
     * Translates (cyclically shifts) the specified tensor (image or matrix)
     * horizontally and vertically by the given offsets using modular arithmetic.
     * <p>
     * Unlike {@link #shift(String, String, int, int)}, CuBridge.getInstance() operation wraps around:
     * pixels that move outside one edge reappear on the opposite side,
     * effectively performing a circular shift.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting translated tensor.
     * @param tW  Horizontal translation (in pixels). Positive → right, Negative → left.
     * @param tH  Vertical translation (in pixels). Positive → down, Negative → up.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge translate(String a, String out, int tW, int tH) {
        if (CuBridgeJNI.translate(a, out, tW, tH)) return CuBridge.getInstance();
        else System.err.println("Error | translate | " + a + " | " + out + " | tW=" + tW + " | tH=" + tH);
        return null;
    }

    /**
     * **Translate — Overload using a Tensor object**
     *
     * Performs modular translation on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #translate(String, String, int, int)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting translated tensor.
     * @param tW  Horizontal translation (in pixels). Positive → right.
     * @param tH  Vertical translation (in pixels). Positive → down.
     * @return A {@link CuBridge} instance representing the translation operation.
     * @see #translate(String, String, int, int)
     */
    default CuBridge translate(Tensor a, String out, int tW, int tH) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return translate(aName, out, tW, tH);
    }

    /**
     * **TranslateI — Immediate modular translation on a named tensor**
     *
     * Performs modular translation on the named tensor
     * and directly returns the resulting translated {@link Tensor}.
     * <p>
     * Pixels wrap around at the boundaries according to modular arithmetic.
     * </p>
     *
     * @param a  The name of the input tensor.
     * @param tW Horizontal translation (in pixels). Positive → right.
     * @param tH Vertical translation (in pixels). Positive → down.
     * @return A {@link Tensor} representing the translated output tensor.
     * @see #translate(String, String, int, int)
     */
    default Tensor translateI(String a, int tW, int tH) {
        String oName = genRandomNameImage();
        return translate(a, oName, tW, tH).get(oName);
    }

    /**
     * **TranslateI — Immediate modular translation on a Tensor object**
     *
     * Performs modular translation (wrap-around shift) on the given {@link Tensor} object
     * and directly returns the resulting translated {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #translate(String, String, int, int)}.
     * </p>
     *
     * @param a  The input tensor object.
     * @param tW Horizontal translation (in pixels). Positive → right.
     * @param tH Vertical translation (in pixels). Positive → down.
     * @return A {@link Tensor} representing the translated output tensor.
     * @see #translate(String, String, int, int)
     */
    default Tensor translateI(Tensor a, int tW, int tH) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return translate(aName, oName, tW, tH).get(oName);
    }


    /**
     * **Resize — Core spatial scaling operation**
     *
     * Resizes the specified tensor (image or matrix) by the given scaling factors.
     * <p>
     * The output dimensions are scaled by {@code scaleW} horizontally
     * and {@code scaleH} vertically. Bilinear interpolation is used internally
     * for smooth resizing.
     * </p>
     *
     * @param a       The name of the input tensor.
     * @param out     The name to store the resulting resized tensor.
     * @param scaleW  Horizontal scaling factor. (>1 = enlarge, <1 = shrink)
     * @param scaleH  Vertical scaling factor. (>1 = enlarge, <1 = shrink)
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge resize(String a, String out, float scaleW, float scaleH) {
        if (CuBridgeJNI.resize(a, out, scaleW, scaleH)) return CuBridge.getInstance();
        else System.err.println("Error | resize | " + a + " | " + out +
                " | scaleW=" + scaleW + " | scaleH=" + scaleH);
        return null;
    }

    /**
     * **Resize — Overload using a Tensor object**
     *
     * Resizes the given {@link Tensor} object by the specified horizontal and vertical scaling factors.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #resize(String, String, float, float)}.
     * </p>
     *
     * @param a       The input tensor object.
     * @param out     The name to store the resulting resized tensor.
     * @param scaleW  Horizontal scaling factor.
     * @param scaleH  Vertical scaling factor.
     * @return A {@link CuBridge} instance representing the resize operation.
     * @see #resize(String, String, float, float)
     */
    default CuBridge resize(Tensor a, String out, float scaleW, float scaleH) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return resize(aName, out, scaleW, scaleH);
    }

    /**
     * **ResizeI — Immediate resize operation on a named tensor**
     *
     * Resizes the named tensor by the given scaling factors
     * and directly returns the resulting resized {@link Tensor}.
     *
     * @param a       The name of the input tensor.
     * @param scaleW  Horizontal scaling factor. (>1 = enlarge)
     * @param scaleH  Vertical scaling factor. (>1 = enlarge)
     * @return A {@link Tensor} representing the resized output tensor.
     * @see #resize(String, String, float, float)
     */
    default Tensor resizeI(String a, float scaleW, float scaleH) {
        String oName = genRandomNameImage();
        return resize(a, oName, scaleW, scaleH).get(oName);
    }

    /**
     * **ResizeI — Immediate resize operation on a Tensor object**
     *
     * Resizes the given {@link Tensor} object by the specified scaling factors
     * and directly returns the resulting resized {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #resize(String, String, float, float)}.
     * </p>
     *
     * @param a       The input tensor object.
     * @param scaleW  Horizontal scaling factor.
     * @param scaleH  Vertical scaling factor.
     * @return A {@link Tensor} representing the resized output tensor.
     * @see #resize(String, String, float, float)
     */
    default Tensor resizeI(Tensor a, float scaleW, float scaleH) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return resize(aName, oName, scaleW, scaleH).get(oName);
    }


    /**
     * **Crop — Core region extraction operation**
     *
     * Crops a rectangular region from the specified tensor (image or matrix)
     * starting at the given offsets and with the specified height and width.
     * <p>
     * The resulting tensor contains only the selected subregion of the input tensor.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting cropped tensor.
     * @param cH  Crop height (in pixels).
     * @param cW  Crop width (in pixels).
     * @param sH  Start offset (top coordinate, in pixels).
     * @param sW  Start offset (left coordinate, in pixels).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge crop(String a, String out, int cH, int cW, int sH, int sW) {
        if (CuBridgeJNI.crop(a, out, cH, cW, sH, sW)) return CuBridge.getInstance();
        else System.err.println("Error | crop | " + a + " | " + out +
                " | cH=" + cH + " | cW=" + cW + " | sH=" + sH + " | sW=" + sW);
        return null;
    }

    /**
     * **Crop — Overload using a Tensor object**
     *
     * Crops the given {@link Tensor} object at the specified region.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #crop(String, String, int, int, int, int)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting cropped tensor.
     * @param cH  Crop height.
     * @param cW  Crop width.
     * @param sH  Start offset (top).
     * @param sW  Start offset (left).
     * @return A {@link CuBridge} instance representing the crop operation.
     * @see #crop(String, String, int, int, int, int)
     */
    default CuBridge crop(Tensor a, String out, int cH, int cW, int sH, int sW) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return crop(aName, out, cH, cW, sH, sW);
    }

    /**
     * **CropI — Immediate region extraction on a named tensor**
     *
     * Crops the specified region from the named tensor and directly returns the result.
     *
     * @param a  The name of the input tensor.
     * @param cH Crop height.
     * @param cW Crop width.
     * @param sH Start offset (top).
     * @param sW Start offset (left).
     * @return A {@link Tensor} representing the cropped output tensor.
     * @see #crop(String, String, int, int, int, int)
     */
    default Tensor cropI(String a, int cH, int cW, int sH, int sW) {
        String oName = genRandomNameImage();
        return crop(a, oName, cH, cW, sH, sW).get(oName);
    }

    /**
     * **CropI — Immediate region extraction on a Tensor object**
     *
     * Crops the specified region from the given {@link Tensor} object
     * and directly returns the resulting subregion.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #crop(String, String, int, int, int, int)}.
     * </p>
     *
     * @param a  The input tensor object.
     * @param cH Crop height.
     * @param cW Crop width.
     * @param sH Start offset (top).
     * @param sW Start offset (left).
     * @return A {@link Tensor} representing the cropped output tensor.
     * @see #crop(String, String, int, int, int, int)
     */
    default Tensor cropI(Tensor a, int cH, int cW, int sH, int sW) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return crop(aName, oName, cH, cW, sH, sW).get(oName);
    }


    /**
     * **Mask — Core masking operation**
     *
     * Applies a rectangular mask region to the specified tensor (image or matrix),
     * zeroing out all elements outside the defined region.
     * <p>
     * The mask region is defined by its height and width and starting coordinates.
     * Pixels outside the mask are set to 0.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting masked tensor.
     * @param mH  Mask height (in pixels).
     * @param mW  Mask width (in pixels).
     * @param sH  Start offset (top coordinate, in pixels).
     * @param sW  Start offset (left coordinate, in pixels).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge mask(String a, String out, int mH, int mW, int sH, int sW) {
        if (CuBridgeJNI.mask(a, out, mH, mW, sH, sW)) return CuBridge.getInstance();
        else System.err.println("Error | mask | " + a + " | " + out +
                " | mH=" + mH + " | mW=" + mW + " | sH=" + sH + " | sW=" + sW);
        return null;
    }

    /**
     * **Mask — Overload using a Tensor object**
     *
     * Applies a rectangular mask to the given {@link Tensor} object,
     * setting all pixels outside the defined region to 0.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #mask(String, String, int, int, int, int)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting masked tensor.
     * @param mH  Mask height.
     * @param mW  Mask width.
     * @param sH  Start offset (top).
     * @param sW  Start offset (left).
     * @return A {@link CuBridge} instance representing the masking operation.
     * @see #mask(String, String, int, int, int, int)
     */
    default CuBridge mask(Tensor a, String out, int mH, int mW, int sH, int sW) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return mask(aName, out, mH, mW, sH, sW);
    }

    /**
     * **MaskI — Immediate masking operation on a named tensor**
     *
     * Applies a rectangular mask to the named tensor and directly returns the masked output.
     *
     * @param a  The name of the input tensor.
     * @param mH Mask height.
     * @param mW Mask width.
     * @param sH Start offset (top).
     * @param sW Start offset (left).
     * @return A {@link Tensor} representing the masked output tensor.
     * @see #mask(String, String, int, int, int, int)
     */
    default Tensor maskI(String a, int mH, int mW, int sH, int sW) {
        String oName = genRandomNameImage();
        return mask(a, oName, mH, mW, sH, sW).get(oName);
    }

    /**
     * **MaskI — Immediate masking operation on a Tensor object**
     *
     * Applies a rectangular mask to the given {@link Tensor} object
     * and directly returns the masked output.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #mask(String, String, int, int, int, int)}.
     * </p>
     *
     * @param a  The input tensor object.
     * @param mH Mask height.
     * @param mW Mask width.
     * @param sH Start offset (top).
     * @param sW Start offset (left).
     * @return A {@link Tensor} representing the masked output tensor.
     * @see #mask(String, String, int, int, int, int)
     */
    default Tensor maskI(Tensor a, int mH, int mW, int sH, int sW) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return mask(aName, oName, mH, mW, sH, sW).get(oName);
    }


    /**
     * **Pad — Core padding operation**
     *
     * Pads the specified tensor (image or matrix) with the given number of pixels
     * on all sides, filling the added regions with a constant value.
     * <p>
     * The resulting tensor will have dimensions increased by {@code 2*pH} vertically
     * and {@code 2*pW} horizontally.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the resulting padded tensor.
     * @param pH  Padding height (number of pixels to add on top and bottom).
     * @param pW  Padding width (number of pixels to add on left and right).
     * @param val The constant value used to fill the padded region.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge pad(String a, String out, int pH, int pW, float val) {
        if (CuBridgeJNI.pad(a, out, pH, pW, val)) return CuBridge.getInstance();
        else System.err.println("Error | pad | " + a + " | " + out +
                " | pH=" + pH + " | pW=" + pW + " | val=" + val);
        return null;
    }

    /**
     * **Pad — Overload using a Tensor object**
     *
     * Pads the given {@link Tensor} object with the specified number of pixels
     * on each side and fills the padded region with the given value.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #pad(String, String, int, int, float)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the resulting padded tensor.
     * @param pH  Padding height (top and bottom).
     * @param pW  Padding width (left and right).
     * @param val The constant value used for padding.
     * @return A {@link CuBridge} instance representing the padding operation.
     * @see #pad(String, String, int, int, float)
     */
    default CuBridge pad(Tensor a, String out, int pH, int pW, float val) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return pad(aName, out, pH, pW, val);
    }

    /**
     * **PadI — Immediate padding operation on a named tensor**
     *
     * Pads the named tensor with the specified number of pixels and returns
     * the padded {@link Tensor} directly.
     *
     * @param a   The name of the input tensor.
     * @param pH  Padding height (top and bottom).
     * @param pW  Padding width (left and right).
     * @param val The constant value used for padding.
     * @return A {@link Tensor} representing the padded output tensor.
     * @see #pad(String, String, int, int, float)
     */
    default Tensor padI(String a, int pH, int pW, float val) {
        String oName = genRandomNameImage();
        return pad(a, oName, pH, pW, val).get(oName);
    }

    /**
     * **PadI — Immediate padding operation on a Tensor object**
     *
     * Pads the given {@link Tensor} object with the specified number of pixels
     * and directly returns the padded output tensor.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #pad(String, String, int, int, float)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param pH  Padding height (top and bottom).
     * @param pW  Padding width (left and right).
     * @param val The constant value used for padding.
     * @return A {@link Tensor} representing the padded output tensor.
     * @see #pad(String, String, int, int, float)
     */
    default Tensor padI(Tensor a, int pH, int pW, float val) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return pad(aName, oName, pH, pW, val).get(oName);
    }


    /**
     * **BoxBlur — Core box filter operation**
     *
     * Applies a simple mean (box) blur to the specified tensor (image or matrix)
     * using a square kernel of size {@code kSize × kSize}.
     * <p>
     * Each output pixel is the average of its neighboring pixels
     * within the specified kernel window.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting blurred tensor.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge boxBlur(String a, String out, int kSize) {
        if (CuBridgeJNI.boxBlur(a, out, kSize)) return CuBridge.getInstance();
        else System.err.println("Error | boxBlur | " + a + " | " + out + " | kSize=" + kSize);
        return null;
    }

    /**
     * **BoxBlur — Overload using a Tensor object**
     *
     * Applies a mean (box) blur to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #boxBlur(String, String, int)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param out   The name to store the resulting blurred tensor.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link CuBridge} instance representing the blur operation.
     * @see #boxBlur(String, String, int)
     */
    default CuBridge boxBlur(Tensor a, String out, int kSize) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return boxBlur(aName, out, kSize);
    }

    /**
     * **BoxBlurI — Immediate box blur operation on a named tensor**
     *
     * Applies a box blur to the specified named tensor and directly returns the result.
     *
     * @param a     The name of the input tensor.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link Tensor} representing the blurred output tensor.
     * @see #boxBlur(String, String, int)
     */
    default Tensor boxBlurI(String a, int kSize) {
        String oName = genRandomNameImage();
        return boxBlur(a, oName, kSize).get(oName);
    }

    /**
     * **BoxBlurI — Immediate box blur operation on a Tensor object**
     *
     * Applies a box blur to the given {@link Tensor} object and directly returns the blurred output.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #boxBlur(String, String, int)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link Tensor} representing the blurred output tensor.
     * @see #boxBlur(String, String, int)
     */
    default Tensor boxBlurI(Tensor a, int kSize) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return boxBlur(aName, oName, kSize).get(oName);
    }


    /**
     * **GaussianBlur — Core Gaussian smoothing operation**
     *
     * Applies a Gaussian blur to the specified tensor (image or matrix)
     * using a kernel of size {@code kSize × kSize}.
     * <p>
     * The kernel is generated from a 2D Gaussian distribution.
     * Larger kernel sizes produce stronger blurring effects.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting blurred tensor.
     * @param kSize The Gaussian kernel size (must be odd, e.g., 3, 5, 7).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge gaussianBlur(String a, String out, int kSize) {
        if (CuBridgeJNI.gaussianBlur(a, out, kSize)) return CuBridge.getInstance();
        else System.err.println("Error | gaussianBlur | " + a + " | " + out + " | kSize=" + kSize);
        return null;
    }

    /**
     * **GaussianBlur — Overload using a Tensor object**
     *
     * Applies Gaussian smoothing to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #gaussianBlur(String, String, int)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param out   The name to store the resulting blurred tensor.
     * @param kSize The Gaussian kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link CuBridge} instance representing the blur operation.
     * @see #gaussianBlur(String, String, int)
     */
    default CuBridge gaussianBlur(Tensor a, String out, int kSize) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return gaussianBlur(aName, out, kSize);
    }

    /**
     * **GaussianBlurI — Immediate Gaussian blur operation on a named tensor**
     *
     * Applies Gaussian smoothing to the specified named tensor
     * and directly returns the blurred {@link Tensor}.
     *
     * @param a     The name of the input tensor.
     * @param kSize The Gaussian kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link Tensor} representing the blurred output tensor.
     * @see #gaussianBlur(String, String, int)
     */
    default Tensor gaussianBlurI(String a, int kSize) {
        String oName = genRandomNameImage();
        return gaussianBlur(a, oName, kSize).get(oName);
    }

    /**
     * **GaussianBlurI — Immediate Gaussian blur operation on a Tensor object**
     *
     * Applies Gaussian smoothing to the given {@link Tensor} object
     * and directly returns the blurred {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #gaussianBlur(String, String, int)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param kSize The Gaussian kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link Tensor} representing the blurred output tensor.
     * @see #gaussianBlur(String, String, int)
     */
    default Tensor gaussianBlurI(Tensor a, int kSize) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return gaussianBlur(aName, oName, kSize).get(oName);
    }


    /**
     * **MedianBlur — Core median filter operation**
     *
     * Applies a median blur to the specified tensor (image or matrix)
     * using a square kernel of size {@code kSize × kSize}.
     * <p>
     * Each output pixel is replaced by the median value of its neighboring pixels
     * within the kernel window. This effectively removes salt-and-pepper noise
     * while preserving edges.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the resulting blurred tensor.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge medianBlur(String a, String out, int kSize) {
        if (CuBridgeJNI.medianBlur(a, out, kSize)) return CuBridge.getInstance();
        else System.err.println("Error | medianBlur | " + a + " | " + out + " | kSize=" + kSize);
        return null;
    }

    /**
     * **MedianBlur — Overload using a Tensor object**
     *
     * Applies median filtering to the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #medianBlur(String, String, int)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param out   The name to store the resulting blurred tensor.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link CuBridge} instance representing the blur operation.
     * @see #medianBlur(String, String, int)
     */
    default CuBridge medianBlur(Tensor a, String out, int kSize) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return medianBlur(aName, out, kSize);
    }

    /**
     * **MedianBlurI — Immediate median blur operation on a named tensor**
     *
     * Applies a median blur to the specified named tensor
     * and directly returns the blurred {@link Tensor}.
     *
     * @param a     The name of the input tensor.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link Tensor} representing the blurred output tensor.
     * @see #medianBlur(String, String, int)
     */
    default Tensor medianBlurI(String a, int kSize) {
        String oName = genRandomNameImage();
        return medianBlur(a, oName, kSize).get(oName);
    }

    /**
     * **MedianBlurI — Immediate median blur operation on a Tensor object**
     *
     * Applies a median blur to the given {@link Tensor} object
     * and directly returns the blurred {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #medianBlur(String, String, int)}.
     * </p>
     *
     * @param a     The input tensor object.
     * @param kSize The kernel size (must be odd, e.g., 3, 5, 7).
     * @return A {@link Tensor} representing the blurred output tensor.
     * @see #medianBlur(String, String, int)
     */
    default Tensor medianBlurI(Tensor a, int kSize) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return medianBlur(aName, oName, kSize).get(oName);
    }


    /**
     * **FlipH — Core horizontal flip operation**
     *
     * Performs a horizontal flip (mirror) on the specified tensor (image or matrix),
     * reversing the order of columns.
     * <p>
     * Each row is reversed left-to-right, producing a mirror image along the vertical axis.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the horizontally flipped tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge flipH(String a, String out) {
        if (CuBridgeJNI.flipH(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | flipH | " + a + " | " + out);
        return null;
    }

    /**
     * **FlipH — Overload using a Tensor object**
     *
     * Performs a horizontal flip (mirror) on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #flipH(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the horizontally flipped tensor.
     * @return A {@link CuBridge} instance representing the flip operation.
     * @see #flipH(String, String)
     */
    default CuBridge flipH(Tensor a, String out) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return flipH(aName, out);
    }

    /**
     * **FlipHI — Immediate horizontal flip on a named tensor**
     *
     * Performs a horizontal flip on the named tensor
     * and directly returns the resulting flipped {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} representing the horizontally flipped output tensor.
     * @see #flipH(String, String)
     */
    default Tensor flipHI(String a) {
        String oName = genRandomNameImage();
        return flipH(a, oName).get(oName);
    }

    /**
     * **FlipHI — Immediate horizontal flip on a Tensor object**
     *
     * Performs a horizontal flip on the given {@link Tensor} object
     * and directly returns the flipped {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #flipH(String, String)}.
     * </p>
     *
     * @param a The input tensor object.
     * @return A {@link Tensor} representing the horizontally flipped output tensor.
     * @see #flipH(String, String)
     */
    default Tensor flipHI(Tensor a) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return flipH(aName, oName).get(oName);
    }


    /**
     * **FlipV — Core vertical flip operation**
     *
     * Performs a vertical flip on the specified tensor (image or matrix),
     * reversing the order of rows.
     * <p>
     * The top and bottom of the tensor are swapped,
     * producing a mirror image along the horizontal axis.
     * </p>
     *
     * @param a   The name of the input tensor.
     * @param out The name to store the vertically flipped tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge flipV(String a, String out) {
        if (CuBridgeJNI.flipV(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | flipV | " + a + " | " + out);
        return null;
    }

    /**
     * **FlipV — Overload using a Tensor object**
     *
     * Performs a vertical flip on the given {@link Tensor} object.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #flipV(String, String)}.
     * </p>
     *
     * @param a   The input tensor object.
     * @param out The name to store the vertically flipped tensor.
     * @return A {@link CuBridge} instance representing the flip operation.
     * @see #flipV(String, String)
     */
    default CuBridge flipV(Tensor a, String out) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return flipV(aName, out);
    }

    /**
     * **FlipVI — Immediate vertical flip on a named tensor**
     *
     * Performs a vertical flip on the named tensor
     * and directly returns the resulting flipped {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} representing the vertically flipped output tensor.
     * @see #flipV(String, String)
     */
    default Tensor flipVI(String a) {
        String oName = genRandomNameImage();
        return flipV(a, oName).get(oName);
    }

    /**
     * **FlipVI — Immediate vertical flip on a Tensor object**
     *
     * Performs a vertical flip on the given {@link Tensor} object
     * and directly returns the flipped {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #flipV(String, String)}.
     * </p>
     *
     * @param a The input tensor object.
     * @return A {@link Tensor} representing the vertically flipped output tensor.
     * @see #flipV(String, String)
     */
    default Tensor flipVI(Tensor a) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return flipV(aName, oName).get(oName);
    }


    /**
     * **GrayScale — Core color-to-grayscale conversion**
     *
     * Converts the specified color tensor (RGB image) into a single-channel
     * grayscale tensor using standard luminance coefficients.
     * <p>
     * The conversion follows the BT.601 formula:
     * <pre>
     * Gray = 0.299 * R + 0.587 * G + 0.114 * B
     * </pre>
     * The resulting tensor has the same spatial dimensions as the input but only one channel.
     * </p>
     *
     * @param a   The name of the input color tensor.
     * @param out The name to store the resulting grayscale tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge grayScale(String a, String out) {
        if (CuBridgeJNI.grayScale(a, out)) return CuBridge.getInstance();
        else System.err.println("Error | grayScale | " + a + " | " + out);
        return null;
    }

    /**
     * **GrayScale — Overload using a Tensor object**
     *
     * Converts the given {@link Tensor} object from RGB color to grayscale.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #grayScale(String, String)}.
     * </p>
     *
     * @param a   The input color tensor object.
     * @param out The name to store the resulting grayscale tensor.
     * @return A {@link CuBridge} instance representing the grayscale conversion operation.
     * @see #grayScale(String, String)
     */
    default CuBridge grayScale(Tensor a, String out) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return grayScale(aName, out);
    }

    /**
     * **GrayScaleI — Immediate grayscale conversion on a named tensor**
     *
     * Converts the specified named color tensor into grayscale
     * and directly returns the resulting single-channel {@link Tensor}.
     *
     * @param a The name of the input color tensor.
     * @return A {@link Tensor} representing the grayscale output tensor.
     * @see #grayScale(String, String)
     */
    default Tensor grayScaleI(String a) {
        String oName = genRandomNameImage();
        return grayScale(a, oName).get(oName);
    }

    /**
     * **GrayScaleI — Immediate grayscale conversion on a Tensor object**
     *
     * Converts the given {@link Tensor} object into grayscale
     * and directly returns the resulting single-channel {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #grayScale(String, String)}.
     * </p>
     *
     * @param a The input color tensor object.
     * @return A {@link Tensor} representing the grayscale output tensor.
     * @see #grayScale(String, String)
     */
    default Tensor grayScaleI(Tensor a) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameImage();
        return grayScale(aName, oName).get(oName);
    }


    /**
     * **ChSplit — Core RGB channel separation operation**
     *
     * Splits a color tensor (RGB or RGBA image) into three separate single-channel tensors:
     * {@code R}, {@code G}, and {@code B}.
     * <p>
     * The input tensor must have three channels (RGB) or four channels (RGBA, with alpha ignored).
     * Each output tensor will contain one channel with the same spatial dimensions.
     * </p>
     *
     * @param a The name of the input color tensor.
     * @param R The name to store the red channel tensor.
     * @param G The name to store the green channel tensor.
     * @param B The name to store the blue channel tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge chSplit(String a, String R, String G, String B) {
        if (CuBridgeJNI.chSplit(a, R, G, B)) return CuBridge.getInstance();
        else System.err.println("Error | chSplit | " + a + " | R=" + R + " | G=" + G + " | B=" + B);
        return null;
    }

    /**
     * **ChSplit — Overload using a Tensor object**
     *
     * Splits the given {@link Tensor} object into three separate single-channel tensors:
     * {@code R}, {@code G}, and {@code B}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #chSplit(String, String, String, String)}.
     * </p>
     *
     * @param a The input color tensor object.
     * @param R The name to store the red channel tensor.
     * @param G The name to store the green channel tensor.
     * @param B The name to store the blue channel tensor.
     * @return A {@link CuBridge} instance representing the channel split operation.
     * @see #chSplit(String, String, String, String)
     */
    default CuBridge chSplit(Tensor a, String R, String G, String B) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return chSplit(aName, R, G, B);
    }

    /**
     * **ChSplitI — Immediate RGB channel separation on a named tensor**
     *
     * Splits a named RGB tensor into its {@code R}, {@code G}, and {@code B} components
     * and directly returns them as an array of {@link Tensor}.
     *
     * @param a The name of the input color tensor.
     * @return An array of three {@link Tensor} objects: {R, G, B}.
     * @see #chSplit(String, String, String, String)
     */
    default Tensor[] chSplitI(String a) {
        String rName = genRandomNameImage();
        String gName = genRandomNameImage();
        String bName = genRandomNameImage();
        chSplit(a, rName, gName, bName);
        return new Tensor[] {   CuBridge.getInstance().get(rName),
                                CuBridge.getInstance().get(gName),
                                CuBridge.getInstance().get(bName) };
    }

    /**
     * **ChSplitI — Immediate RGB channel separation on a Tensor object**
     *
     * Splits the given {@link Tensor} object into its RGB components
     * and directly returns them as an array of {@link Tensor}.
     * <p>
     * Automatically assigns a random internal name to the tensor before executing
     * {@link #chSplit(String, String, String, String)}.
     * </p>
     *
     * @param a The input color tensor object.
     * @return An array of three {@link Tensor} objects: {R, G, B}.
     * @see #chSplit(String, String, String, String)
     */
    default Tensor[] chSplitI(Tensor a) {
        String aName = genRandomNameImage(); CuBridge.getInstance().put(a, aName);
        return chSplitI(aName);
    }


    /**
     * **ChMerge — Core RGB channel merge operation**
     *
     * Merges three single-channel tensors {@code r}, {@code g}, and {@code b}
     * into a single RGB color tensor.
     * <p>
     * Each input tensor must have the same spatial dimensions.
     * The resulting tensor will have three channels (RGB).
     * </p>
     *
     * @param r   The name of the red channel tensor.
     * @param g   The name of the green channel tensor.
     * @param b   The name of the blue channel tensor.
     * @param out The name to store the resulting RGB tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge chMerge(String r, String g, String b, String out) {
        if (CuBridgeJNI.chMerge(r, g, b, out)) return CuBridge.getInstance();
        else System.err.println("Error | chMerge | r=" + r + " | g=" + g + " | b=" + b + " | out=" + out);
        return null;
    }

    /**
     * **ChMerge — Overload using Tensor objects**
     *
     * Merges the given single-channel {@link Tensor} objects into a single RGB tensor.
     * <p>
     * Automatically assigns random internal names to each input tensor before executing
     * {@link #chMerge(String, String, String, String)}.
     * </p>
     *
     * @param r   The red channel tensor.
     * @param g   The green channel tensor.
     * @param b   The blue channel tensor.
     * @param out The name to store the resulting RGB tensor.
     * @return A {@link CuBridge} instance representing the channel merge operation.
     * @see #chMerge(String, String, String, String)
     */
    default CuBridge chMerge(Tensor r, Tensor g, Tensor b, String out) {
        String rName = genRandomNameImage(); CuBridge.getInstance().put(r, rName);
        String gName = genRandomNameImage(); CuBridge.getInstance().put(g, gName);
        String bName = genRandomNameImage(); CuBridge.getInstance().put(b, bName);
        return chMerge(rName, gName, bName, out);
    }

    /**
     * **ChMergeI — Immediate RGB merge operation**
     *
     * Merges the three named channel tensors into one RGB tensor
     * and directly returns the resulting {@link Tensor}.
     *
     * @param r The name of the red channel tensor.
     * @param g The name of the green channel tensor.
     * @param b The name of the blue channel tensor.
     * @return A {@link Tensor} representing the merged RGB tensor.
     * @see #chMerge(String, String, String, String)
     */
    default Tensor chMergeI(String r, String g, String b) {
        String oName = genRandomNameImage();
        return chMerge(r, g, b, oName).get(oName);
    }

    /**
     * **ChMergeI — Immediate RGB merge operation using Tensor objects**
     *
     * Merges the three given single-channel {@link Tensor} objects into one RGB tensor
     * and directly returns the resulting {@link Tensor}.
     * <p>
     * Automatically assigns random internal names to each tensor before executing
     * {@link #chMerge(String, String, String, String)}.
     * </p>
     *
     * @param r The red channel tensor.
     * @param g The green channel tensor.
     * @param b The blue channel tensor.
     * @return A {@link Tensor} representing the merged RGB tensor.
     * @see #chMerge(String, String, String, String)
     */
    default Tensor chMergeI(Tensor r, Tensor g, Tensor b) {
        String rName = genRandomNameImage(); CuBridge.getInstance().put(r, rName);
        String gName = genRandomNameImage(); CuBridge.getInstance().put(g, gName);
        String bName = genRandomNameImage(); CuBridge.getInstance().put(b, bName);
        String oName = genRandomNameImage();
        return chMerge(rName, gName, bName, oName).get(oName);
    }


    /**
     * **Im2Col1D — Core 1D convolution unfolding operation**
     *
     * Converts a 1D input tensor into column representation suitable for
     * matrix multiplication–based convolution.
     * effectively reversing the {@link #col2im1D(String, String, String, int, int, int)} operation.
     * <p>
     * Each column represents a receptive field segment extracted from the input,
     * based on the given kernel size, padding, and stride.
     * </p>
     *
     * @param input  The name of the input tensor.
     * @param kernel The name of the kernel tensor.
     * @param out    The name to store the resulting column-expanded tensor.
     * @param pad    The number of zero-padding elements added to both sides.
     * @param stride The step size for sliding the kernel.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge im2col1D(String input, String kernel, String out, int pad, int stride) {
        if (CuBridgeJNI.im2col1D(input, kernel, out, pad, stride)) return CuBridge.getInstance();
        else System.err.println("Error | im2col1D | " + input + " | " + kernel +
                " | pad=" + pad + " | stride=" + stride);
        return null;
    }

    /**
     * **Im2Col1D — Overload without padding (pad = 0)**
     *
     * Performs the same operation as {@link #im2col1D(String, String, String, int, int)}
     * but without zero-padding. Equivalent to calling the same function with {@code pad = 0}.
     *
     * @param input  The name of the input tensor.
     * @param kernel The name of the kernel tensor.
     * @param out    The name to store the resulting column-expanded tensor.
     * @param stride The step size for sliding the kernel.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge im2col1D(String input, String kernel, String out, int stride) {
        return im2col1D(input, kernel, out, 0, stride);
    }

    /**
     * **Im2Col1D — Overload using Tensor objects**
     *
     * Converts the given {@link Tensor} objects (input and kernel)
     * into column representation suitable for 1D convolution.
     *
     * @param input  The input tensor object.
     * @param kernel The kernel tensor object.
     * @param out    The name to store the resulting column-expanded tensor.
     * @param pad    The number of zero-padding elements added to both sides.
     * @param stride The step size for sliding the kernel.
     * @return A {@link CuBridge} instance representing the im2col operation.
     * @see #im2col1D(String, String, String, int, int)
     */
    default CuBridge im2col1D(Tensor input, Tensor kernel, String out, int pad, int stride) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return im2col1D(iName, kName, out, pad, stride);
    }

    /**
     * **Im2Col1D — Overload using Tensor objects (no padding, pad = 0)**
     *
     * Converts the given {@link Tensor} objects (input and kernel)
     * into column representation suitable for 1D convolution without padding.
     *
     * @param input  The input tensor object.
     * @param kernel The kernel tensor object.
     * @param out    The name to store the resulting column-expanded tensor.
     * @param stride The step size for sliding the kernel.
     * @return A {@link CuBridge} instance representing the im2col operation.
     * @see #im2col1D(String, String, String, int, int)
     */
    default CuBridge im2col1D(Tensor input, Tensor kernel, String out, int stride) {
        return im2col1D(input, kernel, out, 0, stride);
    }

    /**
     * **Im2Col1DI — Immediate column expansion on named tensors**
     *
     * Converts the named input tensor and kernel tensor into column representation
     * and directly returns the resulting {@link Tensor}.
     *
     * @param input  The name of the input tensor.
     * @param kernel The name of the kernel tensor.
     * @param pad    The number of zero-padding elements.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the column-expanded output.
     * @see #im2col1D(String, String, String, int, int)
     */
    default Tensor im2col1DI(String input, String kernel, int pad, int stride) {
        String oName = genRandomNameImage();
        return im2col1D(input, kernel, oName, pad, stride).get(oName);
    }

    /**
     * **Im2Col1DI — Immediate column expansion without padding (pad = 0)**
     *
     * Same as {@link #im2col1DI(String, String, int, int)} but with {@code pad = 0}.
     *
     * @param input  The name of the input tensor.
     * @param kernel The name of the kernel tensor.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the column-expanded output.
     * @see #im2col1D(String, String, String, int, int)
     */
    default Tensor im2col1DI(String input, String kernel, int stride) {
        return im2col1DI(input, kernel, 0, stride);
    }

    /**
     * **Im2Col1DI — Immediate column expansion on Tensor objects**
     *
     * Converts the given {@link Tensor} input and kernel into column representation
     * and directly returns the resulting {@link Tensor}.
     *
     * @param input  The input tensor object.
     * @param kernel The kernel tensor object.
     * @param pad    The number of zero-padding elements.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the column-expanded output tensor.
     * @see #im2col1D(String, String, String, int, int)
     */
    default Tensor im2col1DI(Tensor input, Tensor kernel, int pad, int stride) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return im2col1DI(iName, kName, pad, stride);
    }

    /**
     * **Im2Col1DI — Immediate column expansion on Tensor objects without padding**
     *
     * Performs {@code pad = 0} im2col operation on Tensor inputs.
     *
     * @param input  The input tensor object.
     * @param kernel The kernel tensor object.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the column-expanded output tensor.
     * @see #im2col1D(String, String, String, int, int)
     */
    default Tensor im2col1DI(Tensor input, Tensor kernel, int stride) {
        return im2col1DI(input, kernel, 0, stride);
    }


    /**
     * **Col2Im1D — Core 1D convolution reconstruction operation**
     *
     * Converts column-expanded data back into a 1D tensor representation,
     * effectively reversing the {@link #im2col1D(String, String, String, int, int)} operation.
     *
     * @param input  The name of the column-expanded tensor.
     * @param kernel The name of the kernel tensor.
     * @param out    The name to store the reconstructed tensor.
     * @param oL     The output length (number of elements in the reconstructed tensor).
     * @param pad    The number of zero-padding elements used during {@code im2col1D}.
     * @param stride The stride size used during convolution.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge col2im1D(String input, String kernel, String out, int oL, int pad, int stride) {
        if (CuBridgeJNI.col2im1D(input, kernel, out, oL, pad, stride)) return CuBridge.getInstance();
        else System.err.println("Error | col2im1D | " + input + " | " + kernel +
                " | oL=" + oL + " | pad=" + pad + " | stride=" + stride);
        return null;
    }

    /**
     * **Col2Im1D — Overload without padding (pad = 0)**
     *
     * Performs the same operation as {@link #col2im1D(String, String, String, int, int, int)}
     * but assumes no padding was applied during {@code im2col1D}.
     *
     * @param input  The name of the column-expanded tensor.
     * @param kernel The name of the kernel tensor.
     * @param out    The name to store the reconstructed tensor.
     * @param oL     The output length.
     * @param stride The stride size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge col2im1D(String input, String kernel, String out, int oL, int stride) {
        return col2im1D(input, kernel, out, oL, 0, stride);
    }

    /**
     * **Col2Im1D — Overload using Tensor objects**
     *
     * Performs the 1D column-to-image reconstruction using given {@link Tensor} objects.
     *
     * @param input  The column-expanded input tensor.
     * @param kernel The kernel tensor.
     * @param out    The name to store the reconstructed tensor.
     * @param oL     The output length.
     * @param pad    The number of zero-padding elements.
     * @param stride The stride size.
     * @return A {@link CuBridge} instance representing the reconstruction operation.
     * @see #col2im1D(String, String, String, int, int, int)
     */
    default CuBridge col2im1D(Tensor input, Tensor kernel, String out, int oL, int pad, int stride) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return col2im1D(iName, kName, out, oL, pad, stride);
    }

    /**
     * **Col2Im1D — Overload using Tensor objects (no padding, pad = 0)**
     *
     * Performs the 1D column-to-image reconstruction using the given {@link Tensor} objects
     * without padding.
     *
     * @param input  The column-expanded input tensor.
     * @param kernel The kernel tensor.
     * @param out    The name to store the reconstructed tensor.
     * @param oL     The output length.
     * @param stride The stride size.
     * @return A {@link CuBridge} instance representing the reconstruction operation.
     * @see #col2im1D(String, String, String, int, int, int)
     */
    default CuBridge col2im1D(Tensor input, Tensor kernel, String out, int oL, int stride) {
        return col2im1D(input, kernel, out, oL, 0, stride);
    }

    /**
     * **Col2Im1DI — Immediate reconstruction on named tensors**
     *
     * Converts column-expanded data back into its 1D tensor form
     * and directly returns the reconstructed {@link Tensor}.
     *
     * @param input  The name of the column-expanded tensor.
     * @param kernel The name of the kernel tensor.
     * @param oL     The output length.
     * @param pad    The number of zero-padding elements.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the reconstructed 1D output.
     * @see #col2im1D(String, String, String, int, int, int)
     */
    default Tensor col2im1DI(String input, String kernel, int oL, int pad, int stride) {
        String oName = genRandomNameImage();
        return col2im1D(input, kernel, oName, oL, pad, stride).get(oName);
    }

    /**
     * **Col2Im1DI — Immediate reconstruction without padding (pad = 0)**
     *
     * Performs the same as {@link #col2im1DI(String, String, int, int, int)} but assumes no padding.
     *
     * @param input  The name of the column-expanded tensor.
     * @param kernel The name of the kernel tensor.
     * @param oL     The output length.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the reconstructed 1D output.
     * @see #col2im1D(String, String, String, int, int, int)
     */
    default Tensor col2im1DI(String input, String kernel, int oL, int stride) {
        return col2im1DI(input, kernel, oL, 0, stride);
    }

    /**
     * **Col2Im1DI — Immediate reconstruction using Tensor objects**
     *
     * Reconstructs the 1D tensor using the given {@link Tensor} inputs
     * and directly returns the resulting {@link Tensor}.
     *
     * @param input  The column-expanded tensor object.
     * @param kernel The kernel tensor object.
     * @param oL     The output length.
     * @param pad    The number of zero-padding elements.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the reconstructed output tensor.
     * @see #col2im1D(String, String, String, int, int, int)
     */
    default Tensor col2im1DI(Tensor input, Tensor kernel, int oL, int pad, int stride) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return col2im1DI(iName, kName, oL, pad, stride);
    }

    /**
     * **Col2Im1DI — Immediate reconstruction using Tensor objects without padding**
     *
     * Same as {@link #col2im1DI(Tensor, Tensor, int, int, int)} but with {@code pad = 0}.
     *
     * @param input  The column-expanded tensor object.
     * @param kernel The kernel tensor object.
     * @param oL     The output length.
     * @param stride The stride size.
     * @return A {@link Tensor} representing the reconstructed output tensor.
     * @see #col2im1D(String, String, String, int, int, int)
     */
    default Tensor col2im1DI(Tensor input, Tensor kernel, int oL, int stride) {
        return col2im1DI(input, kernel, oL, 0, stride);
    }


    /**
     * **Im2Col2D — Core 2D convolution unfolding operation**
     *
     * Converts a 2D input tensor (e.g., image) into column representation suitable for
     * matrix multiplication–based convolution.
     * <p>
     * Each column corresponds to one receptive field region of the input tensor.
     * Padding and stride are specified separately for height and width.
     * </p>
     *
     * @param input   The name of the input tensor.
     * @param kernel  The name of the kernel tensor.
     * @param out     The name to store the resulting column-expanded tensor.
     * @param padH    Padding applied to top and bottom.
     * @param padW    Padding applied to left and right.
     * @param strideH Stride along the height dimension.
     * @param strideW Stride along the width dimension.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW) {
        if (CuBridgeJNI.im2col2D(input, kernel, out, padH, padW, strideH, strideW))
            return CuBridge.getInstance();
        else System.err.println("Error | im2col2D | " + input + " | " + kernel +
                " | padH=" + padH + ", padW=" + padW +
                " | strideH=" + strideH + ", strideW=" + strideW);
        return null;
    }

    /**
     * **Im2Col2D — Simplified overload (single pad and stride value)**
     *
     * Performs {@link #im2col2D(String, String, String, int, int, int, int)} using
     * equal padding and stride values along both dimensions.
     *
     * @param input   The name of the input tensor.
     * @param kernel  The name of the kernel tensor.
     * @param out     The name to store the resulting column-expanded tensor.
     * @param pad     Padding applied to all sides (same for H and W).
     * @param stride  Stride applied to both height and width.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge im2col2D(String input, String kernel, String out, int pad, int stride) {
        return im2col2D(input, kernel, out, pad, pad, stride, stride);
    }

    /**
     * **Im2Col2D — Overload using Tensor objects**
     *
     * Converts the given {@link Tensor} input and kernel into column representation
     * suitable for 2D convolution.
     *
     * @param input   The input tensor object.
     * @param kernel  The kernel tensor object.
     * @param out     The name to store the resulting column-expanded tensor.
     * @param padH    Padding for height.
     * @param padW    Padding for width.
     * @param strideH Stride for height.
     * @param strideW Stride for width.
     * @return A {@link CuBridge} instance representing the im2col operation.
     */
    default CuBridge im2col2D(Tensor input, Tensor kernel, String out, int padH, int padW, int strideH, int strideW) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return im2col2D(iName, kName, out, padH, padW, strideH, strideW);
    }

    /**
     * **Im2Col2D — Tensor overload with equal pad and stride**
     *
     * Same as {@link #im2col2D(Tensor, Tensor, String, int, int, int, int)} but applies
     * equal padding and stride along both dimensions.
     *
     * @param input  The input tensor object.
     * @param kernel The kernel tensor object.
     * @param out    The name to store the resulting column-expanded tensor.
     * @param pad    Padding applied to all sides.
     * @param stride Stride applied to both axes.
     * @return A {@link CuBridge} instance representing the im2col operation.
     */
    default CuBridge im2col2D(Tensor input, Tensor kernel, String out, int pad, int stride) {
        return im2col2D(input, kernel, out, pad, pad, stride, stride);
    }

    /**
     * **Im2Col2DI — Immediate im2col operation on named tensors**
     *
     * Converts the named input and kernel tensors into column representation
     * and directly returns the resulting {@link Tensor}.
     *
     * @param input   The name of the input tensor.
     * @param kernel  The name of the kernel tensor.
     * @param padH    Padding for height.
     * @param padW    Padding for width.
     * @param strideH Stride for height.
     * @param strideW Stride for width.
     * @return A {@link Tensor} representing the column-expanded output.
     */
    default Tensor im2col2DI(String input, String kernel, int padH, int padW, int strideH, int strideW) {
        String oName = genRandomNameImage();
        return im2col2D(input, kernel, oName, padH, padW, strideH, strideW).get(oName);
    }

    /**
     * **Im2Col2DI — Immediate version with equal pad and stride**
     *
     * Same as {@link #im2col2DI(String, String, int, int, int, int)} but applies equal
     * pad and stride along both axes.
     *
     * @param input  The name of the input tensor.
     * @param kernel The name of the kernel tensor.
     * @param pad    Padding applied to all sides.
     * @param stride Stride applied to both axes.
     * @return A {@link Tensor} representing the column-expanded output.
     */
    default Tensor im2col2DI(String input, String kernel, int pad, int stride) {
        return im2col2DI(input, kernel, pad, pad, stride, stride);
    }

    /**
     * **Im2Col2DI — Immediate Tensor-based im2col operation**
     *
     * Converts the given {@link Tensor} objects into column representation
     * and directly returns the resulting {@link Tensor}.
     *
     * @param input   The input tensor object.
     * @param kernel  The kernel tensor object.
     * @param padH    Padding for height.
     * @param padW    Padding for width.
     * @param strideH Stride for height.
     * @param strideW Stride for width.
     * @return A {@link Tensor} representing the column-expanded output tensor.
     */
    default Tensor im2col2DI(Tensor input, Tensor kernel, int padH, int padW, int strideH, int strideW) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return im2col2DI(iName, kName, padH, padW, strideH, strideW);
    }

    /**
     * **Im2Col2DI — Immediate Tensor-based im2col with equal pad and stride**
     *
     * Performs im2col with same pad and stride on both axes.
     *
     * @param input  The input tensor object.
     * @param kernel The kernel tensor object.
     * @param pad    Padding (same for H and W).
     * @param stride Stride (same for H and W).
     * @return A {@link Tensor} representing the column-expanded output.
     */
    default Tensor im2col2DI(Tensor input, Tensor kernel, int pad, int stride) {
        return im2col2DI(input, kernel, pad, pad, stride, stride);
    }


    /**
     * **Col2Im2D — Core 2D convolution reconstruction operation**
     *
     * Converts a 2D column-expanded tensor back into its spatial tensor form,
     * effectively reversing the {@link #im2col2D(String, String, String, int, int, int, int)} operation.
     *
     * @param input   The name of the column-expanded tensor.
     * @param kernel  The name of the kernel tensor.
     * @param out     The name to store the reconstructed tensor.
     * @param oH      Output height.
     * @param oW      Output width.
     * @param padH    Padding applied during im2col (height).
     * @param padW    Padding applied during im2col (width).
     * @param strideH Stride along height.
     * @param strideW Stride along width.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW) {
        if (CuBridgeJNI.col2im2D(input, kernel, out, oH, oW, padH, padW, strideH, strideW))
            return CuBridge.getInstance();
        else System.err.println("Error | col2im2D | " + input + " | " + kernel +
                " | oH=" + oH + ", oW=" + oW +
                " | padH=" + padH + ", padW=" + padW +
                " | strideH=" + strideH + ", strideW=" + strideW);
        return null;
    }

    /**
     * **Col2Im2D — Simplified overload (single o, pad, stride)**
     *
     * Performs the same operation using equal height and width for output size,
     * padding, and stride.
     *
     * @param input  The name of the column-expanded tensor.
     * @param kernel The name of the kernel tensor.
     * @param out    The name to store the reconstructed tensor.
     * @param o      Output height and width (same value).
     * @param pad    Padding applied equally to all sides.
     * @param stride Stride applied equally to both axes.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge col2im2D(String input, String kernel, String out, int o, int pad, int stride) {
        return col2im2D(input, kernel, out, o, o, pad, pad, stride, stride);
    }

    /**
     * **Col2Im2D — Overload using Tensor objects**
     *
     * Performs the reconstruction operation using {@link Tensor} inputs.
     *
     * @param input   The column-expanded input tensor.
     * @param kernel  The kernel tensor.
     * @param out     The name to store the reconstructed tensor.
     * @param oH      Output height.
     * @param oW      Output width.
     * @param padH    Padding for height.
     * @param padW    Padding for width.
     * @param strideH Stride for height.
     * @param strideW Stride for width.
     * @return A {@link CuBridge} instance representing the reconstruction operation.
     */
    default CuBridge col2im2D(Tensor input, Tensor kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return col2im2D(iName, kName, out, oH, oW, padH, padW, strideH, strideW);
    }

    /**
     * **Col2Im2D — Tensor overload with equal o, pad, and stride**
     *
     * Performs the reconstruction with same height and width for all parameters.
     *
     * @param input  The column-expanded tensor object.
     * @param kernel The kernel tensor object.
     * @param out    The name to store the reconstructed tensor.
     * @param o      Output height and width.
     * @param pad    Padding applied equally.
     * @param stride Stride applied equally.
     * @return A {@link CuBridge} instance representing the reconstruction operation.
     */
    default CuBridge col2im2D(Tensor input, Tensor kernel, String out, int o, int pad, int stride) {
        return col2im2D(input, kernel, out, o, o, pad, pad, stride, stride);
    }

    /**
     * **Col2Im2DI — Immediate reconstruction on named tensors**
     *
     * Converts a column-expanded tensor back into its 2D form
     * and directly returns the reconstructed {@link Tensor}.
     *
     * @param input   The name of the column-expanded tensor.
     * @param kernel  The name of the kernel tensor.
     * @param oH      Output height.
     * @param oW      Output width.
     * @param padH    Padding applied during im2col.
     * @param padW    Padding applied during im2col.
     * @param strideH Stride for height.
     * @param strideW Stride for width.
     * @return A {@link Tensor} representing the reconstructed 2D output.
     */
    default Tensor col2im2DI(String input, String kernel, int oH, int oW, int padH, int padW, int strideH, int strideW) {
        String oName = genRandomNameImage();
        return col2im2D(input, kernel, oName, oH, oW, padH, padW, strideH, strideW).get(oName);
    }

    /**
     * **Col2Im2DI — Immediate version (single o, pad, stride)**
     *
     * Same as {@link #col2im2DI(String, String, int, int, int, int, int, int)} but applies
     * same values to both dimensions.
     *
     * @param input  The name of the column-expanded tensor.
     * @param kernel The name of the kernel tensor.
     * @param o      Output height and width.
     * @param pad    Padding applied equally.
     * @param stride Stride applied equally.
     * @return A {@link Tensor} representing the reconstructed output tensor.
     */
    default Tensor col2im2DI(String input, String kernel, int o, int pad, int stride) {
        return col2im2DI(input, kernel, o, o, pad, pad, stride, stride);
    }

    /**
     * **Col2Im2DI — Immediate Tensor-based reconstruction**
     *
     * Reconstructs the 2D tensor from column data using Tensor inputs and directly returns it.
     *
     * @param input   The column-expanded tensor object.
     * @param kernel  The kernel tensor object.
     * @param oH      Output height.
     * @param oW      Output width.
     * @param padH    Padding for height.
     * @param padW    Padding for width.
     * @param strideH Stride for height.
     * @param strideW Stride for width.
     * @return A {@link Tensor} representing the reconstructed output tensor.
     */
    default Tensor col2im2DI(Tensor input, Tensor kernel, int oH, int oW, int padH, int padW, int strideH, int strideW) {
        String iName = genRandomNameImage(); CuBridge.getInstance().put(input, iName);
        String kName = genRandomNameImage(); CuBridge.getInstance().put(kernel, kName);
        return col2im2DI(iName, kName, oH, oW, padH, padW, strideH, strideW);
    }

    /**
     * **Col2Im2DI — Immediate Tensor-based reconstruction (equal o, pad, stride)**
     *
     * Performs col2im reconstruction with same height and width for all parameters.
     *
     * @param input  The column-expanded tensor object.
     * @param kernel The kernel tensor object.
     * @param o      Output height and width.
     * @param pad    Padding applied equally.
     * @param stride Stride applied equally.
     * @return A {@link Tensor} representing the reconstructed 2D output.
     */
    default Tensor col2im2DI(Tensor input, Tensor kernel, int o, int pad, int stride) {
        return col2im2DI(input, kernel, o, o, pad, pad, stride, stride);
    }

}
