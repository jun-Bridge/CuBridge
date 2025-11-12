package CuBridge;

import java.util.UUID;

public interface AudioOps {

    private String genRandomNameAudio() {
        return "AudioOps_TMP_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    /**
     * **Pre-Emphasis (Low) — Apply low-frequency emphasis**
     *
     * Applies a low-frequency emphasis filter to the input tensor {@code a} and
     * stores the result in {@code out}. This variant uses a fixed coefficient of
     * {@code alpha = 0.15}, emphasizing low-band energy (0–1 kHz) such as vowels
     * or bass components.
     *
     * @param a   The name of the input tensor (e.g., waveform signal).
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(String, String, float)
     */
    default CuBridge preEmphasisLow(String a, String out) {
        return preEmphasis(a, out, 0.15f);
    }

    /**
     * **Pre-Emphasis (Low) — Apply low-frequency emphasis to a tensor input**
     *
     * Applies a low-frequency emphasis filter ({@code alpha = 0.15}) to the given
     * {@link Tensor} input and stores the result in {@code out}.
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasisLow(String, String)
     */
    default CuBridge preEmphasisLow(Tensor a, String out) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return preEmphasisLow(aName, out);
    }

    /**
     * **Pre-Emphasis (Low, Immediate) — Immediate low-frequency emphasis**
     *
     * Immediately applies a low-frequency emphasis filter ({@code alpha = 0.15})
     * to the named tensor and returns the result as a {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasisLow(String, String)
     */
    default Tensor preEmphasisLowI(String a) {
        String oName = genRandomNameAudio();
        return preEmphasisLow(a, oName).get(oName);
    }

    /**
     * **Pre-Emphasis (Low, Immediate) — Apply low-frequency emphasis to a tensor input**
     *
     * Immediately applies a low-frequency emphasis filter ({@code alpha = 0.15})
     * to the given input {@link Tensor} and returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasisLow(Tensor, String)
     */
    default Tensor preEmphasisLowI(Tensor a) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return preEmphasisLow(aName, oName).get(oName);
    }


    /**
     * **Pre-Emphasis (Mid) — Apply mid-frequency emphasis**
     *
     * Applies a mid-frequency emphasis filter to the input tensor {@code a} and
     * stores the result in {@code out}. This variant uses a fixed coefficient of
     * {@code alpha = 0.50}, emphasizing the mid-band range (1–4 kHz) to enhance
     * clarity and presence of vocals or instruments.
     *
     * @param a   The name of the input tensor (e.g., waveform signal).
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(String, String, float)
     */
    default CuBridge preEmphasisMid(String a, String out) {
        return preEmphasis(a, out, 0.50f);
    }

    /**
     * **Pre-Emphasis (Mid) — Apply mid-frequency emphasis to a tensor input**
     *
     * Applies a mid-frequency emphasis filter ({@code alpha = 0.50}) to the given
     * {@link Tensor} input and stores the result in {@code out}.
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasisMid(String, String)
     */
    default CuBridge preEmphasisMid(Tensor a, String out) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return preEmphasisMid(aName, out);
    }

    /**
     * **Pre-Emphasis (Mid, Immediate) — Immediate mid-frequency emphasis**
     *
     * Immediately applies a mid-frequency emphasis filter ({@code alpha = 0.50})
     * to the named tensor and returns the result as a {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasisMid(String, String)
     */
    default Tensor preEmphasisMidI(String a) {
        String oName = genRandomNameAudio();
        return preEmphasisMid(a, oName).get(oName);
    }

    /**
     * **Pre-Emphasis (Mid, Immediate) — Apply mid-frequency emphasis to a tensor input**
     *
     * Immediately applies a mid-frequency emphasis filter ({@code alpha = 0.50})
     * to the given input {@link Tensor} and returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasisMid(Tensor, String)
     */
    default Tensor preEmphasisMidI(Tensor a) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return preEmphasisMid(aName, oName).get(oName);
    }


    /**
     * **Pre-Emphasis (All) — Apply full-band emphasis**
     *
     * Applies a full-band emphasis (broadband flattening) filter to the input tensor {@code a}
     * and stores the result in {@code out}. This variant uses a fixed coefficient of
     * {@code alpha = 0.95}, uniformly adjusting energy across all frequency bands
     * to achieve a balanced spectral response.
     *
     * @param a   The name of the input tensor (e.g., waveform signal).
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(String, String, float)
     */
    default CuBridge preEmphasisAll(String a, String out) {
        return preEmphasis(a, out, 0.95f);
    }

    /**
     * **Pre-Emphasis (All) — Apply full-band emphasis to a tensor input**
     *
     * Applies a full-band emphasis filter ({@code alpha = 0.95}) to the given
     * {@link Tensor} input and stores the result in {@code out}.
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasisAll(String, String)
     */
    default CuBridge preEmphasisAll(Tensor a, String out) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return preEmphasisAll(aName, out);
    }

    /**
     * **Pre-Emphasis (All, Immediate) — Immediate full-band emphasis**
     *
     * Immediately applies a full-band emphasis filter ({@code alpha = 0.95})
     * to the named tensor and returns the result as a {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the emphasized (flattened) signal.
     * @see #preEmphasisAll(String, String)
     */
    default Tensor preEmphasisAllI(String a) {
        String oName = genRandomNameAudio();
        return preEmphasisAll(a, oName).get(oName);
    }

    /**
     * **Pre-Emphasis (All, Immediate) — Apply full-band emphasis to a tensor input**
     *
     * Immediately applies a full-band emphasis filter ({@code alpha = 0.95})
     * to the given input {@link Tensor} and returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the emphasized (flattened) signal.
     * @see #preEmphasisAll(Tensor, String)
     */
    default Tensor preEmphasisAllI(Tensor a) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return preEmphasisAll(aName, oName).get(oName);
    }


    /**
     * **Pre-Emphasis (High) — Apply high-frequency emphasis**
     *
     * Applies a high-frequency emphasis filter to the input tensor {@code a} and
     * stores the result in {@code out}. This variant uses a fixed coefficient of
     * {@code alpha = 0.97}, which is the standard pre-emphasis setting in most
     * speech and audio processing pipelines. It enhances high-frequency energy
     * (typically above 4 kHz) to improve clarity and compensate for natural
     * high-end roll-off during recording.
     *
     * @param a   The name of the input tensor (e.g., waveform signal).
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(String, String, float)
     */
    default CuBridge preEmphasisHigh(String a, String out) {
        return preEmphasis(a, out, 0.97f);
    }

    /**
     * **Pre-Emphasis (High) — Apply high-frequency emphasis to a tensor input**
     *
     * Applies a high-frequency emphasis filter ({@code alpha = 0.97}) to the given
     * {@link Tensor} input and stores the result in {@code out}.
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasisHigh(String, String)
     */
    default CuBridge preEmphasisHigh(Tensor a, String out) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return preEmphasisHigh(aName, out);
    }

    /**
     * **Pre-Emphasis (High, Immediate) — Immediate high-frequency emphasis**
     *
     * Immediately applies a high-frequency emphasis filter ({@code alpha = 0.97})
     * to the named tensor and returns the result as a {@link Tensor}.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasisHigh(String, String)
     */
    default Tensor preEmphasisHighI(String a) {
        String oName = genRandomNameAudio();
        return preEmphasisHigh(a, oName).get(oName);
    }

    /**
     * **Pre-Emphasis (High, Immediate) — Apply high-frequency emphasis to a tensor input**
     *
     * Immediately applies a high-frequency emphasis filter ({@code alpha = 0.97})
     * to the given input {@link Tensor} and returns the resulting {@link Tensor}.
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasisHigh(Tensor, String)
     */
    default Tensor preEmphasisHighI(Tensor a) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return preEmphasisHigh(aName, oName).get(oName);
    }


    /**
     * **Pre-Emphasis — Apply high-frequency emphasis**
     *
     * Applies a standard high-frequency pre-emphasis filter to the input tensor {@code a}
     * and stores the result in {@code out}. This operation compensates for natural
     * high-end roll-off during recording by boosting high-frequency components.
     * <p>
     * The default coefficient is {@code alpha = 0.97}, which corresponds to the
     * commonly used value in speech and audio preprocessing pipelines.
     * </p>
     *
     * @param a   The name of the input tensor (e.g., waveform signal).
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(String, String, float)
     */
    default CuBridge preEmphasis(String a, String out) {
        return preEmphasis(a, out, 0.97f);
    }

    /**
     * **Pre-Emphasis — Apply high-frequency emphasis with custom alpha**
     *
     * Applies a pre-emphasis filter to the input tensor {@code a} using a specified
     * coefficient {@code alpha}. The coefficient determines the degree of high-frequency
     * amplification, where higher values (closer to 1.0) produce stronger emphasis.
     * <p>
     * This operation enhances clarity by reducing spectral tilt and increasing
     * the relative magnitude of higher-frequency content.
     * </p>
     *
     * @param a     The name of the input tensor.
     * @param out   The name to store the emphasized output tensor.
     * @param alpha The pre-emphasis coefficient (commonly 0.95–0.98).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge preEmphasis(String a, String out, float alpha) {
        if (CuBridgeJNI.preEmphasis(a, out, alpha))
            return CuBridge.getInstance();
        else
            System.err.println("Error | preEmphasis | " + a + " | " + out + " | " + alpha);
        return null;
    }

    /**
     * **Pre-Emphasis — Apply high-frequency emphasis to a tensor input**
     *
     * Applies a default pre-emphasis filter ({@code alpha = 0.97}) to the given
     * input {@link Tensor} and stores the emphasized result in {@code out}.
     * This version automatically registers the tensor in the internal queue.
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the emphasized output tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(String, String)
     */
    default CuBridge preEmphasis(Tensor a, String out) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return preEmphasis(aName, out, 0.97f);
    }

    /**
     * **Pre-Emphasis — Apply high-frequency emphasis to a tensor input with custom alpha**
     *
     * Applies a pre-emphasis filter with the specified {@code alpha} coefficient
     * to the given input {@link Tensor} and stores the result in {@code out}.
     * The {@code alpha} parameter controls the slope of the spectral correction.
     *
     * @param a     The input {@link Tensor}.
     * @param out   The name to store the emphasized output tensor.
     * @param alpha The pre-emphasis coefficient.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #preEmphasis(Tensor, String)
     */
    default CuBridge preEmphasis(Tensor a, String out, float alpha) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return preEmphasis(aName, out, alpha);
    }

    /**
     * **Pre-Emphasis (Immediate) — Immediate high-frequency emphasis**
     *
     * Immediately applies a default pre-emphasis filter ({@code alpha = 0.97})
     * to the named tensor and returns the result as a {@link Tensor}.
     * This version is used for one-shot filtering without queue registration.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasis(String, String)
     */
    default Tensor preEmphasisI(String a) {
        String oName = genRandomNameAudio();
        return preEmphasis(a, oName, 0.97f).get(oName);
    }

    /**
     * **Pre-Emphasis (Immediate) — Immediate high-frequency emphasis with custom alpha**
     *
     * Immediately applies a pre-emphasis filter with the given {@code alpha}
     * to the named tensor and returns the result as a {@link Tensor}.
     * This is equivalent to {@link #preEmphasis(String, String, float)} but returns
     * the emphasized signal directly.
     *
     * @param a     The name of the input tensor.
     * @param alpha The pre-emphasis coefficient (commonly 0.95–0.98).
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasis(String, String, float)
     */
    default Tensor preEmphasisI(String a, float alpha) {
        String oName = genRandomNameAudio();
        return preEmphasis(a, oName, alpha).get(oName);
    }

    /**
     * **Pre-Emphasis (Immediate) — Apply pre-emphasis to a tensor input**
     *
     * Immediately applies a default pre-emphasis filter ({@code alpha = 0.97})
     * to the given input {@link Tensor} and returns the resulting {@link Tensor}.
     * This version performs all operations in-memory and does not modify the queue.
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasis(Tensor, String)
     */
    default Tensor preEmphasisI(Tensor a) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return preEmphasis(aName, oName, 0.97f).get(oName);
    }

    /**
     * **Pre-Emphasis (Immediate) — Apply pre-emphasis to a tensor input with custom alpha**
     *
     * Immediately applies a pre-emphasis filter with the specified {@code alpha}
     * to the given input {@link Tensor} and returns the resulting {@link Tensor}.
     * The {@code alpha} parameter controls how strongly high frequencies are emphasized.
     *
     * @param a     The input {@link Tensor}.
     * @param alpha The pre-emphasis coefficient.
     * @return A {@link Tensor} containing the emphasized signal.
     * @see #preEmphasis(Tensor, String, float)
     */
    default Tensor preEmphasisI(Tensor a, float alpha) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return preEmphasis(aName, oName, alpha).get(oName);
    }


    /**
     * **Apply Window — Apply a window function to a tensor**
     *
     * Applies a window function to the input tensor {@code a} using the specified window tensor name.
     * <p>
     * The window is applied in frames according to {@code hopSize}, and the resulting
     * windowed segments are stored in the output tensor {@code out}.
     * </p>
     * <p>
     * This function is typically used in STFT or spectrogram preprocessing,
     * where a signal is segmented and multiplied by a window function before FFT.
     * </p>
     *
     * @param a          The name of the input tensor (usually a waveform or 1D signal).
     * @param out        The name to store the output tensor.
     * @param windowName The name of the window tensor to apply (e.g., "hann", "hamming").
     * @param hopSize    The hop length between consecutive windows.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge applyWindow(String a, String out, String windowName, int hopSize){
        if (CuBridgeJNI.applyWindow(a, out, windowName, hopSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | applyWindow | " + a + " | " + out + " | " + windowName + " | " + hopSize);
        return null;
    }

    /**
     * **Apply Window — Apply a window to a tensor input**
     *
     * Applies a window function to the given input {@link Tensor} {@code a},
     * using the specified {@code windowName}.
     * <p>
     * Internally assigns a random name to the input tensor before applying the window.
     * </p>
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the output tensor.
     * @param windowName The name of the window tensor to apply.
     * @param hopSize    The hop length between consecutive windows.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #applyWindow(String, String, String, int)
     */
    default CuBridge applyWindow(Tensor a, String out, String windowName, int hopSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return applyWindow(aName, out, windowName, hopSize);
    }

    /**
     * **Apply Window — Apply a tensor-based window**
     *
     * Applies a user-provided {@link Tensor} window to the input tensor {@code a}.
     * <p>
     * Internally assigns a random name to the window tensor before applying it.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param out        The name to store the output tensor.
     * @param windowName The {@link Tensor} containing the window values.
     * @param hopSize    The hop length between consecutive windows.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #applyWindow(String, String, String, int)
     */
    default CuBridge applyWindow(String a, String out, Tensor windowName, int hopSize){
        String wName = genRandomNameAudio(); CuBridge.getInstance().put(windowName, wName);
        return applyWindow(a, out, wName, hopSize);
    }

    /**
     * **Apply Window — Tensor input and tensor window**
     *
     * Applies a tensor-based window to a tensor input, both provided as {@link Tensor} objects.
     * <p>
     * Internally assigns random temporary names to both the input and window tensors.
     * </p>
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the output tensor.
     * @param windowName The window {@link Tensor}.
     * @param hopSize    The hop length between consecutive windows.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #applyWindow(String, String, String, int)
     */
    default CuBridge applyWindow(Tensor a, String out, Tensor windowName, int hopSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String wName = genRandomNameAudio(); CuBridge.getInstance().put(windowName, wName);
        return applyWindow(aName, out, wName, hopSize);
    }

    /**
     * **Apply Window (Immediate) — Immediate window application**
     *
     * Immediately applies a named window to a tensor already stored in memory.
     * <p>
     * Assigns a random internal name for the output tensor and returns the resulting {@link Tensor}.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param windowName The name of the window tensor to apply.
     * @param hopSize    The hop length between consecutive windows.
     * @return A {@link Tensor} containing the windowed result.
     * @see #applyWindow(String, String, String, int)
     */
    default Tensor applyWindowI(String a, String windowName, int hopSize){
        String oName = genRandomNameAudio();
        return applyWindow(a, oName, windowName, hopSize).get(oName);
    }

    /**
     * **Apply Window (Immediate) — Immediate window application with tensor input**
     *
     * Immediately applies a named window to a given {@link Tensor} input,
     * assigns a random internal name for both input and output, and returns the result.
     *
     * @param a          The input {@link Tensor}.
     * @param windowName The name of the window tensor to apply.
     * @param hopSize    The hop length between consecutive windows.
     * @return A {@link Tensor} containing the windowed result.
     * @see #applyWindow(Tensor, String, String, int)
     */
    default Tensor applyWindowI(Tensor a, String windowName, int hopSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return applyWindow(aName, oName, windowName, hopSize).get(oName);
    }

    /**
     * **Apply Window (Immediate) — Immediate tensor-based window application**
     *
     * Immediately applies a user-provided {@link Tensor} window to a named input tensor,
     * assigns a random internal name for both the window and output, and returns the result.
     *
     * @param a          The name of the input tensor.
     * @param windowName The {@link Tensor} containing the window values.
     * @param hopSize    The hop length between consecutive windows.
     * @return A {@link Tensor} containing the windowed result.
     * @see #applyWindow(String, String, Tensor, int)
     */
    default Tensor applyWindowI(String a, Tensor windowName, int hopSize){
        String wName = genRandomNameAudio(); CuBridge.getInstance().put(windowName, wName);
        String oName = genRandomNameAudio();
        return applyWindow(a, oName, wName, hopSize).get(oName);
    }

    /**
     * **Apply Window (Immediate) — Immediate tensor-to-tensor window application**
     *
     * Immediately applies a user-provided {@link Tensor} window to a given {@link Tensor} input.
     * <p>
     * Assigns random internal names for both the input and window tensors,
     * performs the operation, and returns the windowed output tensor.
     * </p>
     *
     * @param a          The input {@link Tensor}.
     * @param windowName The {@link Tensor} containing the window values.
     * @param hopSize    The hop length between consecutive windows.
     * @return A {@link Tensor} containing the windowed result.
     * @see #applyWindow(Tensor, String, Tensor, int)
     */
    default Tensor applyWindowI(Tensor a, Tensor windowName, int hopSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String wName = genRandomNameAudio(); CuBridge.getInstance().put(windowName, wName);
        String oName = genRandomNameAudio();
        return applyWindow(aName, oName, wName, hopSize).get(oName);
    }


    /**
     * **Apply Filter — Apply a filter bank to a tensor**
     *
     * Applies a precomputed filter bank tensor (e.g., Mel, Bark, ERB, Chroma) to the input tensor {@code a}.
     * <p>
     * This function performs a matrix multiplication or projection of {@code a} through the specified filter,
     * storing the filtered result in {@code out}.
     * </p>
     * <p>
     * Commonly used in spectrogram feature extraction, such as Mel-spectrogram computation or
     * auditory-scale transformations.
     * </p>
     *
     * @param a          The name of the input tensor (usually a magnitude or power spectrum).
     * @param out        The name to store the filtered output tensor.
     * @param filterName The name of the filter tensor to apply (e.g., "mel64", "bark24", "chroma12").
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge applyFilter(String a, String out, String filterName){
        if (CuBridgeJNI.applyFilter(a, out, filterName))
            return CuBridge.getInstance();
        else
            System.err.println("Error | applyFilter | " + a + " | " + out + " | " + filterName);
        return null;
    }

    /**
     * **Apply Filter — Apply a named filter to a tensor input**
     *
     * Applies the specified named filter to the provided input {@link Tensor}.
     * <p>
     * Internally assigns a random name to the input tensor before applying the filter.
     * </p>
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the filtered output tensor.
     * @param filterName The name of the filter tensor to apply.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #applyFilter(String, String, String)
     */
    default CuBridge applyFilter(Tensor a, String out, String filterName){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return applyFilter(aName, out, filterName);
    }

    /**
     * **Apply Filter — Apply a tensor-based filter to a named tensor**
     *
     * Applies a filter provided as a {@link Tensor} object to the input tensor {@code a}.
     * <p>
     * Internally assigns a random name to the filter tensor before applying it.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param out        The name to store the filtered output tensor.
     * @param filter     The {@link Tensor} containing the filter bank.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #applyFilter(String, String, String)
     */
    default CuBridge applyFilter(String a, String out, Tensor filter){
        String fName = genRandomNameAudio(); CuBridge.getInstance().put(filter, fName);
        return applyFilter(a, out, fName);
    }

    /**
     * **Apply Filter — Apply a tensor-based filter to a tensor input**
     *
     * Applies a user-provided filter tensor to a tensor input, both supplied as {@link Tensor} objects.
     * <p>
     * Internally assigns random temporary names to both tensors before executing the operation.
     * </p>
     *
     * @param a       The input {@link Tensor}.
     * @param out     The name to store the filtered output tensor.
     * @param filter  The {@link Tensor} containing the filter bank.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #applyFilter(String, String, String)
     */
    default CuBridge applyFilter(Tensor a, String out, Tensor filter){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String fName = genRandomNameAudio(); CuBridge.getInstance().put(filter, fName);
        return applyFilter(aName, out, fName);
    }

    /**
     * **Apply Filter (Immediate) — Immediate filter application**
     *
     * Immediately applies a named filter to a tensor already stored in memory.
     * <p>
     * Assigns a random internal name for the output tensor and returns the resulting {@link Tensor}.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param filterName The name of the filter tensor to apply.
     * @return A {@link Tensor} containing the filtered output.
     * @see #applyFilter(String, String, String)
     */
    default Tensor applyFilterI(String a, String filterName){
        String oName = genRandomNameAudio();
        return applyFilter(a, oName, filterName).get(oName);
    }

    /**
     * **Apply Filter (Immediate) — Immediate filter application with tensor input**
     *
     * Immediately applies a named filter to a given {@link Tensor} input.
     * <p>
     * Assigns random names for the input and output tensors internally.
     * </p>
     *
     * @param a          The input {@link Tensor}.
     * @param filterName The name of the filter tensor to apply.
     * @return A {@link Tensor} containing the filtered output.
     * @see #applyFilter(Tensor, String, String)
     */
    default Tensor applyFilterI(Tensor a, String filterName){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return applyFilter(aName, oName, filterName).get(oName);
    }

    /**
     * **Apply Filter (Immediate) — Immediate tensor-based filter application**
     *
     * Immediately applies a user-provided {@link Tensor} filter to a named input tensor.
     * <p>
     * Assigns a random internal name for both the filter and output tensors.
     * </p>
     *
     * @param a       The name of the input tensor.
     * @param filter  The {@link Tensor} containing the filter bank.
     * @return A {@link Tensor} containing the filtered output.
     * @see #applyFilter(String, String, Tensor)
     */
    default Tensor applyFilterI(String a, Tensor filter){
        String fName = genRandomNameAudio(); CuBridge.getInstance().put(filter, fName);
        String oName = genRandomNameAudio();
        return applyFilter(a, oName, fName).get(oName);
    }

    /**
     * **Apply Filter (Immediate) — Immediate tensor-to-tensor filter application**
     *
     * Immediately applies a user-provided {@link Tensor} filter to a given {@link Tensor} input.
     * <p>
     * Assigns random internal names for both input and filter tensors,
     * performs the filtering operation, and returns the resulting {@link Tensor}.
     * </p>
     *
     * @param a       The input {@link Tensor}.
     * @param filter  The {@link Tensor} containing the filter bank.
     * @return A {@link Tensor} containing the filtered output.
     * @see #applyFilter(Tensor, String, Tensor)
     */
    default Tensor applyFilterI(Tensor a, Tensor filter){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String fName = genRandomNameAudio(); CuBridge.getInstance().put(filter, fName);
        String oName = genRandomNameAudio();
        return applyFilter(aName, oName, fName).get(oName);
    }


    /**
     * **FFT — Compute Fast Fourier Transform**
     *
     * Computes the Fast Fourier Transform (FFT) of the input tensor {@code a}
     * using the specified {@code fftSize}, and stores the resulting complex-valued spectrum in {@code out}.
     * <p>
     * The output tensor contains two channels representing the real and imaginary components
     * of the complex FFT result.
     * </p>
     *
     * @param a        The name of the input tensor (e.g., time-domain signal).
     * @param out      The name to store the resulting FFT tensor.
     * @param fftSize  The FFT size (e.g., 512, 1024, 2048).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge fft(String a, String out, int fftSize){
        if (CuBridgeJNI.fft(a, out, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | fft | " + a + " | " + out + " | " + fftSize);
        return null;
    }

    /**
     * **FFT — Apply Fast Fourier Transform to a tensor input**
     *
     * Computes the FFT of the given {@link Tensor} input using the specified {@code fftSize},
     * and stores the resulting spectrum in {@code out}.
     * <p>
     * The output tensor contains two channels: one for the real part and one for the imaginary part.
     * </p>
     *
     * @param a        The input {@link Tensor}.
     * @param out      The name to store the resulting FFT tensor.
     * @param fftSize  The FFT size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #fft(String, String, int)
     */
    default CuBridge fft(Tensor a, String out, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return fft(aName, out, fftSize);
    }

    /**
     * **FFT (Immediate) — Immediate Fast Fourier Transform**
     *
     * Immediately computes the FFT of the named tensor using the specified {@code fftSize}
     * and returns the resulting frequency-domain {@link Tensor}.
     * <p>
     * The output tensor contains two channels representing the real and imaginary components.
     * </p>
     *
     * @param a        The name of the input tensor.
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the FFT-transformed output.
     * @see #fft(String, String, int)
     */
    default Tensor fftI(String a, int fftSize){
        String oName = genRandomNameAudio();
        return fft(a, oName, fftSize).get(oName);
    }

    /**
     * **FFT (Immediate) — Apply Fast Fourier Transform to a tensor input**
     *
     * Immediately computes the FFT of the given {@link Tensor} input using the specified {@code fftSize}
     * and returns the resulting {@link Tensor}.
     * <p>
     * The output tensor contains two channels: real and imaginary.
     * </p>
     *
     * @param a        The input {@link Tensor}.
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the FFT-transformed output.
     * @see #fft(Tensor, String, int)
     */
    default Tensor fftI(Tensor a, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return fft(aName, oName, fftSize).get(oName);
    }


    /**
     * **RFFT — Compute Real-input Fast Fourier Transform**
     *
     * Computes the Real Fast Fourier Transform (RFFT) of the input tensor {@code a}
     * using the specified {@code fftSize}, and stores the resulting real-valued spectrum in {@code out}.
     * <p>
     * Unlike the standard {@link #fft(String, String, int)} function, CuBridge.getInstance() operation assumes
     * that the input tensor is purely real and outputs only one channel containing the real-valued frequency components.
     * </p>
     *
     * @param a        The name of the input tensor (real-valued signal).
     * @param out      The name to store the resulting real-valued spectrum tensor.
     * @param fftSize  The FFT size (e.g., 512, 1024, 2048).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge rfft(String a, String out, int fftSize){
        if (CuBridgeJNI.rfft(a, out, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | rfft | " + a + " | " + out + " | " + fftSize);
        return null;
    }

    /**
     * **RFFT — Apply Real-input FFT to a tensor input**
     *
     * Computes the RFFT of the given {@link Tensor} input using the specified {@code fftSize},
     * and stores the resulting single-channel real-valued spectrum in {@code out}.
     *
     * @param a        The input {@link Tensor} (real-valued).
     * @param out      The name to store the resulting real-valued spectrum tensor.
     * @param fftSize  The FFT size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #rfft(String, String, int)
     */
    default CuBridge rfft(Tensor a, String out, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return rfft(aName, out, fftSize);
    }

    /**
     * **RFFT (Immediate) — Immediate Real-input FFT**
     *
     * Immediately computes the RFFT of the named tensor using the specified {@code fftSize}
     * and returns the resulting single-channel real-valued {@link Tensor}.
     *
     * @param a        The name of the input tensor.
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the real-valued frequency spectrum.
     * @see #rfft(String, String, int)
     */
    default Tensor rfftI(String a, int fftSize){
        String oName = genRandomNameAudio();
        return rfft(a, oName, fftSize).get(oName);
    }

    /**
     * **RFFT (Immediate) — Apply Real-input FFT to a tensor input**
     *
     * Immediately computes the RFFT of the given {@link Tensor} input using the specified {@code fftSize}
     * and returns the resulting single-channel real-valued {@link Tensor}.
     *
     * @param a        The input {@link Tensor}.
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the real-valued frequency spectrum.
     * @see #rfft(Tensor, String, int)
     */
    default Tensor rfftI(Tensor a, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return rfft(aName, oName, fftSize).get(oName);
    }


    /**
     * **IFFT — Compute Inverse Fast Fourier Transform**
     *
     * Computes the Inverse Fast Fourier Transform (IFFT) of the complex-valued input tensor {@code a}
     * using the specified {@code fftSize}, and stores the resulting time-domain signal in {@code out}.
     * <p>
     * The input tensor must contain two channels representing the real and imaginary components
     * of the frequency-domain spectrum produced by {@link #fft(String, String, int)}.
     * The output is a single-channel real-valued waveform reconstructed from the complex spectrum.
     * </p>
     *
     * @param a        The name of the input tensor (2-channel complex spectrum).
     * @param out      The name to store the reconstructed real-valued signal.
     * @param fftSize  The FFT size (e.g., 512, 1024, 2048).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge ifft(String a, String out, int fftSize){
        if (CuBridgeJNI.ifft(a, out, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | ifft | " + a + " | " + out + " | " + fftSize);
        return null;
    }

    /**
     * **IFFT — Apply Inverse FFT to a tensor input**
     *
     * Computes the IFFT of the given complex-valued {@link Tensor} input using the specified {@code fftSize},
     * and stores the resulting real-valued signal in {@code out}.
     * <p>
     * The input tensor must contain two channels: one for the real part and one for the imaginary part.
     * </p>
     *
     * @param a        The input {@link Tensor} (2-channel complex spectrum).
     * @param out      The name to store the reconstructed real-valued signal.
     * @param fftSize  The FFT size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #ifft(String, String, int)
     */
    default CuBridge ifft(Tensor a, String out, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return ifft(aName, out, fftSize);
    }

    /**
     * **IFFT (Immediate) — Immediate Inverse Fast Fourier Transform**
     *
     * Immediately performs the IFFT on a named 2-channel complex tensor using the specified {@code fftSize},
     * reconstructing the real-valued time-domain waveform and returning it as a {@link Tensor}.
     *
     * @param a        The name of the input tensor (2-channel complex spectrum).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the reconstructed real-valued signal.
     * @see #ifft(String, String, int)
     */
    default Tensor ifftI(String a, int fftSize){
        String oName = genRandomNameAudio();
        return ifft(a, oName, fftSize).get(oName);
    }

    /**
     * **IFFT (Immediate) — Apply Inverse FFT to a tensor input**
     *
     * Immediately performs the IFFT on the given complex-valued {@link Tensor} input using the specified {@code fftSize},
     * reconstructing the real-valued time-domain waveform and returning it as a {@link Tensor}.
     * <p>
     * The input tensor must contain two channels representing the real and imaginary components.
     * </p>
     *
     * @param a        The input {@link Tensor} (2-channel complex spectrum).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the reconstructed real-valued signal.
     * @see #ifft(Tensor, String, int)
     */
    default Tensor ifftI(Tensor a, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return ifft(aName, oName, fftSize).get(oName);
    }


    /**
     * **PowFFT — Compute power spectrum from windowed signal**
     *
     * Computes the power spectrum of the input tensor {@code a} using the specified {@code fftSize},
     * and stores the resulting single-channel real-valued power tensor in {@code out}.
     * <p>
     * Internally performs an FFT on the windowed real-valued signal and calculates
     * {@code power = real^2 + imag^2} for each frequency bin.
     * The output contains the power spectrum representing the energy distribution across frequencies.
     * </p>
     *
     * @param a        The name of the input tensor (windowed real signal).
     * @param out      The name to store the resulting power spectrum tensor.
     * @param fftSize  The FFT size (e.g., 512, 1024, 2048).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge powfft(String a, String out, int fftSize){
        if (CuBridgeJNI.powfft(a, out, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | powfft | " + a + " | " + out + " | " + fftSize);
        return null;
    }

    /**
     * **PowFFT — Apply power spectrum computation to a tensor input**
     *
     * Computes the power spectrum from the given windowed real-valued {@link Tensor} input
     * using the specified {@code fftSize}, and stores the resulting power tensor in {@code out}.
     *
     * @param a        The input {@link Tensor} (windowed real signal).
     * @param out      The name to store the resulting power spectrum tensor.
     * @param fftSize  The FFT size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #powfft(String, String, int)
     */
    default CuBridge powfft(Tensor a, String out, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return powfft(aName, out, fftSize);
    }

    /**
     * **PowFFT (Immediate) — Immediate power spectrum computation**
     *
     * Immediately computes the power spectrum of the given windowed signal
     * using the specified {@code fftSize}, and returns the resulting real-valued {@link Tensor}.
     * <p>
     * Internally performs FFT and computes {@code real^2 + imag^2} for each frequency bin.
     * </p>
     *
     * @param a        The name of the input tensor (windowed real signal).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the computed power spectrum.
     * @see #powfft(String, String, int)
     */
    default Tensor powfftI(String a, int fftSize){
        String oName = genRandomNameAudio();
        return powfft(a, oName, fftSize).get(oName);
    }

    /**
     * **PowFFT (Immediate) — Apply power spectrum computation to a tensor input**
     *
     * Immediately computes the power spectrum from the given windowed {@link Tensor} input
     * using the specified {@code fftSize}, and returns the resulting single-channel {@link Tensor}.
     *
     * @param a        The input {@link Tensor} (windowed real signal).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the computed power spectrum.
     * @see #powfft(Tensor, String, int)
     */
    default Tensor powfftI(Tensor a, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return powfft(aName, oName, fftSize).get(oName);
    }


    /**
     * **MagFFT — Compute magnitude spectrum from windowed signal**
     *
     * Computes the magnitude spectrum of the input tensor {@code a} using the specified {@code fftSize},
     * and stores the resulting single-channel real-valued magnitude tensor in {@code out}.
     * <p>
     * Internally performs an FFT on the windowed real-valued signal and calculates
     * {@code magnitude = sqrt(real^2 + imag^2)} for each frequency bin.
     * The output contains the magnitude spectrum representing the amplitude of each frequency component.
     * </p>
     *
     * @param a        The name of the input tensor (windowed real signal).
     * @param out      The name to store the resulting magnitude spectrum tensor.
     * @param fftSize  The FFT size (e.g., 512, 1024, 2048).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge magfft(String a, String out, int fftSize){
        if (CuBridgeJNI.magfft(a, out, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | magfft | " + a + " | " + out + " | " + fftSize);
        return null;
    }

    /**
     * **MagFFT — Apply magnitude spectrum computation to a tensor input**
     *
     * Computes the magnitude spectrum from the given windowed real-valued {@link Tensor} input
     * using the specified {@code fftSize}, and stores the result in {@code out}.
     *
     * @param a        The input {@link Tensor} (windowed real signal).
     * @param out      The name to store the resulting magnitude spectrum tensor.
     * @param fftSize  The FFT size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #magfft(String, String, int)
     */
    default CuBridge magfft(Tensor a, String out, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return magfft(aName, out, fftSize);
    }

    /**
     * **MagFFT (Immediate) — Immediate magnitude spectrum computation**
     *
     * Immediately computes the magnitude spectrum of the given windowed signal
     * using the specified {@code fftSize}, and returns the resulting real-valued {@link Tensor}.
     * <p>
     * Internally performs FFT and computes {@code sqrt(real^2 + imag^2)} for each frequency bin.
     * </p>
     *
     * @param a        The name of the input tensor (windowed real signal).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the computed magnitude spectrum.
     * @see #magfft(String, String, int)
     */
    default Tensor magfftI(String a, int fftSize){
        String oName = genRandomNameAudio();
        return magfft(a, oName, fftSize).get(oName);
    }

    /**
     * **MagFFT (Immediate) — Apply magnitude spectrum computation to a tensor input**
     *
     * Immediately computes the magnitude spectrum from the given windowed {@link Tensor} input
     * using the specified {@code fftSize}, and returns the resulting single-channel {@link Tensor}.
     *
     * @param a        The input {@link Tensor} (windowed real signal).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the computed magnitude spectrum.
     * @see #magfft(Tensor, String, int)
     */
    default Tensor magfftI(Tensor a, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return magfft(aName, oName, fftSize).get(oName);
    }


    /**
     * **PhaseFFT — Compute phase spectrum from windowed signal**
     *
     * Computes the phase spectrum of the input tensor {@code a} using the specified {@code fftSize},
     * and stores the resulting single-channel real-valued phase tensor in {@code out}.
     * <p>
     * Internally performs an FFT on the windowed real-valued signal and calculates
     * the phase of each frequency bin as {@code atan2(imag, real)}.
     * The output phase values are in radians, ranging from {@code -π} to {@code +π}.
     * </p>
     *
     * @param a        The name of the input tensor (windowed real signal).
     * @param out      The name to store the resulting phase spectrum tensor.
     * @param fftSize  The FFT size (e.g., 512, 1024, 2048).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge phasefft(String a, String out, int fftSize){
        if (CuBridgeJNI.phasefft(a, out, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | phasefft | " + a + " | " + out + " | " + fftSize);
        return null;
    }

    /**
     * **PhaseFFT — Apply phase spectrum computation to a tensor input**
     *
     * Computes the phase spectrum from the given windowed real-valued {@link Tensor} input
     * using the specified {@code fftSize}, and stores the result in {@code out}.
     *
     * @param a        The input {@link Tensor} (windowed real signal).
     * @param out      The name to store the resulting phase spectrum tensor.
     * @param fftSize  The FFT size.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #phasefft(String, String, int)
     */
    default CuBridge phasefft(Tensor a, String out, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return phasefft(aName, out, fftSize);
    }

    /**
     * **PhaseFFT (Immediate) — Immediate phase spectrum computation**
     *
     * Immediately computes the phase spectrum of the given windowed signal
     * using the specified {@code fftSize}, and returns the resulting real-valued {@link Tensor}.
     * <p>
     * Internally performs FFT and extracts the phase in radians using {@code atan2(imag, real)}.
     * </p>
     *
     * @param a        The name of the input tensor (windowed real signal).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the computed phase spectrum in radians.
     * @see #phasefft(String, String, int)
     */
    default Tensor phasefftI(String a, int fftSize){
        String oName = genRandomNameAudio();
        return phasefft(a, oName, fftSize).get(oName);
    }

    /**
     * **PhaseFFT (Immediate) — Apply phase spectrum computation to a tensor input**
     *
     * Immediately computes the phase spectrum from the given windowed {@link Tensor} input
     * using the specified {@code fftSize}, and returns the resulting single-channel real-valued {@link Tensor}.
     *
     * @param a        The input {@link Tensor} (windowed real signal).
     * @param fftSize  The FFT size.
     * @return A {@link Tensor} containing the phase spectrum in radians.
     * @see #phasefft(Tensor, String, int)
     */
    default Tensor phasefftI(Tensor a, int fftSize){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return phasefft(aName, oName, fftSize).get(oName);
    }


    /**
     * **BoostLow — Apply low-frequency amplification (0–1 kHz)**
     *
     * Amplifies the low-frequency range of the input tensor {@code a}, typically
     * below 1 kHz, to enhance bass and vowel formants in speech or music.
     * <p>
     * This operation increases the amplitude of low-frequency components using
     * an FFT-based gain adjustment. The frequency range in Hz is internally converted
     * to FFT bin indices according to the provided {@code sampleRate}.
     * </p>
     *
     * @param a          The name of the input tensor (e.g., FFT magnitude or waveform).
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostLow(Tensor, String, int)
     * @see #boostLowI(String, int)
     * @see #boostLowI(Tensor, int)
     */
    default CuBridge boostLow(String a, String out, int sampleRate) {
        if (CuBridgeJNI.boost(a, out, sampleRate, 0f, 1000f, 1.4f))
            return CuBridge.getInstance();
        else
            System.err.println("Error | boostLow | " + a + " | " + out);
        return null;
    }

    /**
     * **BoostLow — Apply low-frequency amplification to a tensor input**
     *
     * Applies a low-frequency boost (0–1 kHz) to the given {@link Tensor} {@code a}
     * and stores the result in {@code out}. The specified {@code sampleRate} is
     * used to convert the cutoff frequencies from Hz to FFT bins.
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostLow(String, String, int)
     */
    default CuBridge boostLow(Tensor a, String out, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return boostLow(aName, out, sampleRate);
    }

    /**
     * **BoostLow (Immediate) — Immediate low-frequency amplification**
     *
     * Immediately applies a low-frequency boost (0–1 kHz) to the named tensor
     * {@code a} and returns the resulting boosted {@link Tensor}. This is a
     * one-shot operation that does not modify the internal queue.
     *
     * @param a          The name of the input tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} containing the amplified spectrum.
     * @see #boostLow(String, String, int)
     */
    default Tensor boostLowI(String a, int sampleRate) {
        String oName = genRandomNameAudio();
        return boostLow(a, oName, sampleRate).get(oName);
    }

    /**
     * **BoostLow (Immediate) — Apply low-frequency amplification to a tensor input**
     *
     * Immediately applies a low-frequency boost (0–1 kHz) to the given
     * {@link Tensor} {@code a} and returns the resulting boosted tensor.
     * This version performs all computations in-memory and does not alter
     * the processing queue.
     *
     * @param a          The input {@link Tensor}.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} containing the amplified spectrum.
     * @see #boostLow(Tensor, String, int)
     */
    default Tensor boostLowI(Tensor a, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return boostLow(aName, oName, sampleRate).get(oName);
    }


    /**
     * **BoostMid — Apply mid-frequency amplification (1–4 kHz)**
     *
     * Enhances the clarity and presence of speech by amplifying mid-range
     * frequencies between 1 kHz and 4 kHz in the input tensor {@code a}.
     * <p>
     * This operation increases the magnitude of the mid-frequency band
     * using FFT-based gain adjustment. The specified frequency range (Hz)
     * is automatically converted to FFT bin indices based on the provided
     * {@code sampleRate}.
     * </p>
     *
     * @param a          The name of the input tensor (e.g., FFT magnitude or waveform).
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostMid(Tensor, String, int)
     * @see #boostMidI(String, int)
     * @see #boostMidI(Tensor, int)
     */
    default CuBridge boostMid(String a, String out, int sampleRate) {
        if (CuBridgeJNI.boost(a, out, sampleRate, 1000f, 4000f, 1.2f))
            return CuBridge.getInstance();
        else
            System.err.println("Error | boostMid | " + a + " | " + out);
        return null;
    }

    /**
     * **BoostMid — Apply mid-frequency amplification to a tensor input**
     *
     * Applies a mid-frequency boost (1–4 kHz) to the given {@link Tensor} {@code a}
     * and stores the resulting amplified tensor in {@code out}. The specified
     * {@code sampleRate} determines how frequency values in Hz are mapped
     * to FFT bins.
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostMid(String, String, int)
     */
    default CuBridge boostMid(Tensor a, String out, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return boostMid(aName, out, sampleRate);
    }

    /**
     * **BoostMid (Immediate) — Immediate mid-frequency amplification**
     *
     * Immediately applies a mid-frequency boost (1–4 kHz) to the named input
     * tensor {@code a} and returns the resulting boosted {@link Tensor}.
     * <p>
     * This variant is intended for one-shot enhancement without modifying
     * the CuBridge queue.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} containing the enhanced mid-range frequencies.
     * @see #boostMid(String, String, int)
     */
    default Tensor boostMidI(String a, int sampleRate) {
        String oName = genRandomNameAudio();
        return boostMid(a, oName, sampleRate).get(oName);
    }

    /**
     * **BoostMid (Immediate) — Apply mid-frequency amplification to a tensor input**
     *
     * Immediately applies a mid-frequency boost (1–4 kHz) to the given
     * {@link Tensor} {@code a} and returns the resulting boosted tensor.
     * This method performs all operations in-memory and does not alter
     * the processing queue.
     *
     * @param a          The input {@link Tensor}.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} with enhanced mid-frequency content.
     * @see #boostMid(Tensor, String, int)
     */
    default Tensor boostMidI(Tensor a, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return boostMid(aName, oName, sampleRate).get(oName);
    }


    /**
     * **BoostHigh — Apply high-frequency amplification (4–8 kHz)**
     *
     * Increases the intensity and brightness of high-frequency components,
     * typically between 4 kHz and 8 kHz, to emphasize consonants and fine
     * spectral details. This is often used to improve clarity in speech
     * or crispness in musical audio.
     * <p>
     * The specified frequency range (Hz) is internally converted to FFT bin
     * indices using the provided {@code sampleRate}, and the gain is applied
     * within that band.
     * </p>
     *
     * @param a          The name of the input tensor (e.g., FFT magnitude or waveform).
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostHigh(Tensor, String, int)
     * @see #boostHighI(String, int)
     * @see #boostHighI(Tensor, int)
     */
    default CuBridge boostHigh(String a, String out, int sampleRate) {
        if (CuBridgeJNI.boost(a, out, sampleRate, 4000f, 8000f, 1.8f))
            return CuBridge.getInstance();
        else
            System.err.println("Error | boostHigh | " + a + " | " + out);
        return null;
    }

    /**
     * **BoostHigh — Apply high-frequency amplification to a tensor input**
     *
     * Applies a high-frequency boost (4–8 kHz) to the given {@link Tensor} {@code a}
     * and stores the resulting amplified tensor in {@code out}. The specified
     * {@code sampleRate} determines how the cutoff frequencies (in Hz) are mapped
     * to FFT bins.
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostHigh(String, String, int)
     */
    default CuBridge boostHigh(Tensor a, String out, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return boostHigh(aName, out, sampleRate);
    }

    /**
     * **BoostHigh (Immediate) — Immediate high-frequency amplification**
     *
     * Immediately applies a high-frequency boost (4–8 kHz) to the named tensor
     * {@code a} and returns the resulting boosted {@link Tensor}.
     * <p>
     * This operation is designed for one-shot enhancement and does not modify
     * the internal CuBridge processing queue.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} containing enhanced high-frequency detail.
     * @see #boostHigh(String, String, int)
     */
    default Tensor boostHighI(String a, int sampleRate) {
        String oName = genRandomNameAudio();
        return boostHigh(a, oName, sampleRate).get(oName);
    }

    /**
     * **BoostHigh (Immediate) — Apply high-frequency amplification to a tensor input**
     *
     * Immediately applies a high-frequency boost (4–8 kHz) to the given
     * {@link Tensor} {@code a} and returns the resulting boosted tensor.
     * This method performs the operation entirely in-memory and does not
     * alter the CuBridge processing queue.
     *
     * @param a          The input {@link Tensor}.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} with enhanced high-frequency content.
     * @see #boostHigh(Tensor, String, int)
     */
    default Tensor boostHighI(Tensor a, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return boostHigh(aName, oName, sampleRate).get(oName);
    }


    /**
     * **BoostBand — Apply custom frequency-band amplification**
     *
     * Amplifies a user-specified frequency range in the input tensor {@code a}
     * using the provided cutoff frequencies and gain factor. This allows
     * flexible control of which part of the spectrum is emphasized.
     * <p>
     * The given {@code lowCut} and {@code highCut} values are specified in Hertz (Hz)
     * and automatically converted to FFT bin indices based on the supplied
     * {@code sampleRate}. The {@code gain} determines the amount of amplification
     * applied to all bins within the selected range.
     * </p>
     *
     * @param a          The name of the input tensor (e.g., FFT magnitude or waveform).
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @param lowCut     The lower cutoff frequency (in Hz).
     * @param highCut    The upper cutoff frequency (in Hz).
     * @param gain       The amplification factor (e.g., {@code 1.2f} = +1.58 dB).
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostBand(Tensor, String, int, float, float, float)
     * @see #boostBandI(String, int, float, float, float)
     * @see #boostBandI(Tensor, int, float, float, float)
     */
    default CuBridge boostBand(String a, String out, int sampleRate, float lowCut, float highCut, float gain) {
        if (CuBridgeJNI.boost(a, out, sampleRate, lowCut, highCut, gain))
            return CuBridge.getInstance();
        else
            System.err.println("Error | boostBand | " + a + " | " + out);
        return null;
    }

    /**
     * **BoostBand — Apply custom frequency-band amplification to a tensor input**
     *
     * Applies a custom frequency-band boost to the given {@link Tensor} {@code a}
     * and stores the resulting amplified tensor in {@code out}. The cutoff
     * frequencies are specified in Hz and mapped to FFT bins using {@code sampleRate}.
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @param lowCut     The lower cutoff frequency (in Hz).
     * @param highCut    The upper cutoff frequency (in Hz).
     * @param gain       The amplification factor.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostBand(String, String, int, float, float, float)
     */
    default CuBridge boostBand(Tensor a, String out, int sampleRate, float lowCut, float highCut, float gain) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return boostBand(aName, out, sampleRate, lowCut, highCut, gain);
    }

    /**
     * **BoostBand (Immediate) — Immediate custom frequency-band amplification**
     *
     * Immediately applies a custom frequency boost to the named input tensor {@code a}
     * using the specified cutoff frequencies and gain, then returns the resulting
     * boosted {@link Tensor}.
     * <p>
     * This operation runs as a one-shot enhancement and does not alter the
     * internal CuBridge processing queue.
     * </p>
     *
     * @param a          The name of the input tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @param lowCut     The lower cutoff frequency (in Hz).
     * @param highCut    The upper cutoff frequency (in Hz).
     * @param gain       The amplification factor.
     * @return A boosted {@link Tensor} containing the enhanced frequency band.
     * @see #boostBand(String, String, int, float, float, float)
     */
    default Tensor boostBandI(String a, int sampleRate, float lowCut, float highCut, float gain) {
        String oName = genRandomNameAudio();
        return boostBand(a, oName, sampleRate, lowCut, highCut, gain).get(oName);
    }

    /**
     * **BoostBand (Immediate) — Apply custom frequency-band amplification to a tensor input**
     *
     * Immediately applies a frequency-band boost with the specified parameters
     * to the given {@link Tensor} {@code a} and returns the resulting boosted tensor.
     * This version executes entirely in-memory and does not modify the CuBridge queue.
     *
     * @param a          The input {@link Tensor}.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @param lowCut     The lower cutoff frequency (in Hz).
     * @param highCut    The upper cutoff frequency (in Hz).
     * @param gain       The amplification factor.
     * @return A boosted {@link Tensor} emphasizing the selected frequency band.
     * @see #boostBand(Tensor, String, int, float, float, float)
     */
    default Tensor boostBandI(Tensor a, int sampleRate, float lowCut, float highCut, float gain) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return boostBand(aName, oName, sampleRate, lowCut, highCut, gain).get(oName);
    }


    /**
     * **BoostAll — Apply uniform amplification across all frequencies**
     *
     * Uniformly amplifies all frequency components in the input tensor {@code a}
     * without altering the relative spectral balance. This operation increases
     * the overall loudness or magnitude of the signal while preserving its
     * tonal characteristics.
     * <p>
     * The amplification is applied evenly across all FFT bins, and the
     * {@code sampleRate} parameter is included for API consistency.
     * </p>
     *
     * @param a          The name of the input tensor (e.g., FFT magnitude or waveform).
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostAll(Tensor, String, int)
     * @see #boostAllI(String, int)
     * @see #boostAllI(Tensor, int)
     */
    default CuBridge boostAll(String a, String out, int sampleRate) {
        if (CuBridgeJNI.boost(a, out, sampleRate, 0f, Float.MAX_VALUE, 1.3f))
            return CuBridge.getInstance();
        else
            System.err.println("Error | boostAll | " + a + " | " + out);
        return null;
    }

    /**
     * **BoostAll — Apply uniform amplification to a tensor input**
     *
     * Applies a uniform amplification to the given {@link Tensor} {@code a}
     * and stores the result in {@code out}. The amplification increases
     * the signal’s overall level equally across all frequencies, maintaining
     * spectral consistency.
     *
     * @param a          The input {@link Tensor}.
     * @param out        The name to store the boosted tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return {@link CuBridge} instance if successful; otherwise {@code null}.
     * @see #boostAll(String, String, int)
     */
    default CuBridge boostAll(Tensor a, String out, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        return boostAll(aName, out, sampleRate);
    }

    /**
     * **BoostAll (Immediate) — Immediate uniform amplification**
     *
     * Immediately applies uniform amplification across all frequency bins
     * to the named input tensor {@code a}, returning the resulting boosted
     * {@link Tensor}. This operation does not modify the CuBridge queue.
     *
     * @param a          The name of the input tensor.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} with uniformly increased magnitude.
     * @see #boostAll(String, String, int)
     */
    default Tensor boostAllI(String a, int sampleRate) {
        String oName = genRandomNameAudio();
        return boostAll(a, oName, sampleRate).get(oName);
    }

    /**
     * **BoostAll (Immediate) — Apply uniform amplification to a tensor input**
     *
     * Immediately applies a uniform boost across all frequency bins to the given
     * {@link Tensor} {@code a} and returns the resulting boosted tensor.
     * <p>
     * This method performs all operations in-memory and is intended for
     * one-shot amplitude enhancement.
     * </p>
     *
     * @param a          The input {@link Tensor}.
     * @param sampleRate The sampling rate of the input signal, in Hz.
     * @return A boosted {@link Tensor} with amplified spectral magnitude.
     * @see #boostAll(Tensor, String, int)
     */
    default Tensor boostAllI(Tensor a, int sampleRate) {
        String aName = genRandomNameAudio();
        CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return boostAll(aName, oName, sampleRate).get(oName);
    }


    /**
     * **Spectrogram — Convert magnitude or power spectrum to dB scale**
     *
     * Converts the input tensor {@code a} to a logarithmic (dB) scale representation.
     * <p>
     * Internally computes {@code 10 * log10(max(a, 1e-10))}, ensuring numerical stability
     * by clamping very small values to {@code 1e-10}.
     * </p>
     * <p>
     * Commonly used after STFT or filter bank operations to obtain a perceptually-scaled
     * spectrogram suitable for visualization or further analysis.
     * </p>
     *
     * @param a   The name of the input tensor (e.g., magnitude or power spectrum).
     * @param out The name to store the resulting dB-scaled spectrogram tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge spectrogram(String a, String out){
        if (CuBridgeJNI.spectrogram(a, out))
            return CuBridge.getInstance();
        else
            System.err.println("Error | spectrogram | " + a + " | " + out);
        return null;
    }

    /**
     * **Spectrogram — Apply dB-scale conversion to a tensor input**
     *
     * Converts the given input {@link Tensor} to a dB-scaled spectrogram by applying
     * {@code 10 * log10(max(a, 1e-10))}.
     *
     * @param a   The input {@link Tensor}.
     * @param out The name to store the resulting dB-scaled spectrogram tensor.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #spectrogram(String, String)
     */
    default CuBridge spectrogram(Tensor a, String out){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return spectrogram(aName, out);
    }

    /**
     * **Spectrogram (Immediate) — Immediate dB-scale conversion**
     *
     * Immediately converts a named tensor to its dB-scaled spectrogram equivalent,
     * using {@code 10 * log10(max(a, 1e-10))}, and returns the result directly.
     *
     * @param a The name of the input tensor.
     * @return A {@link Tensor} containing the dB-scaled spectrogram.
     * @see #spectrogram(String, String)
     */
    default Tensor spectrogramI(String a){
        String oName = genRandomNameAudio();
        return spectrogram(a, oName).get(oName);
    }

    /**
     * **Spectrogram (Immediate) — Apply dB-scale conversion to a tensor input**
     *
     * Immediately converts the given input {@link Tensor} to a dB-scaled spectrogram,
     * using {@code 10 * log10(max(a, 1e-10))}, and returns the result.
     *
     * @param a The input {@link Tensor}.
     * @return A {@link Tensor} containing the dB-scaled spectrogram.
     * @see #spectrogram(Tensor, String)
     */
    default Tensor spectrogramI(Tensor a){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return spectrogram(aName, oName).get(oName);
    }


    /**
     * **DCT — Apply Discrete Cosine Transform**
     *
     * Applies the Discrete Cosine Transform (type-II) to the input tensor {@code a}
     * and stores the first {@code nCoeffs} coefficients in {@code out}.
     * <p>
     * Commonly used after log-mel spectrogram computation to obtain MFCC features.
     * </p>
     *
     * @param a        The name of the input tensor (e.g., log-mel spectrogram).
     * @param out      The name to store the DCT-transformed tensor.
     * @param nCoeffs  The number of DCT coefficients to retain (e.g., 13 for MFCC).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge dct(String a, String out, int nCoeffs){
        if (CuBridgeJNI.dct(a, out, nCoeffs))
            return CuBridge.getInstance();
        else
            System.err.println("Error | dct | " + a + " | " + out + " | " + nCoeffs);
        return null;
    }

    /**
     * **DCT — Apply Discrete Cosine Transform to a tensor input**
     *
     * Applies the Discrete Cosine Transform (type-II) to the given input {@link Tensor}
     * and stores the first {@code nCoeffs} coefficients in {@code out}.
     *
     * @param a        The input {@link Tensor}.
     * @param out      The name to store the DCT-transformed tensor.
     * @param nCoeffs  The number of DCT coefficients to retain.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #dct(String, String, int)
     */
    default CuBridge dct(Tensor a, String out, int nCoeffs){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return dct(aName, out, nCoeffs);
    }

    /**
     * **DCT (Immediate) — Immediate Discrete Cosine Transform**
     *
     * Immediately applies a DCT (type-II) to a named tensor and returns the result as a {@link Tensor}.
     * Only the first {@code nCoeffs} coefficients are retained.
     *
     * @param a        The name of the input tensor.
     * @param nCoeffs  The number of DCT coefficients to retain.
     * @return A {@link Tensor} containing the DCT-transformed coefficients.
     * @see #dct(String, String, int)
     */
    default Tensor dctI(String a, int nCoeffs){
        String oName = genRandomNameAudio();
        return dct(a, oName, nCoeffs).get(oName);
    }

    /**
     * **DCT (Immediate) — Apply Discrete Cosine Transform to a tensor input**
     *
     * Immediately applies a DCT (type-II) to the given {@link Tensor} input and returns the result as a {@link Tensor}.
     * Only the first {@code nCoeffs} coefficients are retained.
     *
     * @param a        The input {@link Tensor}.
     * @param nCoeffs  The number of DCT coefficients to retain.
     * @return A {@link Tensor} containing the DCT-transformed coefficients.
     * @see #dct(Tensor, String, int)
     */
    default Tensor dctI(Tensor a, int nCoeffs){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return dct(aName, oName, nCoeffs).get(oName);
    }


    /**
     * **MFCC — Apply DCT on Mel-scaled spectrogram**
     *
     * Applies a Discrete Cosine Transform (DCT-II) to the Mel-scaled spectrogram tensor {@code a},
     * extracting the first {@code nCoeffs} Mel-Frequency Cepstral Coefficients (MFCCs)
     * and storing them in {@code out}.
     * <p>
     * This function is specialized for Mel spectrogram inputs, typically after logarithmic scaling.
     * It is equivalent to a DCT operation optimized for MFCC computation.
     * </p>
     *
     * @param a        The name of the input Mel-scaled spectrogram tensor.
     * @param out      The name to store the resulting MFCC tensor.
     * @param nCoeffs  The number of MFCC coefficients to compute (e.g., 13).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge mfcc(String a, String out, int nCoeffs){
        if (CuBridgeJNI.mfcc(a, out, nCoeffs))
            return CuBridge.getInstance();
        else
            System.err.println("Error | mfcc | " + a + " | " + out + " | " + nCoeffs);
        return null;
    }

    /**
     * **MFCC — Apply DCT on Mel-scaled spectrogram tensor input**
     *
     * Applies a DCT-II transform to the given Mel-scaled spectrogram {@link Tensor},
     * extracting the first {@code nCoeffs} coefficients and storing them in {@code out}.
     *
     * @param a        The input {@link Tensor} representing a Mel-scaled spectrogram.
     * @param out      The name to store the resulting MFCC tensor.
     * @param nCoeffs  The number of MFCC coefficients to compute.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     * @see #mfcc(String, String, int)
     */
    default CuBridge mfcc(Tensor a, String out, int nCoeffs){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        return mfcc(aName, out, nCoeffs);
    }

    /**
     * **MFCC (Immediate) — Immediate Mel cepstral transform**
     *
     * Immediately applies a DCT-II transform to the named Mel-scaled spectrogram tensor,
     * extracting {@code nCoeffs} MFCC coefficients and returning the result as a {@link Tensor}.
     *
     * @param a        The name of the Mel-scaled spectrogram tensor.
     * @param nCoeffs  The number of MFCC coefficients to compute.
     * @return A {@link Tensor} containing the computed MFCC coefficients.
     * @see #mfcc(String, String, int)
     */
    default Tensor mfccI(String a, int nCoeffs){
        String oName = genRandomNameAudio();
        return mfcc(a, oName, nCoeffs).get(oName);
    }

    /**
     * **MFCC (Immediate) — Apply DCT on a Mel-scaled spectrogram tensor**
     *
     * Immediately applies a DCT-II transform to the given Mel-scaled {@link Tensor} input
     * and returns the resulting MFCC tensor.
     *
     * @param a        The input {@link Tensor} representing a Mel-scaled spectrogram.
     * @param nCoeffs  The number of MFCC coefficients to compute.
     * @return A {@link Tensor} containing the computed MFCC coefficients.
     * @see #mfcc(Tensor, String, int)
     */
    default Tensor mfccI(Tensor a, int nCoeffs){
        String aName = genRandomNameAudio(); CuBridge.getInstance().put(a, aName);
        String oName = genRandomNameAudio();
        return mfcc(aName, oName, nCoeffs).get(oName);
    }


    /**
     * **Gaussian Window — Gaussian window generator**
     *
     * Generates a Gaussian window tensor of size {@code winSize} with the given standard deviation {@code sigma}.
     * <p>
     * The Gaussian window provides smooth tapering with controllable spread defined by {@code sigma},
     * reducing spectral leakage while preserving time resolution.
     * </p>
     *
     * @param out     The name to store the generated window tensor.
     * @param winSize The number of window samples.
     * @param sigma   The standard deviation controlling the shape (typically 0.3–0.5).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     *
     * @example
     * <pre>
     * // Example: Generate a 1024-sample Gaussian window with sigma=0.4
     * cb.makeGaussianWindow("gauss", 1024, 0.4f);
     * </pre>
     */
    default CuBridge makeGaussianWindow(String out, int winSize, float sigma){
        if (CuBridgeJNI.makeGaussianWindow(out, winSize, sigma))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeGaussianWindow | " + out + " | " + winSize + " | " + sigma);
        return null;
    }

    /**
     * **Gaussian Window (Immediate) — Immediate Gaussian window generator**
     *
     * Immediately generates a Gaussian window tensor with the specified size and standard deviation,
     * assigns a random internal name, and returns the resulting {@link Tensor} directly.
     *
     * @param winSize The number of window samples.
     * @param sigma   The standard deviation controlling the shape (typically 0.3–0.5).
     * @return A {@link Tensor} containing the generated Gaussian window.
     * @see #makeGaussianWindow(String, int, float)
     */
    default Tensor makeGaussianWindowI(int winSize, float sigma){
        String oName = genRandomNameAudio();
        return makeGaussianWindow(oName, winSize, sigma).get(oName);
    }


    /**
     * **Rectangular Window — rectangular window generator**
     *
     * Generates a rectangular (boxcar) window tensor of size {@code winSize}.
     * <p>
     * The rectangular window applies no tapering and keeps all samples at full amplitude.
     * While simple, it results in higher spectral leakage compared to tapered windows.
     * </p>
     *
     * @param out     The name to store the generated window tensor.
     * @param winSize The number of window samples.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     *
     * @example
     * <pre>
     * // Example: Generate a 512-sample rectangular window
     * cb.makeRectWindow("rect", 512);
     * </pre>
     */
    default CuBridge makeRectWindow(String out, int winSize){
        if (CuBridgeJNI.makeRectWindow(out, winSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeRectWindow | " + out + " | " + winSize);
        return null;
    }

    /**
     * **Rectangular Window (Immediate) — Immediate rectangular window generator**
     *
     * Immediately generates a rectangular window tensor with the specified size,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param winSize The number of window samples.
     * @return A {@link Tensor} containing the generated rectangular window.
     * @see #makeRectWindow(String, int)
     */
    default Tensor makeRectWindowI(int winSize){
        String oName = genRandomNameAudio();
        return makeRectWindow(oName, winSize).get(oName);
    }


    /**
     * **Hann Window — Hann window generator**
     *
     * Generates a Hann (raised cosine) window tensor of size {@code winSize}.
     * <p>
     * The Hann window smoothly tapers both ends of the signal to zero,
     * minimizing spectral leakage and commonly used in FFT/STFT processing.
     * </p>
     *
     * @param out     The name to store the generated window tensor.
     * @param winSize The number of window samples.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     *
     * @example
     * <pre>
     * // Example: Generate a 1024-sample Hann window
     * cb.makeHannWindow("hann", 1024);
     * </pre>
     */
    default CuBridge makeHannWindow(String out, int winSize){
        if (CuBridgeJNI.makeHannWindow(out, winSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeHannWindow | " + out + " | " + winSize);
        return null;
    }

    /**
     * **Hann Window (Immediate) — Immediate Hann window generator**
     *
     * Immediately generates a Hann window tensor with the specified size,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param winSize The number of window samples.
     * @return A {@link Tensor} containing the generated Hann window.
     * @see #makeHannWindow(String, int)
     */
    default Tensor makeHannWindowI(int winSize){
        String oName = genRandomNameAudio();
        return makeHannWindow(oName, winSize).get(oName);
    }


    /**
     * **Hamming Window — Hamming window generator**
     *
     * Generates a Hamming window tensor of size {@code winSize}.
     * <p>
     * The Hamming window reduces spectral leakage similarly to Hann,
     * but retains a small nonzero value at both ends, improving frequency resolution.
     * </p>
     *
     * @param out     The name to store the generated window tensor.
     * @param winSize The number of window samples.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     *
     * @example
     * <pre>
     * // Example: Generate a 1024-sample Hamming window
     * cb.makeHammingWindow("hamming", 1024);
     * </pre>
     */
    default CuBridge makeHammingWindow(String out, int winSize){
        if (CuBridgeJNI.makeHammingWindow(out, winSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeHammingWindow | " + out + " | " + winSize);
        return null;
    }

    /**
     * **Hamming Window (Immediate) — Immediate Hamming window generator**
     *
     * Immediately generates a Hamming window tensor with the specified size,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param winSize The number of window samples.
     * @return A {@link Tensor} containing the generated Hamming window.
     * @see #makeHammingWindow(String, int)
     */
    default Tensor makeHammingWindowI(int winSize){
        String oName = genRandomNameAudio();
        return makeHammingWindow(oName, winSize).get(oName);
    }


    /**
     * **Bartlett Window — Bartlett window generator**
     *
     * Generates a Bartlett (triangular) window tensor of size {@code winSize}.
     * <p>
     * The Bartlett window linearly increases to the center and decreases symmetrically,
     * offering a simple shape with moderate spectral leakage reduction.
     * </p>
     *
     * @param out     The name to store the generated window tensor.
     * @param winSize The number of window samples.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     *
     * @example
     * <pre>
     * // Example: Generate a 512-sample Bartlett window
     * cb.makeBartlettWindow("bartlett", 512);
     * </pre>
     */
    default CuBridge makeBartlettWindow(String out, int winSize){
        if (CuBridgeJNI.makeBartlettWindow(out, winSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeBartlettWindow | " + out + " | " + winSize);
        return null;
    }

    /**
     * **Bartlett Window (Immediate) — Immediate Bartlett window generator**
     *
     * Immediately generates a Bartlett (triangular) window tensor with the specified size,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param winSize The number of window samples.
     * @return A {@link Tensor} containing the generated Bartlett window.
     * @see #makeBartlettWindow(String, int)
     */
    default Tensor makeBartlettWindowI(int winSize){
        String oName = genRandomNameAudio();
        return makeBartlettWindow(oName, winSize).get(oName);
    }


    /**
     * **Kaiser Window — Kaiser window generator**
     *
     * Generates a Kaiser window tensor of size {@code winSize} with the specified {@code beta} parameter.
     * <p>
     * The Kaiser window provides adjustable trade-off between main-lobe width and side-lobe attenuation
     * through the {@code beta} parameter, making it highly flexible for spectral control.
     * </p>
     *
     * @param out     The name to store the generated window tensor.
     * @param winSize The number of window samples.
     * @param beta    The shape parameter controlling the tapering (commonly 5–9).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     *
     * @example
     * <pre>
     * // Example: Generate a 1024-sample Kaiser window with β=8
     * cb.makeKaiserWindow("kaiser", 1024, 8.0f);
     * </pre>
     */
    default CuBridge makeKaiserWindow(String out, int winSize, float beta){
        if (CuBridgeJNI.makeKaiserWindow(out, winSize, beta))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeKaiserWindow | " + out + " | " + winSize + " | " + beta);
        return null;
    }

    /**
     * **Kaiser Window (Immediate) — Immediate Kaiser window generator**
     *
     * Immediately generates a Kaiser window tensor with the specified size and beta parameter,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param winSize The number of window samples.
     * @param beta    The shape parameter controlling the tapering (commonly 5–9).
     * @return A {@link Tensor} containing the generated Kaiser window.
     * @see #makeKaiserWindow(String, int, float)
     */
    default Tensor makeKaiserWindowI(int winSize, float beta){
        String oName = genRandomNameAudio();
        return makeKaiserWindow(oName, winSize, beta).get(oName);
    }


    /**
     * **Mel Filter Bank — Mel-scale filter bank generator**
     *
     * Generates a Mel filter bank matrix used for transforming FFT magnitudes into Mel-scaled frequency bins.
     * <p>
     * This function constructs a filter bank with {@code nMels} filters according to the given
     * {@code sampleRate} and {@code fftSize}, and stores the resulting matrix tensor under the name {@code out}.
     * </p>
     * <p>
     * The generated filter bank can be directly used for converting spectrograms or FFT results
     * into Mel-frequency representations in subsequent audio processing pipelines.
     * </p>
     *
     * @param out        The name to store the generated Mel filter tensor.
     * @param nMels      The number of Mel frequency bands (e.g., 40, 64, 128).
     * @param sampleRate The audio sampling rate in Hz (e.g., 16000, 22050, 44100).
     * @param fftSize    The FFT window size used when computing the spectrogram (e.g., 512, 1024).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge makeMelFilter(String out, int nMels, int sampleRate, int fftSize){
        if (CuBridgeJNI.makeMelFilter(out, nMels, sampleRate, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeMelFilter | " + out + " | " + nMels + " | " + sampleRate + " | " + fftSize);
        return null;
    }

    /**
     * **Mel Filter Bank (Immediate) — Immediate Mel filter generator**
     *
     * Immediately generates a Mel filter bank matrix with the specified parameters,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param nMels      The number of Mel frequency bands.
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size.
     * @return A {@link Tensor} containing the generated Mel filter bank.
     * @see #makeMelFilter(String, int, int, int)
     */
    default Tensor makeMelFilterI(int nMels, int sampleRate, int fftSize){
        String oName = genRandomNameAudio();
        return makeMelFilter(oName, nMels, sampleRate, fftSize).get(oName);
    }


    /**
     * **Bark Filter Bank — Bark-scale filter bank generator**
     *
     * Generates a Bark filter bank matrix used for perceptual frequency scaling based on the Bark scale.
     *
     * @param out        The name to store the generated Bark filter tensor.
     * @param nBands     The number of Bark frequency bands (typically around 24).
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size used when computing the spectrogram.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge makeBarkFilter(String out, int nBands, int sampleRate, int fftSize){
        if (CuBridgeJNI.makeBarkFilter(out, nBands, sampleRate, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeBarkFilter | " + out + " | " + nBands + " | " + sampleRate + " | " + fftSize);
        return null;
    }

    /**
     * **Bark Filter Bank (Immediate) — Immediate Bark filter generator**
     *
     * Immediately generates a Bark filter bank with the specified parameters,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param nBands     The number of Bark frequency bands.
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size.
     * @return A {@link Tensor} containing the generated Bark filter bank.
     * @see #makeBarkFilter(String, int, int, int)
     */
    default Tensor makeBarkFilterI(int nBands, int sampleRate, int fftSize){
        String oName = genRandomNameAudio();
        return makeBarkFilter(oName, nBands, sampleRate, fftSize).get(oName);
    }


    /**
     * **ERB Filter Bank — Equivalent Rectangular Bandwidth filter bank generator**
     *
     * Generates an ERB (Equivalent Rectangular Bandwidth) filter bank matrix for auditory modeling.
     *
     * @param out        The name to store the generated ERB filter tensor.
     * @param nBands     The number of ERB bands.
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size used when computing the spectrogram.
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge makeErbFilter(String out, int nBands, int sampleRate, int fftSize){
        if (CuBridgeJNI.makeErbFilter(out, nBands, sampleRate, fftSize))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeErbFilter | " + out + " | " + nBands + " | " + sampleRate + " | " + fftSize);
        return null;
    }

    /**
     * **ERB Filter Bank (Immediate) — Immediate ERB filter generator**
     *
     * Immediately generates an ERB filter bank with the specified parameters,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param nBands     The number of ERB bands.
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size.
     * @return A {@link Tensor} containing the generated ERB filter bank.
     * @see #makeErbFilter(String, int, int, int)
     */
    default Tensor makeErbFilterI(int nBands, int sampleRate, int fftSize){
        String oName = genRandomNameAudio();
        return makeErbFilter(oName, nBands, sampleRate, fftSize).get(oName);
    }


    /**
     * **Chroma Filter Bank — Chroma feature mapping filter generator**
     *
     * Generates a Chroma filter bank matrix used to map spectral bins into chroma (pitch class) features.
     *
     * @param out        The name to store the generated Chroma filter tensor.
     * @param nChroma    The number of chroma bins (typically 12 for one octave).
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size used for spectral analysis.
     * @param fRef       The reference frequency in Hz (e.g., 440.0 for A4).
     * @return {@link CuBridge} instance if successful, otherwise {@code null}.
     */
    default CuBridge makeChromaFilter(String out, int nChroma, int sampleRate, int fftSize, float fRef){
        if (CuBridgeJNI.makeChromaFilter(out, nChroma, sampleRate, fftSize, fRef))
            return CuBridge.getInstance();
        else
            System.err.println("Error | makeChromaFilter | " + out + " | " + nChroma + " | " + sampleRate + " | " + fftSize + " | " + fRef);
        return null;
    }

    /**
     * **Chroma Filter Bank (Immediate) — Immediate Chroma filter generator**
     *
     * Immediately generates a Chroma filter bank with the specified parameters,
     * assigns a random internal name, and returns the resulting {@link Tensor}.
     *
     * @param nChroma    The number of chroma bins.
     * @param sampleRate The audio sampling rate in Hz.
     * @param fftSize    The FFT window size used for spectral analysis.
     * @param fRef       The reference frequency in Hz.
     * @return A {@link Tensor} containing the generated Chroma filter bank.
     * @see #makeChromaFilter(String, int, int, int, float)
     */
    default Tensor makeChromaFilterI(int nChroma, int sampleRate, int fftSize, float fRef){
        String oName = genRandomNameAudio();
        return makeChromaFilter(oName, nChroma, sampleRate, fftSize, fRef).get(oName);
    }

}
