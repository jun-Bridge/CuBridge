# **CuBridge Change Log**

---

## **BETA**

- Verified integration between **Java ↔ C++/CUDA C**
- Established a **triple-memory architecture**: queue / map / buffer
- Built the **put / cal / get** execution system
- Implemented a **dual memory layer**: cpu_ram / gpu_vram
- Added **automatic environment recognition**

### Function Additions
- `auto`, `cal`, `ram`, `env`, `sysinfo`, `clear`
- `visualQueue`, `visualMap`, `put`, `get`

### Operation Additions
- **Unary operations:**  
  `abs, neg, square, sqrt, log, log_2, ln, reciprocal, sin, cos, tan, step, sigmoid, tanh, relu, leakRelu, softplus, exp, round, ceil, floor, not`
- **Binary operations:**  
  `add, sub, mul, div, pow, mod, gt, lt, ge, le, eq, ne, and, or`
- **Axis operations:**  
  `sum, mean, var, std, max, min`
- **Matrix operations:**  
  `transpose, dot`
- **Neural network operations:**  
  `affine, cee, mse, softmax`

---

## **Version 1.0**

### Beta stabilization
- Fixed memory leaks  
- Optimized internal structure  
- Removed map → established dual structure: queue/buffer  
- Removed VRAM policy for better speed and stability  

### Dependency resolution
- Isolated and stabilized CUDA system  

### Deployment optimization
- Single JAR execution supported  

### Function Additions
- `duple`, `broad`, `visualBuffer`, `visualAll`

### Operation Improvements
- Axis operations divided into **axis-integrated** and **axis-independent**
  - Integrated: `sum, mean, var, std, max, min`
  - Independent: `accumulate, compress, expand, argmax, argmin, axisMax, axisMin`
  - Integrated ops reduce along all axes up to the specified one  
  - Independent ops calculate only along the given axis
- **Transpose optimization** for speed and multi-axis support
- **Dot/Matmul separation** – automatic bypass depending on tensor shape
- Added axis argument to **softmax**

---

## **Version 1.1**

### Bug Fixes
- Fixed incorrect axis reversal when using `-1` for automatic axis selection in transpose.

### New Operations
- `rad2deg`, `deg2rad`: Degree–radian conversion
- `im2col1D`, `col2im1D`: 1D convolution input/output reconstruction
- `im2col2D`, `col2im2D`: 2D convolution input/output reconstruction
- `reshape`: Dynamically update tensor shape and length

### Tensor Class Extensions
- Added string-based tensor constructors  
  `Tensor(String[][])` and `Tensor(String[][], float)`

---

## **Version 1.1.1**

### Bug Fixes

1. **Fixed queue name mismatch in `pop()`**
   - Previously, `genRandomName()` was used instead of `""`, causing the function to search internal tensor names instead of the top queue element.  
   - This caused `pop()` to always fail — now fixed.

2. **Fixed broadcast direction error**
   - In binary operations, broadcast direction was reversed.  
   - Example: for `{3,2}` and `{1,6}`, expansion incorrectly applied along columns instead of rows.

---

## **Version 1.2**

### Bug Fixes & Enhancements

#### 1. **Constant Tensor System Added**

**Key Features**
- Names starting with `'_'` and `usageCount < 0` are automatically recognized as constants.
- User-defined constants supported (e.g., `_VAR1`, `_CUSTOM_CONST`)
- All constants are automatically broadcastable (`broadcast = true` by default)
- Constants are **immutable**:
  - Calls to `setUsage`, `setBroad`, or `setReshape` are ignored with warnings
  - `smartPush()` or `put()` using the same name returns an error
- Constants **cannot** be used as output names:
  - Example: `cb.exp("a", "_PI")` → error (`_PI` is immutable)

#### Built-in Constant List

| Name         | Value            | Description               |
|---------------|------------------|----------------------------|
| `_ZERO`       | 0.0              | Default zero constant      |
| `_ONE`        | 1.0              | Identity value             |
| `_TWO`        | 2.0              | Exponentiation, power ops  |
| `_THREE`      | 3.0              |                            |
| `_FOUR`       | 4.0              |                            |
| `_FIVE`       | 5.0              |                            |
| `_SIX`        | 6.0              |                            |
| `_SEVEN`      | 7.0              |                            |
| `_EIGHT`      | 8.0              |                            |
| `_NINE`       | 9.0              |                            |
| `_HALF`       | 0.5              | Mean/Normalization         |
| `_PI`         | 3.14159265359    | For trigonometric ops      |
| `_E`          | 2.718281         | Euler’s number             |
| `_EPSILON`    | 1e-6             | Numerical tolerance        |
| `_RATE`       | 0.001            | Learning rate              |
| `_NEG`        | -1.0             | Negative constant          |
| `_HUNDRED`    | 100.0            | Percent, scaling           |
| `_MAXPIXEL`   | 255.0            | Image normalization        |

#### 2. **Visual Series Improved**
- `visualQueue()` → shows only normal tensors  
- `visualQueueAll()` → includes constants  
- New display format: `"Queue Size : 20 (Const : 18, Var : 2)"`  
- Buffer display unchanged  

#### 3. **Unified Error Message Format**
- All operation functions now display consistent, detailed error messages  
- Includes function name, input/output tensors, and failure reason  

Example:  
`[ERROR][EXP][Cannot Execute][Tensor val1, _PI]`

---

## **Version 1.3**

### Tensor Immediate I/O System Added

#### Immediate Input Functions
- You can now pass tensors directly to operators without calling `put()` first.  
  - Example:  
    `cb.add(Tensor a, Tensor b)`  
    `cb.add(Tensor a, String b)`  
  - In the latter, a tensor and an existing constant or queued tensor can be combined.

#### Immediate Output Functions
- Operators can now return tensors directly without using `get()`.  
- These are suffixed with **`I`**, except for `transpose(T)`, `im2col`, and `col2im`.  
  - Example:  
    `Tensor c = cb.addI(String a, String b)`

#### Immediate Input & Output Combined
- You can perform full GPU operations without `put()` or `get()`.  
  - Example:  
    `Tensor c = cb.addI(Tensor a, Tensor b)`

---

## **Version 1.3.1**

### Bug Fixes & Optimization

1. **Broadcast Bug in Matmul**
   - Fixed incorrect broadcasting in `matmul` where the second matrix depended on the first’s axis length.

2. **`im2col` Bug & Optimization**
   - Fixed improper matrix rearrangement.
   - Reorganized kernel size mapping for performance optimization.

3. **Dot Optimization**
   - Adopted cuBLAS backend for accelerated dot product kernel.

---

## **Version 1.4**

### **Dot and General Operation Optimization**

1. **Added cuBLAS Strided Dot (Batch Dot Product)**
   - Enabled **stride-based batched dot product** to process multiple inputs simultaneously.  
   - Reduced kernel launch overhead and improved memory access pattern, increasing throughput.  
   - Remains separated from `matmul` for optimized 1D/2D vector operations.

2. **Simplified Transpose Flag — for dot/matmul only**
   - In `reshape()`, specifying a **negative first axis** automatically marks the tensor as transposed.  
     Example: `{ -M, N }` = equivalent to `{ N, M }` for internal execution.  
   - This rule applies **only to dot/matmul**.

3. **VRAM Caching Policy Introduced**
   - In GPU mode, results remain **cached in VRAM** for reuse in subsequent operations.  
   - Minimizes host↔device copy overhead and stabilizes performance.  
   - Internal `flush/check` logic refined for reliability.

4. **Optimized Internal Execution Path**
   - Streamlined `gatedMemory → execute → push` process.  
   - Reduced redundant allocation and copy overhead.  
   - Pre-validation for broadcast, axis ops, and transpose flags.  
   - Removed unnecessary variable recovery step → **higher speed and memory efficiency**.

> **Compatibility:**  
> No API changes. Transpose flag is optional (applies only if negative axis used).

---

## **Version 1.5 || 20251112**

### **Massive Function Expansion & Overload Unification**

#### 1. Tensor Class Enhancements
- Added `slice`, `vstack`, `hstack`, `stack`
- Added `Tensor(wav/csv)` `Tensor(float[][])` constructors
- Unified WAV format naming → `"wav_16000_32"`
- Improved function documentation (Javadoc)

#### 2. Function Additions

- **Unary:**  
  `abs, neg, square, sqrt, rsqrt, log, log2, ln, exp, reciprocal, 
    sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, 
    step, sigmoid, relu, leakRelu, softplus, 
    round, ceil, floor, not, deg2rad, rad2deg`

- **Binary:**  
  `add, sub, mul, div, pow, mod, gt, lt, ge, le, eq, ne, and, or`

- **Axis Cascade:**  
  `sum, mean, var, std, max, min`

- **Axis:**  
  `accumulate, compress, expand, axisMax, axisMin, axisVar, axisStd, argMax, argMin`

#### 3. Operation Groups Added

- **Algebra:**  
  `l2normalize, dot, matmul, transpose, 
    trace, inverse, eigen, svd, det, qr, cholesky, rank, 
    normalize, standardize, affine, softmax`

- **Audio:**  
  `low/mid/high/All/preEmphasis, 
    applyWindow, applyFilter, 
    fft, rfft, ifft, powfft, magfft, phasefft, 
    low/mid/high/All/boost, spectrogram, dct, mfcc, 
    makeMelFilter, makeBarkFilter, makeErbFilter, makeChromaFilter, 
    makeGaussianWindow, makeRectWindow, makeHannWindow, makeHammingWindow, makeBartlettWindow, makeKaiserWindow`

- **Image:**  
  `rotate, shift, translate, resize, crop, mask, pad, 
    boxBlur, gaussianBlur, medianBlur, flipH, flipV, 
    grayScale, chSplit, chMerge, 
    im2col1D, col2im1D, im2col2D, col2im2D`

- **Scalar:**  
  `L1Norm, L2Norm, LinfNorm, 
    L1Dist, L2Dist, LinfDist, cosDist, cosSim, 
    mse, bce, cee, mae, rmse, mape, 
    focal, perplexity, dice, iou`

- **Utility:**  
  `clip, softClip, sigClip, tanhClip, logClip`

#### 4. Bug Fixes & Optimization
- Internal stability improvements  
- Performance optimization for multi-type tensor operations  

---

