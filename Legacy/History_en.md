# CuBridge Changelog

## BETA

- Verified integration between Java and C++/CUDA C  
- Established a triple memory structure: queue / map / buffer  
- Implemented put / cal / get operation flow  
- Designed dual memory system: cpu_ram / gpu_vram  
- Automatic environment detection  
- Functions added:
  - auto, cal, ram, env, sysinfo, clear
  - visual queue/map, put, get
- Operations added:
  - **Unary operations**:  
    abs, neg, square, sqrt, log, log_2, ln, reciprocal, sin, cos, tan, step, sigmoid, tanh, relu, leakRelu, softplus, exp, round, ceil, floor, not
  - **Binary operations**:  
    add, sub, mul, div, pow, mod, gt, lt, ge, le, eq, ne, and, or
  - **Axis-based operations**:  
    sum, mean, var, std, max, min
  - **Matrix operations**:  
    transpose, dot
  - **Neural network operations**:  
    affine, cee, mse, softmax

---

## Version 1.0

- BETA stabilization:
  - Memory leak fixes
  - Structural optimization
  - Removed `map` and finalized queue/buffer dual structure
  - Removed VRAM policy for improved speed and stability
- Dependency resolution:
  - Separated and stabilized CUDA system
- Deployment optimization:
  - Now runs with a single JAR file
- Functions added:
  - duple, broad, visualBuffer / All
- Operation enhancements:
  - Split axis operations into **aggregated** and **independent**
    - Aggregated: sum, mean, var, std, max, min
    - Independent: accumulate, compress, expand, argmax, argmin, axisMax, axisMin
    - Aggregated ops combine all axes up to the specified one
    - Independent ops apply only to the specified axis
  - Transpose optimization:
    - Improved performance and multi-axis support
  - Dot product dualization:
    - dot/matmul separated, internally bypassed depending on shape
  - Axis argument added to softmax

---

## Version 1.1

- Bug Fixes
  - Fixed an issue in `transpose` where automatic axis inference using -1 caused reversed axis order.

- New Operations
  - `rad2deg`, `deg2rad`: Support for angle-to-radian and radian-to-angle conversion.
  - `im2col1D`, `col2im1D`: Input/output restructuring functions for 1D convolution.
  - `im2col2D`, `col2im2D`: Input/output restructuring functions for 2D convolution.
  - `reshape`: Dynamically updates the shape and size (`sLen`) of a tensor within the queue.

- Tensor Class Extension
  - Added string-based tensor constructors: `Tensor(String[][])` and `Tensor(String[][], float)` for initialization from string arrays.

---

## Version 1.1.1

- Bug Fixes

1. **Fixed queue name mismatch in `pop()`**
   - Previously, `genRandomName()` was mistakenly called instead of `""`, causing the function to search for tensors with auto-generated names rather than the top of the queue.
   - This critical bug caused `pop()` to always fail. It is now resolved.

2. **Fixed incorrect broadcast direction**
   - In binary operations, the broadcasting axis was incorrectly chosen, resulting in reversed broadcasting behavior.
   - Example: When computing with shapes `{3,2}` and `{1,6}`, expansion was wrongly applied along the columns. It now correctly expands along the rows.


--


## Version 1.2

- Bug Fixes and Feature Improvements

1. Constant Tensor System Introduced

**Key Features**
- A tensor is recognized as a **constant** if its name starts with `'_'` and `usageCount < 0`.
- Constants are **user-definable** (e.g., `_VAR1`, `_CUSTOM_CONST`, etc.).
- All constants automatically set `broadcast = true`. User-defined constants can override it.
- Constants are strictly **immutable**:
  - Calling `setUsage`, `setBroad`, or `setReshape` will print a warning and ignore the change.
  - Overwriting constants via `smartPush()` or duplicate `put()` will return an error.
- Constants **cannot be used as output names** for any operation.
  - Example: `cb.exp("a", "_PI")` -> Error (cannot overwrite constant `_PI`)

#### Built-in Constants

| Name         | Value           | Notes                    |
|--------------|------------------|---------------------------|
| `_ZERO`      | 0.0              | Basic zero constant       |
| `_ONE`       | 1.0              | Identity operand          |
| `_TWO`       | 2.0              | Square, exponentiation    |
| `_THREE`     | 3.0              |                           |
| `_FOUR`      | 4.0              |                           |
| `_FIVE`      | 5.0              |                           |
| `_SIX`       | 6.0              |                           |
| `_SEVEN`     | 7.0              |                           |
| `_EIGHT`     | 8.0              |                           |
| `_NINE`      | 9.0              |                           |
| `_HALF`      | 0.5              | Averaging, normalization  |
| `_PI`        | 3.14159265359    | Trigonometric functions   |
| `_E`         | 2.718281         | Exponential functions     |
| `_EPSILON`   | 1e-6             | Numerical tolerance       |
| `_RATE`      | 0.001            | Learning rate, etc.       |
| `_NEG`       | -1.0             | Negative unit             |
| `_HUNDRED`   | 100.0            | Percentage calculations   |
| `_MAXPIXEL`  | 255.0            | Image normalization       |


2. Visual Series Enhancements

- `visualQueue()` displays only variable tensors.
- `visualQueueAll()` displays all tensors including constants.
- Display format enhanced:
  - Example: `Queue Size : 20 (Const : 18, Var : 2)`
- Buffer visual output remains unchanged.


3. Standardized Error Message Format

- All operation functions now follow a unified error format.
- Error messages now include input and output tensor names for clearer diagnostics.

**Example:**
```text
[ERROR][EXP][Cannot Execute][Tensor val1, _PI]

---

## Version 1.3

- Added direct tensor input/output functionality

- Added direct tensor input functions  
  - You no longer need to explicitly use the put() function; you can now pass Tensor objects directly as parameters to operators.  
    - ex) cb.add(Tensor a, Tensor b), cb.add(Tensor a, String b)...  
    - In the second case, it is possible to perform operations by specifying a constant or a previously stored tensor using a string.

- Added direct tensor output functions  
  - You no longer need to explicitly use the get() function; operators now directly return Tensor objects.  
  - All direct output functions are named with an 'I' appended to the operator name, except for transpose(T), im2col, and col2im (unchanged).  
    - ex) Tensor c = cb.addI(String a, String b)...

- Added direct tensor input/output functions  
  - Both of the above features can now be used simultaneously.  
  - GPU-accelerated operations can now be performed using only operators, without put() and get().  
    - ex) Tensor c = cb.addI(Tensor a, Tensor b)
    
---

## Version 1.3.1

- Bug Fixes & Improvements

1. **Broadcast Bug Fix**
   - Fixed a bug where, during matmul execution, the broadcasting of the second matrix was dependent on the axis size of the first matrix.
   - This issue caused matmul to produce completely incorrect outputs, which has now been resolved.

2. **im2col Bug Fix & Enhancement**
   - Fixed an issue where matrix rearrangement via im2col was not performed correctly.
   - Additionally, optimized performance by rearranging the kernel size for faster execution.

3. **dot Optimization**
   - Improved the inner product kernel performance by introducing cuBLAS.