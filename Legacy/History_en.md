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
