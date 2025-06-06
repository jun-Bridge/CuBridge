# CuBridge Changelog (English)

## BETA

- Verified integration between Java and C++/CUDA C
- Established a triple memory structure: queue / map / buffer
- Implemented core command flow: put / cal / get
- Built dual memory model: cpu_ram / gpu_vram
- Automatic environment detection
- Added utility functions: auto, cal, ram, env, sysinfo, clear, visual queue/map, put, get
- Added operations:
    - Unary: abs, neg, square, sqrt, log, log_2, ln, reciprocal, sin, cos, tan, step, sigmoid, tanh, relu, leakRelu, softplus, exp, round, ceil, floor, not
    - Binary: add, sub, mul, div, pow, mod, gt, lt, ge, le, eq, ne, and, or
    - Axis-based: sum, mean, var, std, max, min
    - Matrix: transpose, dot
    - Neural Network: affine, cee, mse, softmax

## Version 1.0

- BETA stabilization:
    - *Memory leak cleanup*
    - Structural optimization
    - Removed `map`, finalized dual structure: queue / buffer
    - Removed VRAM policy for speed and simplicity
    - Improved performance and stability
- Dependency resolution:
    - CUDA system isolated and stabilized
- Deployment optimization:
    - Now you only need a single JAR file!
- Added functions: duple, broad, visualBuffer / visualAll
- Enhanced operations:
    - Split axis operations into **aggregated** and **independent**
        - Aggregated operations combine all axes up to the specified one
        - Independent operations only apply to the specified axis
        - Aggregated: sum, mean, var, std, max, min
        - Independent: accumulate, compress, expand, argmax, argmin, axisMax, axisMin
    - Transpose optimization: speed-up and multi-axis support
    - Dot product dualization: dot/matmul separated, auto-switched by shape
    - Added axis argument to softmax
