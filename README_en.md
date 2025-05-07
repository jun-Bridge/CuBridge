# CuBridge

**CuBridge** is a lightweight tensor computation engine that connects Java with CUDA to enable high-performance numerical operations.  
All operations are performed based on the `Tensor` object, with automatic separation of RAM/VRAM for GPU acceleration.

> A lightweight Java tensor engine with automatic GPU acceleration via CUDA.  
> 
> **Note: This project is not yet in a stable release. Use with caution in production.**  
>
> This is the first project in the Alphabet Bridge series.

---

##  Key Features

- Java + CUDA integration structure  
- High-speed numerical computation engine based on the Tensor class  
- Environment-adaptive library: automatically switches computation paths based on GPU/CUDA availability  
- Automatically decides whether to use VRAM or RAM for main storage depending on VRAM size (default: 4GB threshold)  
- Javadoc integration and IDE support  
- JNI binding via DLL files  

---

##  Installation (Using Eclipse)

### Add `CuBridge.jar` to your project

1. In Eclipse, open `Project → Properties → Java Build Path`
2. Go to the `Libraries` tab → click **[Add External JARs...]**
3. Choose the `CuBridge.jar` file (move it to your project folder if necessary), then click Apply  

---

##  Example Code

```java
CuBridge cb = CuBridge.getInstance();

Tensor t1 = new Tensor(new double[] {1, 2, 3}, new int[] {3, 1});
Tensor t2 = new Tensor(new double[] {1, 2, 3}, new int[] {1, 3});
Tensor t3 = new Tensor(new double[] {1, 2, 3}, new int[] {3});

cb.put(t1, "x").put(t2, "w").put(t3, "b");
cb.affine("x", "w", "b").get().printData();
// You may omit names for simpler operations
```

Output:
```
Tensor(shape=[3, 3]):
    [  2.000,  4.000,  6.000 ]
    [  3.000,  6.000,  9.000 ]
    [  4.000,  8.000, 12.000 ]
```

---

##  Directory Structure

```
CuBridge/
├─ src/
│  ├─ CuBridgeJNI.java
│  ├─ CuBridge.java
│  ├─ Tensor.java
├─ dll/
│  ├─ CuBridgeDriver.dll
│  ├─ CuBridgeCudaC.dll
├─ doc/
├─ CuBridgeJNI.class
├─ CuBridge.class
├─ Tensor.class
```

---

##  License

This project is licensed under the [MIT License](LICENSE).
