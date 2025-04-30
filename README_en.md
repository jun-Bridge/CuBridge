# CuBridge

**CuBridge** is a lightweight Java-based tensor engine that bridges Java and CUDA to enable high-performance numerical computation.  
All operations are executed on `Tensor` objects, with automatic separation between RAM and VRAM for GPU acceleration.

> A lightweight Java tensor engine with automatic GPU acceleration via CUDA.

**!!! UNSTABLE !!! BE CAREFUL TO USE**
---

##  Key Features

- Java + CUDA bridge architecture
- High-speed tensor-based numerical engine
- Environment-adaptive behavior: switches between CPU and GPU based on availability
- Memory mode switching based on VRAM size (e.g., 4GB threshold for RAM/VRAM choice)
- Full Javadoc integration and IDE support
- JNI-based DLL interaction

---

##  Installation (Eclipse Guide)

### 1. Add `cubridge.jar` to your project

1. In Eclipse, open `Project → Properties → Java Build Path`
2. Go to the `Libraries` tab → click **[Add External JARs...]** under `Classpath`
3. Select `cubridge.jar` (after placing it in your project folder) → Apply

---

### 2. Link native DLL (cubridge.dll)

1. In the `Libraries` tab, click ▼ next to `cubridge.jar`
2. Choose **Native library location → Edit**
3. Select `External Folder...`
4. Choose the folder where `cubridge.dll` is located
5. Click OK → Apply

> Note: The DLL must be outside the JAR, not embedded inside it.

---

### 3. Attach Javadoc

1. Again, click ▼ next to `cubridge.jar`
2. Choose **Javadoc location → Edit**
3. Options:
   - `Javadoc in archive` → select `cubridge-javadoc.jar`
   - or `Javadoc in folder` → choose the folder where `docs/` is stored
4. Click OK → Apply

Once connected, hovering over any function in your IDE will display its documentation.

---

##  Example Usage

```java
CuBridge cb = CuBridge.getInstance();

Tensor t1 = new Tensor(new double[] {1, 2, 3}, new int[] {3, 1});
Tensor t2 = new Tensor(new double[] {1, 2, 3}, new int[] {1, 3});
Tensor t3 = new Tensor(new double[] {1, 2, 3}, new int[] {3});

cb.put(t1, "x").put(t2, "w").put(t3, "b");
cb.affine("x", "w", "b").get().printData();
// For simpler operations, you can omit the name and leave it empty
```

Output:
```
Tensor(shape=[3, 3]):
    [  2.000,  4.000,  6.000 ]
    [  3.000,  6.000,  9.000 ]
    [  4.000,  8.000, 12.000 ]
```

---

##  Example Directory Structure

```
CuBridge/
├─ src/
│  ├─ CuBridgeJNI.java
│  ├─ CuBridge.java
│  ├─ Tensor.java
├─ cubridge.jar
├─ lib/
│  ├─ cubridge.dll
├─ doc/
│  ├─ cubridge-javadoc.jar
├─ README.md
├─ README_en.md
├─ LICENSE
```

---

##  License

This project is distributed under the terms of the [MIT License](LICENSE).
