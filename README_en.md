# CuBridge

**CuBridge** is a lightweight tensor computation engine that connects Java with CUDA to deliver high-performance numerical operations.  
All operations are executed based on Tensor objects, and the system automatically detects whether to run on CPU or GPU.

> A lightweight Java tensor engine with automatic GPU acceleration via CUDA.  
> 
> This is the first project in the Alphabet Bridge series.

---

- This library enables Java programs to utilize CUDA acceleration.
- Unlike other libraries, it depends only on the JDK.
- Runs on CPU even in environments without CUDA or a GPU.
- Supports basic matrix operations, automatic broadcasting, and partial neural network functions (more to be added).
- Broadcasting is supported even when the dimensions are multiples of each other, not just size 1.

- [View CuBridge Changelog](Legacy/History_en.md)

---

##  Key Features

- Java + CUDA integration structure
- High-speed numerical engine based on the Tensor class
- Environment-adaptive execution: dynamically chooses between GPU and CPU based on availability
- JNI binding via DLL
- Broadcasting works with non-1-sized dimensions if they are multiples

---

##  Installation (Eclipse-based)

### Add `CuBridge.jar` to your project

1. In Eclipse, go to `Project → Properties → Java Build Path`
2. Switch to the `Libraries` tab → click **[Add External JARs...]**
3. Select the `CuBridge.jar` file placed in your project folder → Apply

---

### Add `CuBridge.jar` in IntelliJ IDEA

1. In IntelliJ, open **`File → Project Structure (Ctrl+Alt+Shift+S)`**  
2. In the left menu, select **`Modules`**, then click the **`Dependencies`** tab on the right  
3. Click **`+ (Add)` → JARs or Directories...**  
4. Select the downloaded `CuBridge.jar` file and click **OK**  
5. Apply settings:  
   - Set **Scope** to **Compile**  
   - Click **Apply → OK**  
6. Rebuild the project (**Ctrl+F9**) to apply the changes  

---

---

##  Example Code 1

```java
CuBridge cb = CuBridge.getInstance();

Tensor t1 = new Tensor(new double[] {1, 2, 3}, new int[] {3, 1});
Tensor t2 = new Tensor(new double[] {1, 2, 3}, new int[] {1, 3});
Tensor t3 = new Tensor(new double[] {1, 2, 3}, new int[] {3});

cb.put(t1, "x").put(t2, "w").put(t3, "b");
cb.affine("x", "w", "b").get().printData();
// You can omit tensor names for simpler operations
```

## Output:
```
Tensor(shape=[3, 3]):
    [  2.000,  4.000,  6.000 ]
    [  3.000,  6.000,  9.000 ]
    [  4.000,  8.000, 12.000 ]
```


## Example Code 2

```java
CuBridge cb = CuBridge.getInstance();

Tensor t1 = Tensor.randn(3, 1);
Tensor t2 = Tensor.randn(1, 3);

cb.dotI(t1, t2).printData();
// You can input tensors directly into the operator without using put()
// Adding 'I' to the operator name immediately returns a Tensor
```


---

##  License

This project is licensed under the [MIT License](LICENSE).
