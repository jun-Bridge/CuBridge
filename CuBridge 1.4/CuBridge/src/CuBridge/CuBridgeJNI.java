package CuBridge;

import java.io.*;

class CuBridgeJNI {
	
	private static String loadDll(String fileName) throws IOException {
	    File dir = new File(System.getProperty("java.io.tmpdir"), "cubridge_dll_tmp");
	    if (!dir.exists()) dir.mkdirs();

	    File dllFile = File.createTempFile(fileName.replace(".dll", ""), ".dll", dir);

	    try (InputStream in = CuBridgeJNI.class.getResourceAsStream("dll/" + fileName);
	         OutputStream out = new FileOutputStream(dllFile)) {

	        if (in == null)
	            throw new FileNotFoundException("JAR 내부에서 '" + fileName + "' 파일을 찾을 수 없습니다.");

	        byte[] buffer = new byte[1024];
	        int len;
	        
	        while ((len = in.read(buffer)) != -1)
	            out.write(buffer, 0, len);
	        
	    }

	    dllFile.deleteOnExit();

	    return dllFile.getAbsolutePath();
	}

	static {
		try {
			System.load(loadDll("CuBridgeDriver.dll"));	
			init(loadDll("CuBridgeCudaC.dll"));
		} catch (Exception e) {
			System.out.println("CuBridge DLL 로딩 실패" + e);
		}
		
	}
	
	
	
	static native void init(String path);
	static native void refresh();
	static native void setAuto();
	static native void setCAL(boolean flag);
	static native boolean getCAL();
	static native boolean getENV();
	static native String getSysInfo();

	static native void clear();
	static native void bufferClean();

	static native String visualQueueAll();
	static native String visualQueue();
	static native String visualBufferAll();
	static native String visualBuffer();

	static native boolean put(float[] data, int[] shape, int dataLen, int shapeLen, int usageNum, String name, boolean isBroad);
	static native boolean pop(String name);

	static native float[] getData(String name);
	static native int[] getShape(String name);
	static native boolean duple(String name, int usage);
	static native boolean broad(String name, boolean broad);
	static native boolean reshape(String name, int[] shape, int shapeLen);
	
	// 단항
	static native boolean abs(String a, String out);
	static native boolean neg(String a, String out);
	static native boolean square(String a, String out);
	static native boolean sqrt(String a, String out);
	static native boolean log(String a, String out);
	static native boolean log2(String a, String out);
	static native boolean ln(String a, String out);
	static native boolean reciprocal(String a, String out);
	static native boolean sin(String name, String out);
	static native boolean cos(String name, String out);
	static native boolean tan(String name, String out);
	static native boolean step(String name, String out);
	static native boolean sigmoid(String name, String out);
	static native boolean tanh(String name, String out);
	static native boolean ReLu(String name, String out);
	static native boolean leakReLu(String name, String out);
	static native boolean softplus(String name, String out);
	static native boolean exp(String name, String out);
	static native boolean round(String name, String out);
	static native boolean ceil(String name, String out);
	static native boolean floor(String name, String out);
	static native boolean not(String a, String out);
	static native boolean deg2rad(String name, String out);
	static native boolean rad2deg(String name, String out);

	
	
	// 이항
	static native boolean add(String a, String b, String out);
	static native boolean sub(String a, String b, String out);
	static native boolean mul(String a, String b, String out);
	static native boolean div(String a, String b, String out);
	static native boolean pow(String a, String b, String out);
	static native boolean mod(String a, String b, String out);
	static native boolean gt(String a, String b, String out);
	static native boolean lt(String a, String b, String out);
	static native boolean ge(String a, String b, String out);
	static native boolean le(String a, String b, String out);
	static native boolean eq(String a, String b, String out);
	static native boolean ne(String a, String b, String out);
	static native boolean and(String a, String b, String out);
	static native boolean or(String a, String b, String out);
	
	//축 통합
	static native boolean sum(String a, String out, int axis);
	static native boolean mean(String a, String out, int axis);
	static native boolean var(String a, String out, int axis);
	static native boolean std(String a, String out, int axis);
	static native boolean max(String a, String out, int axis);
	static native boolean min(String a, String out, int axis);
	
	//축 독립
	static native boolean accumulate(String a, String out, int axis);
	static native boolean compress(String a, String out, int axis);	
	static native boolean expand(String a, String out, int axis, int expandN);
	static native boolean argMax(String a, String out, int axis);
	static native boolean argMin(String a, String out, int axis);	
	static native boolean axisMax(String a, String out, int axis);
	static native boolean axisMin(String a, String out, int axis);	
	static native boolean transpose(String name, String out, int axis1, int axis2);

	//내적
	static native boolean dot(String a, String b, String out);
	static native boolean matmul(String a, String b, String out);	
	
	//신경망
	static native boolean affine(String x, String w, String b, String out);
	static native boolean softmax(String name, String out, int axis);
	static native boolean mse(String a, String b, String out);
	static native boolean cee(String a, String b, String out);	
	static native boolean im2col1D(String input, String kernel, String out, int pad, int stride);
	static native boolean col2im1D(String input, String kernel, String out, int oL, int pad, int stride);
	static native boolean im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW);
	static native boolean col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW);

	
	
	
	
}
