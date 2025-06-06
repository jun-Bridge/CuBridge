import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

class CuBridgeJNI {
	
	private static String loadDll(String fileName) throws IOException {
	    File dir = new File(System.getProperty("java.io.tmpdir"), "cubridge_dll_tmp");
	    if (!dir.exists()) dir.mkdirs();

	    File dllFile = new File(dir, fileName);

	    if (!dllFile.exists()) {
	        InputStream in = null;

	        in = CuBridgeJNI.class.getResourceAsStream("/dll/" + fileName);

	        if (in == null)
	            throw new FileNotFoundException("JAR 내부에서 '" + fileName + "' 파일을 찾을 수 없습니다.");
	        
	        try (OutputStream out = new FileOutputStream(dllFile)) {
	            byte[] buffer = new byte[1024];
	            int len;
	            
	            while ((len = in.read(buffer)) != -1)
	                out.write(buffer, 0, len);
	            
	        }

	    }
        dllFile.deleteOnExit();

	    return dllFile.getAbsolutePath();
	}
	
	static {
		try {
			System.load(loadDll("CuBridgeDriver.dll"));	
			init(loadDll("CuBridgeCudaC.dll"));
		} catch (IOException e) {
			System.out.println("CuBridge DLL 로딩 실패" + e);
		}
		
	}

	//환경
	static native void init(String path);
	
	static native void refresh();
	
	static native void setAuto(boolean flag);
	
	static native void setRAM(boolean flag);
	
	static native void setCAL(boolean flag);
	
	static native boolean getRAM();
	
	static native boolean getCAL();
	
	static native boolean getENV();

	static native String getSysInfo();
	
	// 제어
	static native void clear();

	static native void cachedClean();

	static native int getQueueSize();

	static native String visualQueue(); // 최선

	static native String visualQueue(String name); // 내부 특정 텐서
	
	static native String visualQueueAll(); // 전체
	
	static native int getMemSize();

	static native String visualMem(String name); // 내부 특정 텐서
	
	static native String visualMemAll(); // 전체

	static native boolean put(double[] data, int[] shape, int dataLen, int shapeLen, int usageNum, String name, boolean isBroad);

	static native boolean pop(String name);

	static native double[] getData();

	static native int[] getShape();
	
	static native boolean duple(String name, int usage);

	// 산술 연산
	// 단항
	static native boolean abs(String a, String out);	//하나 검색해서 그 이름으로 저장

	static native boolean neg(String a, String out);

	static native boolean square(String a, String out);

	static native boolean sqrt(String a, String out);

	static native boolean log(String a, String out);

	static native boolean log2(String a, String out);

	static native boolean ln(String a, String out);

	static native boolean reciprocal(String a, String out);

	// 이항
	static native boolean add(String a, String b, String out);

	static native boolean sub(String a, String b, String out);

	static native boolean mul(String a, String b, String out);

	static native boolean div(String a, String b, String out);

	static native boolean pow(String a, String b, String out);
	
	static native boolean mod(String a, String b, String out);
	
	static native boolean dot(String a, String b, String out);

	// 논리연산
	// 비트연산
	static native boolean gt(String a, String b, String out);

	static native boolean lt(String a, String b, String out);
	
	static native boolean ge(String a, String b, String out);
	
	static native boolean le(String a, String b, String out);

	static native boolean eq(String a, String b, String out);

	static native boolean ne(String a, String b, String out);
	
	static native boolean and(String a, String b, String out);

	static native boolean or(String a, String b, String out);
	
	static native boolean not(String a, String out);

	// 통계 연산
	// 텐서
	static native boolean sum(String a, int axis, String out);

	static native boolean mean(String a, int axis, String out);

	static native boolean max(String a, int axis, String out);

	static native boolean min(String a, int axis, String out);

	static native boolean var(String a, int axis, String out);
	
	static native boolean std(String a, int axis, String out);

	static native boolean argmax(String a, int axis, String out);

	static native boolean argmin(String a, int axis, String out);

	// 함수
	static native boolean sin(String name, String out);

	static native boolean cos(String name, String out);

	static native boolean tan(String name, String out);

	static native boolean step(String name, String out);

	static native boolean sigmoid(String name, String out);
	
	static native boolean tanh(String name, String out);

	static native boolean reLu(String name, String out);

	static native boolean leakReLu(String name, String out);

	static native boolean softmax(String name, String out);

	static native boolean softplus(String name, String out);
	
	static native boolean exp(String name, String out);

	static native boolean round(String name, String out);
	
	static native boolean ceil(String name, String out);

	static native boolean floor(String name, String out);

	static native boolean transpose(String name, String out);

	static native boolean compress(String a, int axis, boolean avg, String out);
	
	static native boolean expand(String a, int axis, int n, String out);
	
	// 신경망 특화
	static native boolean mse(String yh, String y, String out);
	
	static native boolean cee(String yh, String y, String out);
	
	static native boolean affine(String x, String w, String b, String out);
	
}
