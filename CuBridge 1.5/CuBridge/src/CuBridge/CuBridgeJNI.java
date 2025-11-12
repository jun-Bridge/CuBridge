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
	
	//시스템
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
	static native boolean rsqrt(String a, String out);
	static native boolean sin(String name, String out);
	static native boolean cos(String name, String out);
	static native boolean tan(String name, String out);
	static native boolean sinh(String name, String out);
	static native boolean cosh(String name, String out);
	static native boolean tanh(String name, String out);
	static native boolean asin(String name, String out);
	static native boolean acos(String name, String out);
	static native boolean atan(String name, String out);
	static native boolean asinh(String name, String out);
	static native boolean acosh(String name, String out);
	static native boolean atanh(String name, String out);
	static native boolean step(String name, String out);
	static native boolean sigmoid(String name, String out);
	static native boolean relu(String name, String out);
	static native boolean leakRelu(String name, String out);
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
	static native boolean axisMax(String a, String out, int axis);
	static native boolean axisMin(String a, String out, int axis);
	static native boolean axisVar(String a, String out, int axis);
	static native boolean axisStd(String a, String out, int axis);
	static native boolean argMax(String a, String out, int axis);
	static native boolean argMin(String a, String out, int axis);

	//대수
	static native boolean l2normalize(String a, String out);
	static native boolean dot(String a, String b, String out);
	static native boolean matmul(String a, String b, String out);
	static native boolean transpose(String name, String out, int axis1, int axis2);
	static native boolean trace(String a, String out);
	static native boolean inverse(String a, String out);
	static native boolean eigen(String a, String outVal, String outVec);
	static native boolean svd(String a, String outU, String outS, String outVT);
	static native boolean det(String a, String out);
	static native boolean qr(String a, String outQ, String outR);
	static native boolean cholesky(String a, String out);
	static native boolean rank(String a, String out);
	static native boolean normalize(String a, String out, int axis);
	static native boolean standardize(String a, String out, int axis);
	static native boolean affine(String x, String w, String b, String out);
	static native boolean softmax(String name, String out, int axis);

	//유틸리티
	static native boolean clip(String a, String out, float alpha, float beta);
	static native boolean softClip(String a, String out, float alpha);
	static native boolean sigClip(String a, String out, float alpha);
	static native boolean tanhClip(String a, String out, float alpha);
	static native boolean logClip(String a, String out, float alpha);

	//오디오
	static native boolean preEmphasis(String a, String out, float alpha);
	static native boolean applyWindow(String a, String out, String windowName, int hopSize);
	static native boolean applyFilter(String a, String out, String filterName);
	static native boolean fft(String a, String out, int fftSize);
	static native boolean rfft(String a, String out, int fftSize);
	static native boolean ifft(String a, String out, int fftSize);
	static native boolean powfft(String a, String out, int fftSize);
	static native boolean magfft(String a, String out, int fftSize);
	static native boolean phasefft(String a, String out, int fftSize);
	static native boolean boost(String a, String out, int sampleRate, float lowCut, float highCut, float gain);
	static native boolean spectrogram(String a, String out);
	static native boolean dct(String a, String out, int nCoeffs);
	static native boolean mfcc(String a, String out, int nCoeffs);
	static native boolean makeMelFilter(String out, int nMels, int sampleRate, int fftSize);
	static native boolean makeBarkFilter(String out, int nBands, int sampleRate, int fftSize);
	static native boolean makeErbFilter(String out, int nBands, int sampleRate, int fftSize);
	static native boolean makeChromaFilter(String out, int nChroma, int sampleRate, int fftSize, float fRef);
	static native boolean makeGaussianWindow(String out, int winSize, float sigma);
	static native boolean makeRectWindow(String out, int winSize);
	static native boolean makeHannWindow(String out, int winSize);
	static native boolean makeHammingWindow(String out, int winSize);
	static native boolean makeBartlettWindow(String out, int winSize);
	static native boolean makeKaiserWindow(String out, int winSize, float beta);
	
	//스칼라
	static native boolean L1Norm(String a, String out);
	static native boolean L2Norm(String a, String out);
	static native boolean LinfNorm(String a, String out);
	static native boolean L1Dist(String a, String b, String out);
	static native boolean L2Dist(String a, String b, String out);
	static native boolean LinfDist(String a, String b, String out);
	static native boolean cosDist(String a, String b, String out);
	static native boolean cosSim(String a, String b, String out);
	static native boolean mse(String y, String label, String out);
	static native boolean bce(String y, String label, String out);
	static native boolean cee(String y, String label, String out);
	static native boolean mae(String y, String label, String out);
	static native boolean rmse(String y, String label, String out);
	static native boolean mape(String y, String label, String out);
	static native boolean focal(String y, String label, String out);
	static native boolean perplexity(String y, String label, String out);
	static native boolean dice(String y, String label, String out);
	static native boolean iou(String y, String label, String out);

	//이미지
	static native boolean rotate(String a, String out, float angle);
	static native boolean shift(String a, String out, int sW, int sH);
	static native boolean translate(String a, String out, int tW, int tH);
	static native boolean resize(String a, String out, float scaleW, float scaleH);
	static native boolean crop(String a, String out, int cH, int cW, int sH, int sW);
	static native boolean mask(String a, String out, int mH, int mW, int sH, int sW);
	static native boolean pad(String a, String out, int pH, int pW, float val);
	static native boolean boxBlur(String a, String out, int kSize);
	static native boolean gaussianBlur(String a, String out, int kSize);
	static native boolean medianBlur(String a, String out, int kSize);
	static native boolean flipH(String a, String out);
	static native boolean flipV(String a, String out);
	static native boolean grayScale(String a, String out);
	static native boolean chSplit(String a, String R, String G, String B);
	static native boolean chMerge(String r, String g, String b, String out);
	static native boolean im2col1D(String input, String kernel, String out, int pad, int stride);
	static native boolean col2im1D(String input, String kernel, String out, int oL, int pad, int stride);
	static native boolean im2col2D(String input, String kernel, String out, int padH, int padW, int strideH, int strideW);
	static native boolean col2im2D(String input, String kernel, String out, int oH, int oW, int padH, int padW, int strideH, int strideW);
}
