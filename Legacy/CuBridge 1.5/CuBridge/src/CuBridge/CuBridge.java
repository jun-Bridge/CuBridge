package CuBridge;

import CuBridge.CuBridgeJNI;

import java.util.UUID;

public class CuBridge extends SystemOps implements AlgebraOps, AudioOps, AxisCascadeOps,
													AxisOps, BinaryOps, ImageOps, ScalarOps,
													UnaryOps, UtilityOps{
	private static final CuBridge instance = new CuBridge();

	private CuBridge() {
		loadConst();
	}

	private void loadConst() {
	    put(1.0f, "_ONE", -1);
	    put(2.0f, "_TWO", -1);
	    put(3.0f, "_THREE", -1);
	    put(4.0f, "_FOUR", -1);
	    put(5.0f, "_FIVE", -1);
	    put(6.0f, "_SIX", -1);
	    put(7.0f, "_SEVEN", -1);
	    put(8.0f, "_EIGHT", -1);
	    put(9.0f, "_NINE", -1);
	    put(0.0f, "_ZERO", -1);
	    put(0.5f, "_HALF", -1);
	    put(100.0f, "_HUNDRED", -1);
	    put(255.0f, "_MAXPIXEL", -1);
	    put(-1.0f, "_NEG", -1);
	    put(1e-6f, "_EPSILON", -1);
	    put(0.001f, "_RATE", -1);
	    put(3.14159265359f, "_PI", -1);
	    put(2.718281f, "_E", -1);
	}

	/**
	 * Returns the singleton instance of CuBridge.
	 *
	 * @return the global CuBridge instance (singleton)
	 */
	public static CuBridge getInstance() {
		return instance;
	}
}
