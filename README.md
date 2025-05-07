# CuBridge

**CuBridge**는 Java와 CUDA를 연결하여 고성능 수치 연산을 수행하는 경량 텐서 연산 엔진입니다.  
모든 연산은 Tensor 객체를 기반으로 실행되며, RAM/VRAM을 자동으로 분리하여 GPU 연산을 지원합니다.

> A lightweight Java tensor engine with automatic GPU acceleration via CUDA.
> 
> **주의: 이 프로젝트는 아직 안정 버전이 아닙니다. 실사용 시 주의하십시오.**
> 해당 프로젝트는 Alphabet Bridge 시리즈의 첫 프로젝트입니다.

---

##  주요 특징

- Java + CUDA 연결 구조
- Tensor 클래스 기반 고속 수치 연산 엔진
- 환경적응형 라이브러리. GPU 여부, CUDA 여부에 따라 연산 설정 분기
- VRAM 크기에 따른 데이터 주 저장 설정 분기 // 4GB 기준, VRAM or RAM
- Javadoc 연동 및 IDE 지원
- DLL 기반 JNI 연동

---

##  설치 방법 (Eclipse 기준)

### `CuBridge.jar` 프로젝트에 추가

1. Eclipse에서 `Project → Properties → Java Build Path` 열기
2. `Libraries` 탭으로 이동 → classpath 클릭 후 **[Add External JARs...]** 클릭
3. 다운로드에서 프로젝트 폴더로 옮긴 `CuBridge.jar` 선택 → 적용

---

##  예제 코드

```java
CuBridge cb = CuBridge.getInstance();

Tensor t1 = new Tensor(new double[] {1, 2, 3}, new int[] {3, 1});
Tensor t2 = new Tensor(new double[] {1, 2, 3}, new int[] {1, 3});
Tensor t3 = new Tensor(new double[] {1, 2, 3}, new int[] {3});

cb.put(t1, "x").put(t2, "w").put(t3, "b");
cb.affine("x", "w", "b").get().printData();
//간단한 연산은 이름을 지정하지 않고 비워두어도 연산 가능
```

출력:
```
Tensor(shape=[3, 3]):
    [  2.000,  4.000,  6.000 ]
    [  3.000,  6.000,  9.000 ]
    [  4.000,  8.000, 12.000 ]
```

---

##  디렉토리 구조

```
CuBridge/
├─ src/
│  ├─ CuBridgeJNI.java
│  ├─ CuBridge.java
│  ├─ Tensor.java
├─ dll /
│  ├─CuBridgeDriver.dll
│  ├─CuBridgeCudaC.dll
├─ doc /
├─ CuBridgeJNI.class
├─ CuBridge.class
├─ Tensor.class
```

---

##  라이선스

이 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.

