# CuBridge

**CuBridge**는 Java와 CUDA를 연결하여 고성능 수치 연산을 수행하는 경량 텐서 연산 엔진입니다.  
모든 연산은 Tensor 객체를 기반으로 실행되며, RAM/VRAM을 자동으로 분리하여 GPU 연산을 지원합니다.

> A lightweight Java tensor engine with automatic GPU acceleration via CUDA.

---

##  주요 특징

- Java + CUDA 연결 구조
- Tensor 클래스 기반 고속 수치 연산 엔진
- 환경적응형 라이브러리. GPU 여부, CUDA 여부에 따라 연산 설정 분기
- VRAM 크기에 따른 데이터 주 저장 설정 분기 // 4GB, VRAM or RAM
- Javadoc 연동 및 IDE 지원
- DLL 기반 JNI 연동

---

##  설치 방법 (Eclipse 기준)

### 1. `cubridge.jar` 프로젝트에 추가

1. Eclipse에서 `Project → Properties → Java Build Path` 열기
2. `Libraries` 탭으로 이동 → classpath 클릭 후 **[Add External JARs...]** 클릭
3. 다운로드에서 프로젝트 폴더로 옮긴 `cubridge.jar` 선택 → 적용

---

### 2. DLL(native library) 경로 연결

1. `Libraries` 탭에서 `cubridge.jar` 우측 ▼ 클릭
2. **Native library location → Edit**
3. `External Folder...` 선택
4. `cubridge.dll`이 들어 있는 폴더 선택
5. OK → Apply

> 주의: `dll`은 `jar` 내부가 아니라 별도 폴더에 있어야 합니다!

---

### 3. Javadoc 연결

1. 동일하게 `cubridge.jar` 우측 ▼ 클릭
2. **Javadoc location → Edit**
3. 선택지 중 하나:
   - `Javadoc in archive` → `cubridge-javadoc.jar` 선택
   - 또는 `Javadoc in folder` → `docs/` 가 저장되어 있는 폴더 선택
4. 확인 → Apply

이제 IDE에서 함수 이름에 마우스를 올리면 설명이 바로 표시됩니다.

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

##  디렉토리 구조 예시

```
CuBridge/
├─ src/
│  ├─ CuBridgeJNI.java
│  ├─ CuBridge.java
│  ├─ Tensor.java
├─ cubridge.jar
├─ lib /
│  ├─cubridge.dll
├─ doc /
│  ├─cubridge-javadoc.jar
├─ README.md
├─ LICENSE
```

---

##  라이선스

이 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.

