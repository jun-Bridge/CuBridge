# CuBridge

**CuBridge**는 Java와 CUDA를 연결하여 고성능 수치 연산을 수행하는 경량 텐서 연산 엔진입니다.  
모든 연산은 Tensor 객체를 기반으로 실행되며, CPU와 GPU를 자동으로 감지하여 연산합니다.

> A lightweight Java tensor engine with automatic GPU acceleration via CUDA.
> 
> 해당 프로젝트는 Alphabet Bridge 시리즈의 첫 프로젝트입니다.

---

- 이 라이브러리는 자바에서 쿠다를 돌릴 수 있게 해 줍니다.
- 기존 라이브러리와는 달리 JDK에 대해서만 종속성을 가집니다.
- 쿠다 및 GPU가 없는 환경에서도 CPU로 실행될 수 있습니다.
- 기본적인 행렬 연산 및 자동 브로드캐스트, 신경망 기능 일부 지원(추후 추가 예정)

- [CuBridge 변경 내역 보기](Legacy/History.md)

---

##  주요 특징

- Java + CUDA 연결 구조
- Tensor 클래스 기반 고속 수치 연산 엔진
- 환경적응형 라이브러리. GPU 여부, CUDA 여부에 따라 연산 설정 분기
- DLL 기반 JNI 연동
- 사이즈가 1이 아닌 배수일 경우에도 브로드캐스트가 가능합니다

---

## 설치 방법

### Eclipse 환경에서 `CuBridge.jar` 추가

1. Eclipse에서 상단 메뉴 **`Project → Properties → Java Build Path`** 열기  
2. **`Libraries`** 탭으로 이동 → **Classpath** 선택 후 **[Add External JARs...]** 클릭  
3. 다운로드한 `CuBridge.jar` 파일을 프로젝트 폴더로 복사한 뒤 선택  
4. **Apply → Close** 클릭 후 프로젝트 리빌드

---

### IntelliJ IDEA 환경에서 `CuBridge.jar` 추가

1. IntelliJ에서 상단 메뉴 **`File → Project Structure (Ctrl+Alt+Shift+S)`** 열기  
2. 왼쪽 메뉴에서 **`Modules`** 선택 → 우측 상단의 **`Dependencies`** 탭 클릭  
3. **`+ (Add)` → JARs or Directories...** 선택  
4. 다운로드한 `CuBridge.jar` 파일을 선택 후 **OK**  
5. 적용 순서:  
   - `Scope`는 **Compile** 로 설정  
   - **Apply → OK** 클릭  
6. 프로젝트를 **Rebuild (Ctrl+F9)** 하여 반영 확인  

---

##  예제 코드 1 (정석 작성)

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


## 예제 코드 2 (축약 작성)

```java
CuBridge cb = CuBridge.getInstance();

Tensor t1 = Tensor.randn(3, 1);
Tensor t2 = Tensor.randn(1, 3);

cb.dotI(t1, t2).printData();

//put 없이 연산자에 바로 텐서 입력 가능
//연산자 뒤에 I를 붙이면 바로 Tensor 반환
```




##  라이선스

이 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.

