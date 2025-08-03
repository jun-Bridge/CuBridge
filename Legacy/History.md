# CuBridge 변경 이력

## BETA

- 자바와 cpp/cuda c 연동 확인  
- queue / map / buffer 삼중 메모리 구조 구축  
- put / cal / get 구조 구축  
- cpu_ram / gpu_vram 이원화 구조 구축  
- 자동 환경 인식  
- 함수 추가:
  - auto, cal, ram, env, sysinfo, clear
  - visual queue/map, put, get
- 연산 추가:
  - **단항 연산**:  
    abs, neg, square, sqrt, log, log_2, ln, reciprocal, sin, cos, tan, step, sigmoid, tanh, relu, leakRelu, softplus, exp, round, ceil, floor, not
  - **이항 연산**:  
    add, sub, mul, div, pow, mod, gt, lt, ge, le, eq, ne, and, or
  - **축 연산**:  
    sum, mean, var, std, max, min
  - **행렬 연산**:  
    transpose, dot
  - **신경망 연산**:  
    affine, cee, mse, softmax

---

## Version 1.0

- Beta 안정화:
  - 메모리 누수 정리
  - 구조 최적화
  - map 삭제 및 queue/buffer 이원화 정립
  - VRAM 정책 제거로 속도 및 안정성 향상
- 종속성 문제 해결:
  - CUDA 시스템 분리 및 안정화
- 배포 최적화:
  - jar 파일 하나로 실행 가능
- 함수 추가:
  - duple, broad, visualBuffer / All
- 연산 개선:
  - 축 연산을 **축 통합**과 **축 독립**으로 분리
    - 축 통합 연산: sum, mean, var, std, max, min
    - 축 독립 연산: accumulate, compress, expand, argmax, argmin, axisMax, axisMin
    - 통합 연산은 지정 축 이하 모든 축을 통합하여 연산
    - 독립 연산은 지정 축만 계산
  - 전치 최적화:
    - 속도 향상 및 다축 지원
  - 내적 이원화:
    - dot / matmul 분리, 형상에 따라 내부 바이패스 처리
  - softmax에 축 인자 추가

---

## Version 1.1

- 버그 수정
  - transpose에서 -1 자동 지정 시 축 반전 문제 수정

- 연산 함수 추가
  - rad2deg, deg2rad: 각도-라디안 변환 지원
  - im2col1D, col2im1D: 1D convolution 입출력 재구성 함수
  - im2col2D, col2im2D: 2D convolution 입출력 재구성 함수
  - reshape: 기존 텐서의 shape 및 길이를 동적으로 갱신

- Tensor 클래스 확장
  - 문자열 기반 텐서 초기화 생성자 `Tensor(String[][])` 및 `Tensor(String[][], float)` 추가

---

## Version 1.1.1

- 버그 수정

1. **pop() 함수의 큐 이름 불일치 버그 수정**
   - 내부에서 `""`이 아닌 `genRandomName()`을 호출하여 큐 최상단이 아닌 내부의 실제 텐서 이름들을 탐색하는 문제가 발생
   - 이로 인해 `pop()`이 항상 실패하는 치명적인 버그를 수정하였습니다

2. **Broadcast 방향 오류 수정**
   - 이항 연산 시 broadcast 기준 축이 잘못 설정되어, 반복 방향이 역으로 적용되는 현상 해결
   - 예: `{3,2}`와 `{1,6}` 연산에서 의도와 달리 열 방향으로 확장되던 문제를 행 방향으로 정상 처리되도록 수정

---

## Version 1.2

- 버그 수정 및 기능 개선

1. 상수(Constant Tensor) 체계 도입

  주요 특징
  - 이름이 `'_'`로 시작하고 `usageCount < 0`일 경우 상수로 자동 인식됨
  - 상수는 **사용자 정의 가능** (예: `_VAR1`, `_CUSTOM_CONST` 등)
  - 모든 상수는 `broadcast=true`로 자동 설정되며, 사용자 상수는 수동 설정 가능
  - 상수는 절대 수정 불가:
    - `setUsage`, `setBroad`, `setReshape` 호출 시 경고 출력 후 무시
    - `smartPush()` 또는 동일 이름으로 put할 경우 에러 반환
  - 상수는 연산 결과물의 출력 이름으로 사용 불가
    - `cb.exp("a", "_PI")` -> 에러 발생 (`_PI`는 상수이므로 덮어쓰기 불가)

#### 내장 상수 목록

| 이름         | 값              | 비고                     |
|--------------|------------------|--------------------------|
| `_ZERO`      | 0.0              | 기본 0 상수              |
| `_ONE`       | 1.0              | 항등 연산 등             |
| `_TWO`       | 2.0              | 제곱, 지수               |
| `_THREE`     | 3.0              |                          |
| `_FOUR`      | 4.0              |                          |
| `_FIVE`      | 5.0              |                          |
| `_SIX`       | 6.0              |                          |
| `_SEVEN`     | 7.0              |                          |
| `_EIGHT`     | 8.0              |                          |
| `_NINE`      | 9.0              |                          |
| `_HALF`      | 0.5              | 평균, 정규화            |
| `_PI`        | 3.14159265359    | 삼각 함수 등            |
| `_E`         | 2.718281         | 지수 함수               |
| `_EPSILON`   | 1e-6             | 수치 비교                |
| `_RATE`      | 0.001            | 학습률 등                |
| `_NEG`       | -1.0             | 음수 상수                |
| `_HUNDRED`   | 100.0            | 백분율 등                |
| `_MAXPIXEL`  | 255.0            | 이미지 정규화            |

2. Visual 시리즈 개선

  - `visualQueue()`는 일반 변수만 출력
  - `visualQueueAll()`는 상수 포함 전체 출력
  - 출력 형태 개선 : `"Queue Size : 20 (Const : 18, Var : 2)"` 형태로 상세 구분
  - Buffer 출력은 기존과 동일  

3. 에러 메시지 형식 통일
  - 모든 연산 함수에서 에러 메시지 형식 개선
  - 입력 인자, 출력 이름 등 포함하여 정확한 실패 원인 안내

예시:
  [ERROR][EXP][Cannot Execute][Tensor val1, _PI]

---

## Version 1.3

- 텐서 즉응 입출력 기능 추가

- 텐서 즉시입력 함수 추가
  - 이제 put() 함수를 따로 기술할 필요 없이, 연산자에 직접 Tensor 객체를 매개변수로 넣을 수 있습니다.
    - ex) cb.add(Tensor a, Tensor b), cb.add(Tensor a, String b)...
    - 두번째 경우, 텐서와 상수 혹은 미리 넣어 둔 텐서를 문자열로 지정하여 연산하는 것이 가능합니다.

- 텐서 즉시출력 함수 추가
  - 이제 get() 함수를 따로 기술할 필요 없이, 연산자에서 직접 Tensor 객체를 반환받을 수 있습니다.
  - 모든 즉시출력 함수는 연산자의 이름 뒤에 I를 붙인 형태이며, transpose(T)와 im2col, col2im(그대로)은 예외입니다.
    - ex) Tensor c = cb.addI(String a, String b)...

- 텐서 즉시 입출력 함수 추가
  - 앞선 두 기능을 동시에 사용할 수 있습니다.
  - put()과 get() 없이, 오직 연산자 만으로 GPU 가속 연산이 가능합니다.
    - ex) Tensor c = cb.addI(Tensor a, Tensor b)
    