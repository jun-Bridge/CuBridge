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

