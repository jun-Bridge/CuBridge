# CuBridge 변경내역


BETA
    - 자바와 cpp/cuda c 연동 확인
    - queue/map/buffer 삼중 메모리 구조 구축
    - put/cal/get 구조 구축
    - cpu_ram/gpu_vram 이원화 구조 구축
    - 자동 환경 인식
    - 함수 추가 : auto, cal, ram, env, sysinfo, clear, visual queue/map, put, get, 
    - 연산 추가
        - 단항 : abs, neg, square, sqrt, log, log_2, ln, reciprocal, sin, cos, tan, step, sigmoid, tanh, relu, leakRelu, softplus, exp, round, ceil, floor, not
        - 이항 : add, sub, mul, div, pow, mod, gt, lt, ge, le, eq, ne, and, or
        - 축 연산 : sum, mean, var, std, max, min
        - 행렬 연산 : transpose, dot
        - 신경망 연산 : affine, cee, mse, softmax

1.0
    - Beta 안정화 : *메모리 누수 정리*, 구조 최적화, map 삭제 및 queue/buffer 이원화 정립, 최적화를 위한 vram 정책 삭제, 속도 증가 및 안정화
    - 종속성 문제 해결 : cuda 시스템 분리 및 안정화
    - 배포 최적화 : jar 파일 하나만 있으면 됩니다!
    - 함수 추가 : duple, broad, visualBuffer/All
    - 연산 추가 : 
        - 축 연산을 '축 통합'과 '축 독립' 연산으로 분리
            - 축 통합 연산은 지정 축 이하의 축을 전부 통합해서 연산합니다
            - 축 독립 연산은 지정 축 만 연산합니다
            - 축 통합 : sum, mean, var, std, max, min
            - 축 독립 : accumulate, compress, expand, argmax, argmin, axisMax, axisMin
            - 전치 최적화 : 속도 향상 및 다축 지원
            - 내적 이원화 : dot/matmul 분리 및 형상에 따라 내부에서 바이패스로 연결
            - softmax 연산 축 추가


