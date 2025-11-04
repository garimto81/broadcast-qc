# 🎬 Broadcast QC - 방송 품질 관리 시스템

포커 방송(WSOP) 영상의 자동 품질 검수 및 분석 시스템

## 📁 프로젝트 구조

```
broadcast-qc/
│
├── src/                    # 소스 코드
│   ├── core/              # 핵심 모듈
│   ├── analyzers/         # 분석 엔진
│   ├── models/            # ML 모델 (향후 구현)
│   └── utils/             # 유틸리티
│
├── data/                  # 데이터
│   ├── samples/          # 샘플 비디오 및 이미지
│   └── training/         # 학습 데이터 (향후)
│
├── config/               # 설정 파일
├── scripts/              # 스크립트
├── docs/                 # 문서
├── tests/               # 테스트
├── output/              # 분석 결과
└── README.md           # 프로젝트 설명서
```

## 🚀 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 분석 실행
python src/analyzers/wsop_precise_analyzer.py data/samples/video.mp4
```

## 🎯 주요 기능

- 트랜지션 자동 감지
- 테이블 타입 분류 (Feature A/B, Virtual)
- 자막 추출 및 타임코드 생성
- QC 리포트 자동 생성

## 📝 라이선스

Internal Use Only
