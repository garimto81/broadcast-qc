# Re:View Documentation

## 📚 문서 구조 (v3.0)

### 핵심 문서 (3개)

#### 1. [PRD_MASTER.md](./PRD_MASTER.md)
**통합 제품 명세서**
- MVP, Pro, Enterprise 3가지 구현 경로 포함
- 단계별 기능, 기술 스택, 예산 정의
- 아키텍처 및 로드맵

#### 2. [local_dev_setup_guide.md](./local_dev_setup_guide.md)
**개발 환경 구축 가이드**
- Windows 환경 설정
- Step-by-step 설치 가이드
- 문제 해결 방법

#### 3. [optimized_config_for_your_system.md](./optimized_config_for_your_system.md)
**사용자 시스템 최적화 구성**
- AMD Ryzen 9 5950X + RTX 3090 최적화
- GPU 가속 설정
- 성능 벤치마크

---

## 🗂️ 문서 정리 내역

### 통합/삭제된 문서
- ~~prd.md~~ → PRD_MASTER.md에 통합
- ~~prd_v2.0.md~~ → PRD_MASTER.md에 통합
- ~~prd_mvp_minimal_cost.md~~ → PRD_MASTER.md에 통합
- ~~tech_architecture.md~~ → PRD_MASTER.md에 통합

### 통합 이유
1. **중복 제거**: 동일 내용이 여러 문서에 분산
2. **관리 효율성**: 단일 소스로 버전 관리
3. **접근성 향상**: 한 곳에서 모든 정보 확인

---

## 🚀 Quick Start

### MVP 개발 시작
```bash
# PRD_MASTER.md의 [MVP] 섹션 참조
# local_dev_setup_guide.md 따라 환경 구축
```

### GPU 가속 버전
```bash
# PRD_MASTER.md의 [Pro] 섹션 참조
# optimized_config_for_your_system.md 참조
```

---

## 📊 문서 관계도

```
PRD_MASTER.md (메인 명세)
    ├── [MVP Path] → local_dev_setup_guide.md
    ├── [Pro Path] → optimized_config_for_your_system.md
    └── [Enterprise Path] → 클라우드 배포 (추후 작성)
```

---

최종 업데이트: 2025-11-03