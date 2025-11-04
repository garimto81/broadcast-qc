# PRD: 학습 기반 방송 QC 시스템 MVP

## 1. 제품 개요

### 1.1 문제 정의
- **현재**: 규칙 기반 분석으로 트랜지션/테이블 구분 정확도 낮음 (~60%)
- **목표**: 머신러닝으로 95%+ 정확도 달성

### 1.2 솔루션
딥러닝 기반 자동 방송 QC 시스템
- 트랜지션 패턴 자동 학습
- 테이블 타입 자동 분류
- 실시간 QC 리포트 생성

## 2. 기술 아키텍처

### 2.1 2단계 분류 시스템

```
Stage 1: 트랜지션 감지 (Binary Classification)
├── Input: 비디오 프레임
├── Model: CNN (MobileNet/EfficientNet)
├── Output: 트랜지션 여부 (Yes/No) + 신뢰도
└── Threshold: 0.8 이상

Stage 2: 테이블 분류 (Multi-class Classification)
├── Input: 씬 구간 프레임
├── Model: CNN + LSTM (시계열 특성 활용)
├── Classes: Feature_A, Feature_B, Virtual, Unknown
└── Output: 테이블 타입 + 신뢰도
```

### 2.2 핵심 기술 스택

**백엔드:**
- Python 3.9+
- PyTorch/TensorFlow 2.x
- OpenCV 4.x
- FastAPI

**모델:**
- Transfer Learning (ImageNet pre-trained)
- Fine-tuning on WSOP dataset
- Model quantization for speed

**데이터:**
- 라벨링 도구: Label Studio
- 증강: Albumentations
- 저장: HDF5/TFRecord

## 3. MVP 구현 단계

### Phase 1: 데이터 준비 (1주)

#### 3.1.1 데이터 수집
```python
# 필요 데이터 구조
training_data = {
    'transitions': {
        'wsop_logo': ['frame_001.jpg', 'frame_002.jpg', ...],  # 100+ 샘플
        'diagonal_pattern': [...],  # 100+ 샘플
        'black_fade': [...],  # 50+ 샘플
        'non_transition': [...]  # 500+ 샘플 (negative)
    },
    'tables': {
        'feature_a': [...],  # 200+ 샘플 (탑뷰)
        'feature_b': [...],  # 200+ 샘플 (사이드뷰)
        'virtual': [...],    # 200+ 샘플 (그래픽 오버레이)
        'unknown': [...]     # 100+ 샘플
    }
}
```

#### 3.1.2 라벨링 전략
1. **자동 추출**: 현재 영상에서 프레임 추출
2. **수동 라벨링**: 초기 1000개 프레임
3. **반자동화**: Active Learning으로 효율적 라벨링

### Phase 2: 모델 개발 (2주)

#### 3.2.1 트랜지션 감지 모델
```python
class TransitionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # transition/non-transition
        )

    def forward(self, x):
        features = self.backbone.features(x)
        pooled = F.adaptive_avg_pool2d(features, 1).squeeze()
        return self.classifier(pooled)
```

#### 3.2.2 테이블 분류 모델
```python
class TableClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # EfficientNet-B0 for better accuracy
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Temporal features (LSTM for sequence)
        self.lstm = nn.LSTM(1280, 256, batch_first=True)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_sequence):
        # x_sequence: [batch, seq_len, channels, height, width]
        batch_size, seq_len = x_sequence.shape[:2]

        # Extract features for each frame
        features = []
        for i in range(seq_len):
            frame_features = self.backbone.features(x_sequence[:, i])
            pooled = F.adaptive_avg_pool2d(frame_features, 1).squeeze()
            features.append(pooled)

        # LSTM on sequence
        features = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(features)

        # Use last hidden state
        return self.classifier(lstm_out[:, -1])
```

### Phase 3: 학습 파이프라인 (1주)

#### 3.3.1 학습 설정
```yaml
training_config:
  transition_detector:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    optimizer: AdamW
    scheduler: CosineAnnealingLR
    augmentation:
      - RandomBrightness(0.2)
      - RandomContrast(0.2)
      - RandomRotation(5)

  table_classifier:
    epochs: 100
    batch_size: 16
    sequence_length: 5  # 5 frames per sample
    learning_rate: 0.0005
    optimizer: AdamW
    loss: FocalLoss  # for imbalanced classes
```

#### 3.3.2 평가 메트릭
```python
metrics = {
    'transition_detector': {
        'accuracy': 0.95,  # 목표
        'precision': 0.93,
        'recall': 0.97,
        'f1_score': 0.95
    },
    'table_classifier': {
        'accuracy': 0.92,  # 목표
        'per_class_accuracy': {
            'feature_a': 0.95,
            'feature_b': 0.90,
            'virtual': 0.93,
            'unknown': 0.85
        },
        'confusion_matrix': [...]
    }
}
```

### Phase 4: 통합 시스템 (1주)

#### 3.4.1 추론 파이프라인
```python
class WSPOQCPipeline:
    def __init__(self, transition_model, table_model):
        self.transition_detector = transition_model
        self.table_classifier = table_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process_video(self, video_path):
        # 1. 트랜지션 감지
        transitions = self.detect_transitions(video_path)

        # 2. 씬 분할
        scenes = self.segment_scenes(transitions)

        # 3. 테이블 분류
        for scene in scenes:
            scene['table_type'] = self.classify_table(scene)

        # 4. QC 리포트 생성
        return self.generate_report(scenes, transitions)

    def detect_transitions(self, video_path):
        """프레임별 트랜지션 감지"""
        cap = cv2.VideoCapture(video_path)
        transitions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_tensor = self.preprocess_frame(frame)

            # Inference
            with torch.no_grad():
                output = self.transition_detector(input_tensor)
                prob = torch.softmax(output, dim=1)

            if prob[0, 1] > 0.8:  # transition class
                transitions.append({
                    'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                    'confidence': prob[0, 1].item()
                })

        return self.merge_consecutive_transitions(transitions)
```

## 4. 데이터 수집 전략

### 4.1 초기 데이터셋 구축

**Step 1: 샘플 비디오 분석**
```python
# 자동 프레임 추출 도구
def extract_training_frames(video_path, output_dir):
    """학습용 프레임 자동 추출"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 1초마다 프레임 추출
    frame_interval = int(fps)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 프레임 저장
            timestamp = frame_count / fps
            filename = f"frame_{timestamp:.2f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)

        frame_count += 1
```

**Step 2: 라벨링 도구 설정**
```json
{
  "label_studio_config": {
    "project_name": "WSOP_QC_Labeling",
    "label_config": "
      <View>
        <Image name='image' value='$image'/>
        <Choices name='transition' toName='image'>
          <Choice value='wsop_logo'/>
          <Choice value='diagonal_pattern'/>
          <Choice value='black_fade'/>
          <Choice value='not_transition'/>
        </Choices>
        <Choices name='table' toName='image'>
          <Choice value='feature_a'/>
          <Choice value='feature_b'/>
          <Choice value='virtual'/>
          <Choice value='unknown'/>
        </Choices>
      </View>
    "
  }
}
```

### 4.2 Active Learning 전략

```python
class ActiveLearningSelector:
    """불확실성 기반 샘플 선택"""

    def select_uncertain_samples(self, model, unlabeled_data, n_samples=100):
        uncertainties = []

        for data in unlabeled_data:
            with torch.no_grad():
                output = model(data)
                prob = torch.softmax(output, dim=1)

                # Entropy as uncertainty measure
                entropy = -torch.sum(prob * torch.log(prob + 1e-10))
                uncertainties.append((data, entropy.item()))

        # Sort by uncertainty (high to low)
        uncertainties.sort(key=lambda x: x[1], reverse=True)

        # Return top n most uncertain samples
        return [x[0] for x in uncertainties[:n_samples]]
```

## 5. 성능 최적화

### 5.1 모델 경량화
```python
# Quantization for faster inference
def quantize_model(model):
    """INT8 quantization"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model

# ONNX export for deployment
def export_to_onnx(model, sample_input):
    torch.onnx.export(
        model,
        sample_input,
        "wsop_qc_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
```

### 5.2 배치 처리 최적화
```python
class BatchProcessor:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process_frames_batch(self, frames):
        """GPU 배치 처리"""
        results = []

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i+self.batch_size]
            batch_tensor = torch.stack([self.preprocess(f) for f in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                results.extend(outputs.cpu().numpy())

        return results
```

## 6. 평가 및 검증

### 6.1 Cross-validation
```python
def k_fold_validation(dataset, model, k=5):
    """K-fold cross validation"""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        # Split data
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Train
        trained_model = train_model(model, train_subset)

        # Evaluate
        score = evaluate_model(trained_model, val_subset)
        scores.append(score)

        print(f"Fold {fold+1}: {score:.4f}")

    print(f"Average: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

### 6.2 Error Analysis
```python
class ErrorAnalyzer:
    """오분류 분석"""

    def analyze_errors(self, predictions, labels, data):
        errors = []

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if pred != label:
                errors.append({
                    'index': i,
                    'predicted': pred,
                    'actual': label,
                    'confidence': predictions[i].max(),
                    'data': data[i]
                })

        # Group errors by type
        error_matrix = self.create_confusion_matrix(predictions, labels)

        # Identify patterns
        patterns = self.identify_error_patterns(errors)

        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions),
            'confusion_matrix': error_matrix,
            'patterns': patterns
        }
```

## 7. 배포 계획

### 7.1 Docker 컨테이너화
```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/ /app/models/
COPY src/ /app/src/

WORKDIR /app

# Run API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 API 엔드포인트
```python
from fastapi import FastAPI, UploadFile, File
from typing import Dict

app = FastAPI()

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)) -> Dict:
    """비디오 QC 분석 API"""

    # Save uploaded file
    video_path = save_upload(file)

    # Run analysis
    pipeline = WSPOQCPipeline(
        transition_model=load_model('transition_detector.pth'),
        table_model=load_model('table_classifier.pth')
    )

    result = pipeline.process_video(video_path)

    return {
        'status': 'success',
        'transitions': result['transitions'],
        'scenes': result['scenes'],
        'summary': result['summary']
    }

@app.post("/train")
async def train_model(dataset_path: str, model_type: str):
    """모델 재학습 API"""

    # Load dataset
    dataset = load_dataset(dataset_path)

    # Train model
    if model_type == 'transition':
        model = train_transition_detector(dataset)
    else:
        model = train_table_classifier(dataset)

    # Save model
    model_path = save_model(model, model_type)

    return {
        'status': 'success',
        'model_path': model_path,
        'metrics': evaluate_model(model, dataset)
    }
```

## 8. 성공 지표 (KPIs)

### 8.1 정확도 목표
- **트랜지션 감지**: 95%+ accuracy
- **테이블 분류**: 92%+ accuracy
- **End-to-end**: 90%+ 전체 정확도

### 8.2 성능 목표
- **처리 속도**: 실시간의 10배 속도 (10시간 영상 → 1시간 처리)
- **GPU 사용**: RTX 3090에서 60fps 처리
- **CPU 사용**: 일반 PC에서 10fps 처리

### 8.3 사용성 목표
- **자동화율**: 95% 이상 자동 처리
- **수동 개입**: 5% 미만 재확인 필요
- **리포트 생성**: 즉시 (< 1초)

## 9. 리스크 및 완화 방안

### 9.1 데이터 부족
- **리스크**: 초기 학습 데이터 부족
- **완화**: 데이터 증강, Transfer Learning, Active Learning

### 9.2 새로운 패턴
- **리스크**: 학습하지 않은 새로운 트랜지션/테이블
- **완화**: Continuous Learning, 정기적 모델 업데이트

### 9.3 처리 속도
- **리스크**: 실시간 처리 불가능
- **완화**: Model Quantization, TensorRT, 배치 처리

## 10. 로드맵

### Week 1-2: 데이터 준비
- [ ] 프레임 추출 도구 개발
- [ ] 라벨링 도구 설정
- [ ] 초기 1000개 라벨링

### Week 3-4: 모델 개발
- [ ] 트랜지션 감지 모델
- [ ] 테이블 분류 모델
- [ ] 학습 파이프라인

### Week 5: 통합 및 테스트
- [ ] 파이프라인 통합
- [ ] 성능 테스트
- [ ] API 개발

### Week 6: 배포
- [ ] Docker 패키징
- [ ] 문서화
- [ ] 사용자 교육

## 11. 예산 추정

### 11.1 개발 비용
- 개발자 1명 × 6주 = $12,000
- GPU 서버 렌탈 = $500/월
- 라벨링 도구 = $100/월

### 11.2 운영 비용
- 클라우드 추론 서버 = $200/월
- 모델 업데이트 = $500/분기

### 11.3 ROI
- 수동 QC 시간 절감: 10시간 → 1시간 (90% 감소)
- 연간 절감액: $50,000+

## 12. 성공 기준

### 12.1 기술적 성공
- ✅ 5개 트랜지션 모두 정확히 감지
- ✅ 6개 씬 모두 정확히 구분
- ✅ 3가지 테이블 타입 95%+ 정확도

### 12.2 비즈니스 성공
- ✅ QC 시간 90% 감소
- ✅ 일관된 품질 기준 확보
- ✅ 확장 가능한 시스템 구축