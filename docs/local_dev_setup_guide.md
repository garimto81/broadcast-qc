# Re:View MVP - 로컬 개발 환경 설정 가이드

**Version**: 1.0
**Platform**: Windows 11
**Total Setup Time**: ~2시간

---

## 📋 사전 요구사항 체크리스트

```yaml
최소 사양:
  ✅ Windows 10/11 (64bit)
  ✅ 8GB RAM (16GB 권장)
  ✅ 100GB 여유 공간
  ✅ Intel i5 이상
  ✅ 인터넷 연결 (초기 설정용)
```

---

## 🚀 Step-by-Step 설치 가이드

### Step 1: Python 3.11 설치

```powershell
# 1. Python 다운로드
# https://www.python.org/downloads/windows/
# "Windows installer (64-bit)" 선택

# 2. 설치 시 반드시 체크:
# ✅ "Add Python to PATH"
# ✅ "Install for all users"

# 3. 설치 확인
python --version
# 출력: Python 3.11.x

# 4. pip 업그레이드
python -m pip install --upgrade pip
```

### Step 2: Node.js 20 LTS 설치

```powershell
# 1. Node.js 다운로드
# https://nodejs.org/en/download/
# "Windows Installer (.msi) 64-bit" 선택

# 2. 설치 (기본 옵션으로 진행)

# 3. 설치 확인
node --version
# 출력: v20.x.x

npm --version
# 출력: 10.x.x
```

### Step 3: FFmpeg 설치

```powershell
# 1. FFmpeg 다운로드
# https://github.com/BtbN/FFmpeg-Builds/releases
# "ffmpeg-master-latest-win64-gpl.zip" 다운로드

# 2. C:\ffmpeg 폴더 생성 후 압축 해제

# 3. 환경 변수 설정
# - Windows 키 + X → 시스템 → 고급 시스템 설정
# - 환경 변수 → Path 편집
# - 새로 만들기 → C:\ffmpeg\bin 추가

# 4. 새 PowerShell 창에서 확인
ffmpeg -version
```

### Step 4: Git 설치

```powershell
# 1. Git 다운로드
# https://git-scm.com/download/win
# 64-bit Git for Windows Setup

# 2. 설치 (기본 옵션으로 진행)

# 3. 설치 확인
git --version
# 출력: git version 2.x.x
```

### Step 5: VS Code 설치 (선택사항)

```powershell
# 1. VS Code 다운로드
# https://code.visualstudio.com/download
# "User Installer 64bit" 선택

# 2. 권장 확장 프로그램 설치:
# - Python
# - Pylance
# - ES7+ React/Redux/React-Native snippets
# - Prettier
# - SQLite Viewer
```

---

## 🔧 프로젝트 설정

### Step 1: 프로젝트 구조 생성

```powershell
# 프로젝트 디렉토리 생성
New-Item -Path "C:\broadcast-qc-mvp" -ItemType Directory
cd C:\broadcast-qc-mvp

# 하위 폴더 구조 생성
New-Item -Path "backend", "frontend", "data", "docs" -ItemType Directory
New-Item -Path "backend\app", "backend\uploads", "backend\processed" -ItemType Directory
New-Item -Path "data\db", "data\logs" -ItemType Directory
```

### Step 2: 백엔드 초기화

```powershell
cd C:\broadcast-qc-mvp\backend

# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 실행 정책 오류 시:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# requirements.txt 생성
@"
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python-headless==4.8.1.78
numpy==1.24.3
librosa==0.10.1
soundfile==0.12.1
sqlalchemy==2.0.23
aiosqlite==0.19.0
pydantic==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
"@ | Out-File -FilePath requirements.txt -Encoding UTF8

# 의존성 설치
pip install -r requirements.txt
```

### Step 3: 백엔드 기본 코드 생성

```powershell
# main.py 생성
@"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
from pathlib import Path
import shutil

app = FastAPI(title='Re:View MVP API')

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# 업로드 디렉토리 설정
UPLOAD_DIR = Path('uploads')
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get('/')
def read_root():
    return {'message': 'Re:View MVP API Running'}

@app.post('/api/upload')
async def upload_video(file: UploadFile = File(...)):
    try:
        # 파일 ID 생성
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f'{file_id}{file_extension}'

        # 파일 저장
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            'file_id': file_id,
            'filename': file.filename,
            'size': file_path.stat().st_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/health')
def health_check():
    return {'status': 'healthy'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
"@ | Out-File -FilePath app\main.py -Encoding UTF8
```

### Step 4: 프론트엔드 초기화

```powershell
cd C:\broadcast-qc-mvp

# React 앱 생성
npx create-react-app frontend --template typescript

cd frontend

# 추가 의존성 설치
npm install antd@5.11.0
npm install video.js@8.6.1
npm install @types/video.js
npm install recharts@2.9.0
npm install axios@1.6.2
npm install dayjs@1.11.10

# package.json에 proxy 추가
$packageJson = Get-Content package.json | ConvertFrom-Json
$packageJson | Add-Member -NotePropertyName "proxy" -NotePropertyValue "http://localhost:8000" -Force
$packageJson | ConvertTo-Json -Depth 10 | Set-Content package.json
```

### Step 5: 프론트엔드 기본 구조 설정

```powershell
# 컴포넌트 디렉토리 생성
cd src
New-Item -Path "components", "pages", "services", "utils", "types" -ItemType Directory

# App.tsx 수정
@"
import React from 'react';
import { ConfigProvider, Layout, Typography } from 'antd';
import 'antd/dist/reset.css';
import './App.css';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

function App() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ background: '#fff', padding: '0 24px' }}>
          <Title level={3} style={{ margin: '16px 0' }}>
            Re:View MVP - Broadcast QC Platform
          </Title>
        </Header>
        <Content style={{ padding: '24px' }}>
          <div style={{ background: '#fff', padding: 24, minHeight: 360 }}>
            <h2>Welcome to Re:View MVP</h2>
            <p>Upload a video to start quality control analysis.</p>
          </div>
        </Content>
        <Footer style={{ textAlign: 'center' }}>
          Re:View MVP ©2025 - Broadcast Quality Control
        </Footer>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
"@ | Out-File -FilePath App.tsx -Encoding UTF8
```

---

## 🏃 개발 서버 실행

### 백엔드 서버 실행

```powershell
# Terminal 1
cd C:\broadcast-qc-mvp\backend
.\venv\Scripts\Activate.ps1
python app\main.py

# 출력:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete.
```

### 프론트엔드 서버 실행

```powershell
# Terminal 2
cd C:\broadcast-qc-mvp\frontend
npm start

# 브라우저 자동 열림: http://localhost:3000
```

---

## 🧪 설치 검증

### 1. API 테스트

```powershell
# PowerShell에서 API 테스트
Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method GET

# 예상 출력:
# status
# ------
# healthy
```

### 2. 브라우저 테스트

```
1. http://localhost:3000 접속
2. React 앱 정상 표시 확인
3. 콘솔에 에러 없음 확인 (F12)
```

---

## 🛠️ 문제 해결

### Python 관련

```powershell
# pip 설치 실패 시
python -m ensurepip --upgrade

# 가상환경 활성화 실패 시
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

# numpy 설치 오류 시
pip install numpy --upgrade --force-reinstall
```

### Node.js 관련

```powershell
# npm 캐시 문제
npm cache clean --force

# 의존성 충돌
rm -rf node_modules package-lock.json
npm install

# 포트 충돌 (3000번)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### FFmpeg 관련

```powershell
# PATH 인식 안 될 때
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 권한 문제
# 관리자 권한으로 PowerShell 실행
```

---

## 📦 백업 및 복원

### 프로젝트 백업

```powershell
# 전체 백업 (데이터 포함)
Compress-Archive -Path C:\broadcast-qc-mvp -DestinationPath "C:\backup\broadcast-qc-mvp-$(Get-Date -Format 'yyyyMMdd').zip"

# 코드만 백업 (node_modules, venv 제외)
robocopy C:\broadcast-qc-mvp C:\backup\broadcast-qc-mvp-code /E /XD node_modules venv __pycache__ .git uploads processed
```

### Git 설정

```powershell
cd C:\broadcast-qc-mvp
git init
git add .

# .gitignore 생성
@"
# Python
venv/
__pycache__/
*.pyc
.env
*.db

# Node
node_modules/
build/
.env.local

# Data
uploads/
processed/
*.mp4
*.avi
*.mov

# IDE
.vscode/
.idea/
*.swp
"@ | Out-File -FilePath .gitignore -Encoding UTF8

git commit -m "Initial MVP setup"
```

---

## 🚦 다음 단계

### 개발 시작

1. **비디오 분석 모듈 개발**
   - `backend/app/analysis.py` 생성
   - OpenCV 비디오 처리 구현

2. **데이터베이스 설정**
   - `backend/app/database.py` 생성
   - SQLAlchemy 모델 정의

3. **UI 컴포넌트 개발**
   - 업로드 컴포넌트
   - 비디오 플레이어
   - 타임라인 뷰

### 테스트 데이터

```powershell
# 샘플 비디오 다운로드 (테스트용)
Invoke-WebRequest -Uri "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4" -OutFile "C:\broadcast-qc-mvp\data\sample.mp4"
```

---

## 📞 지원

문제 발생 시:
1. 에러 메시지 전체 복사
2. 실행한 명령어 기록
3. Python/Node 버전 확인
4. 시스템 사양 정보

---

이 가이드를 따라하면 약 2시간 내에 전체 개발 환경을 구축할 수 있습니다.
각 단계별로 검증하며 진행하시기 바랍니다.