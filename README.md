
# Blood Pressure Estimation via rPPG

본 프로젝트는 얼굴 영상에서 비접촉 방식으로 **rPPG 신호를 추출**하고, 이를 기반으로 **혈압(SBP/DBP)을 예측**하는 시스템입니다.

RetinaFace를 이용한 얼굴 탐지 + DeepPhys 기반 rPPG 추출 + CNN-LSTM 기반 혈압 예측으로 구성되어 있습니다.

---

## 🧠 프로젝트 구성

```
📁 blood-pressure-project/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # PyTorch 관련 의존성
├── retinaface/             # 얼굴 검출 (서브모듈 또는 직접 설치)
├── *.pth                   # 모델 가중치 (rPPG, BP predictor)
└── README.md
```

---

## 🔧 설치 방법 (Linux 기준)

### 1. Python 3.10+ 설치

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev build-essential
```

### 2. 가상환경 생성 및 활성화

```bash
python3.10 -m venv env
source env/bin/activate
```

### 3. RetinaFace 설치

```bash
git clone https://github.com/serengil/retinaface.git
cd retinaface
pip install -r requirements.txt
pip install .
cd ..
```

### 4. 프로젝트 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📹 실행 방법

```bash
python main.py /path/to/your/video.mp4
```

예시 출력:

```
==============================
 최종 혈압 예측 결과
==============================
수축기 혈압 (SBP): 117.35 mmHg
이완기 혈압 (DBP): 76.98 mmHg
==============================
```

---

## 💡 사용 모델

- **rPPG 추출 모델**: DeepPhys
- **혈압 예측 모델**: CNN + LSTM
- 사전 학습된 PyTorch `.pth` 파일 필요

---

## 📦 requirements.txt 예시

```txt
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.14.0
gdown>=3.10.1
Pillow>=5.2.0
opencv-python>=3.4.4
tqdm
```

> `tensorflow`, `keras`는 `retinaface/requirements.txt`에 포함됨

---

## ✍️ 작성자

- Maintainer: 사용자님
- License: MIT (또는 자유롭게 설정 가능)
