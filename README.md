# 🖋️ 손글씨 문자 분류기 (Handwritten Character Classifier)

이 프로젝트는 간단한 CNN(합성곱 신경망)을 사용하여  
숫자(0-9), 영문자(A-C) 으로 구성된 손글씨 이미지 데이터를 분류하는 딥러닝 모델입니다.

---

## 📦 데이터 구성

데이터는 다음과 같은 폴더 구조를 가집니다:


```

handwritten\_sample\_english/
├── 0/
├── 1/
├── ...
├── A/
├── B/
├── C/

````

- 각 폴더에는 28x28 크기의 흑백 `.png` 이미지가 50장씩 포함됩니다.
- 클래스 수: 총 13개 (`0`~`9`, `A`, `B`, `C`)

---

## 🧠 사용 기술

- Python 3
- TensorFlow / Keras
- scikit-learn
- Matplotlib

---

## 🚀 실행 방법

1. 이 저장소를 클론하거나 다운로드
2. 필요한 라이브러리 설치:

```bash
pip install tensorflow numpy matplotlib scikit-learn
````

3. Jupyter Notebook 또는 Python 파일 실행:

```bash
jupyter notebook final_handwriting_classifier.ipynb
```

또는

```bash
python korean_handwriting_classifier.py
```

---

## 📈 결과 예시

예측 결과 시각화:
![Figure_1](https://github.com/user-attachments/assets/f8afe948-964f-48d9-9720-ef9b440f62e0)

```
T:C | P:C     T:4 | P:4     T:8 | P:8     ...
```

이미지를 예측한 결과가 실제 라벨과 얼마나 잘 일치하는지를 시각적으로 보여줍니다.

---

## ✅ 주요 기능

* CNN 기반 이미지 분류기
* 커스텀 이미지 폴더 구조 대응
* 라벨 자동 인식 및 변환
* 정확도 평가 및 시각화 포함

---

## 📂 파일 설명

| 파일명                                  | 설명                |
| ------------------------------------ | ----------------- |
| `korean_handwriting_classifier.py`   | 전체 파이썬 실행 파일      |
| `final_handwriting_classifier.ipynb` | 주피터 노트북 버전        |
| `handwritten_sample_english/`        | 훈련/테스트용 샘플 이미지 폴더 |




## 📊 모델 성능 요약

이 프로젝트에서는 CNN 기반 모델로 숫자(0~9)와 한글 자모(ㄱ~ㅎ)를 분류하는 OCR 시스템을 구축했습니다.  
다양한 성능 향상 기법을 적용하여 정확도를 지속적으로 개선했습니다.

| 구성 방법 | 설명 | 테스트 정확도 |
|-----------|------|----------------|
| 기본 CNN | 2 Conv + Dense 구조 | 약 92.5% |
| + Dropout, BatchNorm | 과적합 방지 + 안정화 | 약 94.7% |
| + Data Augmentation | 이미지 다양성 증가 | 약 96.1% |
| + EarlyStopping, ReduceLROnPlateau | 조기 종료 + 학습률 튜닝 | 약 96.5% |
| ✅ 최종 조합 결과 | 위 요소 모두 포함 | **96~97%** |

> 데이터: MNIST(60k) + 한글 자모 28x28 이미지 각 클래스당 100장  
> 클래스 수: 총 24개 (숫자 10 + 한글 자모 14개)

### 🎯 자주 혼동되는 클래스 예시
- `3 vs 8`, `ㄱ vs ㄲ`, `ㄴ vs ㅁ`, `ㄹ vs ㄷ`

### 📈 예측 시각화 (Figure_1.png)
