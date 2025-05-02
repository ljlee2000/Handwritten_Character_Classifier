import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 데이터 로딩 함수
def load_data_from_folder(data_dir, img_size=(28, 28)):
    images = []
    labels = []

    class_names = sorted(os.listdir(data_dir))  # 예: ['A', 'B']
    label_dict = {name: i for i, name in enumerate(class_names)}  # {'A': 0, 'B': 1}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label_dict[class_name])

    return np.array(images), to_categorical(labels), label_dict

# ✅ 수정된 경로
data_path = './handwritten_sample_english/'
X, y, label_dict = load_data_from_folder(data_path)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("훈련 이미지 수:", X_train.shape[0])
print("테스트 이미지 수:", X_test.shape[0])
print("클래스 수:", len(label_dict))

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_dict), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print("테스트 정확도: ", test_acc)

# 예측 및 시각화
pred = model.predict(X_test)

n = 10
plt.figure(figsize=(15, 3))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    true_label = list(label_dict.keys())[np.argmax(y_test[i])]
    pred_label = list(label_dict.keys())[np.argmax(pred[i])]
    plt.title(f"T:{true_label}\\nP:{pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
