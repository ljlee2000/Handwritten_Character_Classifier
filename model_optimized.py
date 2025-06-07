# model_optimized.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 데이터 로딩 함수 (간단화)
def load_data(folder):
    images, labels = [], []
    class_names = sorted(os.listdir(folder))
    label_dict = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        for fname in os.listdir(os.path.join(folder, class_name)):
            img = load_img(os.path.join(folder, class_name, fname), target_size=(28, 28), color_mode='grayscale')
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label_dict[class_name])

    return np.array(images), to_categorical(labels), label_dict

# 데이터 준비
X, y, label_dict = load_data('./handwritten_sample_english/')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1
)
datagen.fit(X_train)

# 콜백
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
]

# 최적화된 CNN 모델
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          validation_split=0.1, epochs=30, callbacks=callbacks)

# 평가
loss, acc = model.evaluate(X_test, y_test)
print("최종 테스트 정확도: ", acc)
