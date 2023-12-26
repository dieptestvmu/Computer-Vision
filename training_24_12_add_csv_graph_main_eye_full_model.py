import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Chuẩn bị dữ liệu
data_path = 'C:/tgmt_ytb1/img_training_19_12__remove_up_left'
image_size = (64, 64)

# Load dữ liệu và nhãn từ thư mục
def load_data():
    images = []
    labels = []
    for label in ['left', 'right', 'up', 'down', 'center', 'blink']:
        path = data_path + '/' + label + '/'
        for filename in os.listdir(path):
            img = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Tải dữ liệu
images, labels = load_data()

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Chuyển đổi nhãn sang dạng số nguyên
label_mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3, 'center': 4, 'blink': 5}
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])

# Chuyển đổi nhãn sang dạng one-hot encoding
num_classes = 6
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
batch_size = 64
epochs = 100
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Lưu mô hình vào tệp tin
model.save('C:/tgmt_ytb1/model_cnn_24_12_full_model_main_eye_add_csv_graph.h5')
# keras.saving.save_model(model)

# Lưu quá trình huấn luyện thành file CSV
df = pd.DataFrame(history.history)
df.to_csv('C:/tgmt_ytb1/training_history.csv', index=False)

# Vẽ đồ thị chính xác và mất mát
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('C:/tgmt_ytb1/accuracy.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('C:/tgmt_ytb1/loss.png')

# Đánh giá mô hình trên tập train
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print('Test loss:', train_loss, '<=> {:.2f}%'.format(train_loss *100))
print('Test accuracy:', train_accuracy, '<=> {:.2f}%'.format(train_accuracy *100))

# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss, '<=> {:.2f}%'.format(loss *100))
print('Test accuracy:', accuracy, '<=> {:.2f}%'.format(accuracy *100))
