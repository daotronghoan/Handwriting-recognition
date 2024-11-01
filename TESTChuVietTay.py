import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Hàm để tiền xử lý ảnh
def preprocess_image(image_path):
    # Đọc ảnh từ file
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_img = img.copy()  # Lưu ảnh gốc để hiển thị sau
    # Resize ảnh về kích thước 28x28
    img = cv2.resize(img, (28, 28))
    # Chuyển đổi ảnh thành numpy array và chuẩn hóa
    img = img.astype('float32') / 255.0
    # Thêm một chiều để phù hợp với input của mô hình (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return original_img, img


# Tải mô hình đã lưu
model = load_model('handwritten_model.h5')

# Đường dẫn tới các ảnh viết tay để kiểm tra
image_paths = ['C:/Users/ADMIN/Pictures/Chu0.png', 'C:/Users/ADMIN/Pictures/ChuN.png',
               'C:/Users/ADMIN/Pictures/ChuP.png']

# Dự đoán nhãn cho từng ảnh và hiển thị ảnh trước và sau tiền xử lý
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for image_path in image_paths:
    original_img, processed_img = preprocess_image(image_path)

    # Dự đoán nhãn cho ảnh đã tiền xử lý
    prediction = model.predict(processed_img)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_letter = alphabet[predicted_label[0]]

    # Hiển thị ảnh gốc và ảnh sau khi tiền xử lý
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_img[0, :, :, 0], cmap='gray')
    plt.title(f'Chữ: {predicted_letter}')
    plt.axis('off')

    plt.show()

    print(f'Image: {image_path}, Chữ: {predicted_letter}')
