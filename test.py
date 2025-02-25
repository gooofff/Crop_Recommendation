from flask import Flask, request, jsonify
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from crop_knn_prepocessing import preprocess_data  # Import file tiền xử lý

app = Flask(__name__)

@app.route('/recommendation', methods=['POST'])
def result():
    data = request.json
    # num1 = data.get('num1', 0)
    # num2 = data.get('num2', 0)
    # result = num1 + num2

    # Đường dẫn file dữ liệu
    file_path = "crop_recommendation.csv"  # Thay bằng đường dẫn thực tế

    # 1. Tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)

    # 2. Triển khai KNN
    # Chọn số lượng láng giềng k
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)

    # Huấn luyện mô hình
    knn.fit(X_train, y_train)

    # # 3. Đánh giá mô hình
    # y_pred = knn.predict(X_test)

    # print("\nĐộ chính xác của mô hình (Accuracy):", accuracy_score(y_test, y_pred))
    # print("\nBáo cáo phân loại (Classification Report):")
    # print(classification_report(y_test, y_pred))
    # print("\nMa trận nhầm lẫn (Confusion Matrix):")
    # print(confusion_matrix(y_test, y_pred))

    # 4. Dự đoán cây trồng (ví dụ sử dụng dữ liệu mới)
    # Giá trị đầu vào mẫu: [N, P, K, temperature, humidity, ph, rainfall]
    sample_input = [[data.get('N', 0), data.get('P', 0), data.get('K', 0), data.get('temperature', 0), data.get('humidity', 0), 
                     data.get('ph', 0), data.get('rainfall', 0)]]  # Thay bằng giá trị của bạn
    sample_input_scaled = scaler.transform(sample_input)

    predicted_crop = knn.predict(sample_input_scaled)
    # print("\nCây trồng được khuyến nghị:", predicted_crop[0])
    return jsonify({'result': predicted_crop[0]})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
