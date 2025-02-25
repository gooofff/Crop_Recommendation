from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/recommendation', methods=['POST'])
def result():
    data = request.json
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Giá trị đầu vào mẫu: [N, P, K, temperature, humidity, ph, rainfall]
    sample_input = [[data.get('N', 0), data.get('P', 0), data.get('K', 0), data.get('temperature', 0), data.get('humidity', 0), 
                     data.get('ph', 0), data.get('rainfall', 0)]]  # Thay bằng giá trị của bạn
    sample_input_scaled = scaler.transform(sample_input)

    predicted_crop = knn.predict(sample_input_scaled)
    # print("\nCây trồng được khuyến nghị:", predicted_crop[0])
    return jsonify({'result': predicted_crop[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
