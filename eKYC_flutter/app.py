from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from utils import check


origin_img = cv2.imread("/home/dotronghiep/Documents/AIoT_Lab/eKYC/khanh1.jpg")

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    image_bytes = request.get_data()
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    result = check(origin_img, image_np)
    return jsonify({'result': result})


if __name__ == '__main__':
    # app.run(host="172.17.0.1", port=8000, debug=True)
    app.run(host="localhost", port=8000, debug=True)
