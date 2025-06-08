from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)
model = load_model('sign_model.h5')  # Load your trained model

# Make sure this matches your class_indices from training
LABELS = ['Best of Luck', 'Hi', 'Love']  # Example — adjust this as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    npimg = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0

    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    print("🔍 Prediction probabilities:", prediction)

    label = LABELS[np.argmax(prediction)]
    return jsonify({'prediction': label})


if __name__ == '__main__':
    app.run(debug=True)
