# SignScribe  – Sign Language Detection Web App

This project is a simple web application that detects sign language gestures from webcam images using TensorFlow and served through Flask.

---

##  Features

- Detects 3 hand signs: `Hi`, `Best of Luck`, and `Love`
- Trained using custom hand gesture images
- Real-time prediction via webcam and browser
- Built with Python, Flask, HTML, CSS, and OpenCV

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (base64 image capture)
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow / Keras
- **Computer Vision**: OpenCV

---

## 📂 Folder Structure
sign_language_app/


│


├── static/ # CSS 


├── templates/


│ └── index.html # Main UI


├── dataset/ # Collected gesture images


│ ├── hi/


│ ├── best_of_luck/


│ └── love/


├── collect_images.py # Script to collect images via webcam


├── train_model.py # model training script


├── sign_model.h5 # Trained model file


├── app.py # Flask backend


└── README.md # Project info (this file)



