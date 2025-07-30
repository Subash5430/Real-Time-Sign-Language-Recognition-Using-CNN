 🧠 Sign Language Recognition System

A real-time sign language recognition system that uses a Convolutional Neural Network (CNN) with an accuracy of **99.28%**, integrated with OpenCV for live video capture, and deployed as a web application using Flask.


 📌 Features

- 🖐️ Real-time hand gesture recognition from webcam
- 🎯 CNN-based model trained on ASL alphabet with 99.28% accuracy
- 🖼️ OpenCV for video stream processing
- 🌐 Flask-based lightweight web application
- 📤 Outputs translated text and optional text-to-speech
- ✨ Support for special gestures like `space` and `nothing`

 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript (optional enhancements)
- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, Mediapipe (if used)
- **Deep Learning**: TensorFlow / Keras (CNN Model)

 📂 Project Structure

sign-language-recognition/
├── static/
│ └── index.html
├── model/
│ └── asl_model.h5
├── dataset/
│ └── (ASL alphabet images)
├── app.py
├── utils.py
├── demo.py
├── train_model.py
├── requirements.txt
└── README.md

 🚀 Getting Started

 1. Clone the Repository

git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model (Optional)
If you wish to retrain:

python train_model.py
4. Run the Flask App
python app.py
Navigate to http://127.0.0.1:5000/ in your browser to start using the app.
