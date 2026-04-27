Handwritten Character Recognition using CNN
A deep learning project developed as part of the CodeAlpha Machine Learning Internship. This project uses a Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits from the MNIST dataset.

📌 Project Overview
Handwritten digit recognition is one of the most fundamental problems in computer vision. This project trains a CNN to classify grayscale images of handwritten digits (0–9) with high accuracy.
The model learns spatial features such as edges, curves, and digit structures using convolutional layers, making it significantly more effective than traditional machine learning approaches.

🚀 Features


CNN-based handwritten digit recognition


Trained on the MNIST dataset


Accuracy above 98.95%


Confusion matrix visualization


Prediction sample generation


Modular project architecture


Model saving and loading


Inference on custom handwritten images



🛠️ Technologies Used


Python


PyTorch


TorchVision


NumPy


Matplotlib


Seaborn


Scikit-learn


OpenCV



📂 Project Structure
CodeAlphaHandwrittenCharacterRecognition/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── notebooks/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── inference/
│
├── tests/
├── models/
├── outputs/
│   ├── plots/
│   ├── predictions/
│   └── confusion_matrices/
│
├── experiments/
├── train.py
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore

🧠 Model Architecture
Input (1×28×28)
   ↓
Conv2D (32 filters)
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2D (64 filters)
   ↓
ReLU
   ↓
MaxPool
   ↓
Flatten
   ↓
Fully Connected (128)
   ↓
Output Layer (10 classes)
📊 Results


Test Accuracy: ~98%


Loss: Low and stable after training


Performance: Excellent digit classification on unseen data


Generated Outputs


Training Loss Curve


Confusion Matrix


Sample Predictions



▶️ Installation
git clone https://github.com/yourusername/CodeAlphaHandwrittenCharacterRecognition.git
cd CodeAlphaHandwrittenCharacterRecognition
pip install -r requirements.txt
🏋️ Training
python train.py
This will:


Download MNIST


Train the CNN


Save the trained model


Generate evaluation outputs



🔍 Inference
python inference.py
The script loads the saved model and predicts digits from images stored in:
data/samples/

📁 Output Files
After training, the following files are generated automatically:

outputs/
├── plots/
│   └── loss_curve.png
├── predictions/
│   ├── prediction_0.png
│   └── ...
└── confusion_matrices/
    └── confusion_matrix.png

🎯 Applications


Optical Character Recognition (OCR)


Bank cheque digit recognition


Postal code automation


Form digitization


Educational AI systems



📈 Future Improvements


Extend to full handwritten alphabet recognition using EMNIST


Deploy as a Streamlit web application


Implement real-time webcam digit recognition


Upgrade to CRNN for word-level recognition



👨‍💻 Author
AREEBA KHAN
Developed as part of the CodeAlpha Machine Learning Internship Program.

⭐ Acknowledgements


CodeAlpha


PyTorch Foundation


The MNIST research community


If you found this project useful, consider giving it a star.
