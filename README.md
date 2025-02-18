#  Brain Tumor Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** model for **Brain Tumor Classification** using TensorFlow/Keras. The model is trained on grayscale MRI images to classify tumors into different categories.

---

## Dataset
Below is the link of Kaggle Dataset used in this project

-https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

##  Features
-  **Deep Learning Model**: CNN-based classifier built with TensorFlow/Keras
-  **Image Preprocessing**: Uses **Rescaling, Augmentation, and Normalization**
-  **GPU Acceleration**: Runs efficiently on **Google Colab**
-  **Deployment Ready**: Converted to **TensorFlow.js** for web applications

---

## Dataset
- The dataset consists of **MRI images** of brain tumors.
- Images are **preprocessed** to ensure a consistent input size of **225x225 (Grayscale).**
- The dataset is split into:
  - **Training Set**
  - **Validation Set**
  - **Testing Set**
- The dataset contains **4 classes**:
  - **Glioma**
  - **Meningioma**
  - **Pituitary Tumor**
  - **No Tumor**

---

##  Model Architecture
The CNN model consists of:
1. **Convolutional Layers (Conv2D + ReLU)**
2. **Batch Normalization**
3. **Max Pooling Layers**
4. **Fully Connected Dense Layers**
5. **Dropout for Regularization**
6. **Softmax Activation for Multi-class Classification**

---
## Training Process 
-Optimizer: Adam
-Loss Function: SparseCategoricalCrossentropy
-Performance Metrics: Accuracy
-Batch Size: 32
-Epochs: 10+
-Training on Google Colab T4 GPU for faster convergence

---
## Results
The model has the following Results
-Training Accuracy: 96.22%
-Validation Accuracy: 88.7%
-Testing ccuracy: 84.7%

---
## Accuracy and Loss Curves 
![image](https://github.com/user-attachments/assets/c2f0e6b3-1312-4f37-aa1a-d52e58b9feed)
![image](https://github.com/user-attachments/assets/40cc58c0-3852-4fb4-b5cf-f525e3b8bc8c)

## Installation & Usage
-**1. Clone the Repository**
  -git clone https://github.com/your-username/Brain-Tumor-Classification.git
  -cd Brain-Tumor-Classification
-**2. Install Dependencies**
  -pip install -r requirements.txt
-**3. Train the Model**
  -python train.py
-**4. Evaluate the Model**
  -python evaluate.py








