#  Brain Tumor Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** model for **Brain Tumor Classification** using TensorFlow/Keras. The model is trained on grayscale MRI images to classify tumors into different categories.

---

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
Optimizer: Adam
Loss Function: SparseCategoricalCrossentropy
Performance Metrics: Accuracy
Batch Size: 32
Epochs: 10+
Training on Google Colab T4 GPU for faster convergence

---
## Results




