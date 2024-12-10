# 🎵 Music Genre Classification Using Machine Learning and Deep Learning Models

This project was developed by Naga Gayatri Bandaru and Shaik Faiz Ahamed. It focuses on classifying music tracks into their respective genres using a combination of machine learning and deep learning techniques. The dataset used is the GTZAN Music Genre Dataset, which contains 10 distinct genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock.
 

---

## 📌 Project Overview

Accurately categorizing music genres is a key task in Music Information Retrieval (MIR), with applications in music recommendation systems, playlist generation, and automated tagging for large music databases.

In this project, we explore multiple approaches to classify music genres using machine learning and deep learning models. The models are trained on audio features extracted from the GTZAN dataset, such as MFCCs (Mel-Frequency Cepstral Coefficients), which capture the timbral characteristics of music tracks.

This project was created to:

- **Learn and apply** machine learning and deep learning techniques.
- **Classify music genres** based on audio features.
- **Compare the performance** of different models like Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Convolutional Neural Networks (CNN).

## 🎯 Objectives
- **Explore** different machine learning and deep learning techniques for audio classification.
- **Classify music tracks** into their correct genres using audio features.
- **Compare the performance** of several models to identify the most effective approach.
- **Learn and apply** best practices in data preprocessing, feature extraction, model training, and evaluation.

## 📂 Dataset
### GTZAN Music Genre Dataset
**Total Tracks:** 1,000 audio files
**Duration:** 30 seconds per track
**Genres:**
Blues
Classical
Country
Disco
Hip-Hop
Jazz
Metal
Pop
Reggae
Rock
**Format:** WAV files sampled at 22,050 Hz
This dataset is widely used for benchmarking music genre classification models due to its simplicity and diversity of genres.

## ⚙️ Methodology
**1. Data Preprocessing**
**Label Encoding:** Convert categorical genre labels into numerical values.
**Feature Extraction:** Extract MFCCs (Mel-Frequency Cepstral Coefficients) for each track.
**Feature Scaling:** Normalize the extracted features using MinMaxScaler to ensure all values fall between 0 and 1.
**Train-Test Split:** Split the dataset into 70% for training and 30% for testing.
**2. Models Implemented**
We implemented and compared the following models:

Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
Convolutional Neural Networks (CNN)
Each model was evaluated based on accuracy, training performance, and generalization ability.

**3. Training and Evaluation**
Training: The models were trained on the extracted MFCC features for 150 epochs with early stopping applied to prevent overfitting.
Evaluation Metrics:
Training Accuracy
Test Accuracy
Confusion Matrix for detailed performance analysis

## 📊 Results
<img width="720" alt="Screenshot 2024-12-10 at 10 08 04" src="https://github.com/user-attachments/assets/30bde52b-fbe8-4387-8b9c-7dbf42b1b17e">

**Key Findings:**
**KNN** outperformed other models with a test accuracy of **96.3%**.
Logistic Regression and SVM demonstrated reasonable performance but were outperformed by KNN.
The CNN model showed potential, and future improvements like data augmentation and hyperparameter tuning could enhance its performance.

## 🚀 Future Work
To further improve the project, we plan to:

**Implement Data Augmentation:** Increase the dataset size by applying transformations such as pitch shifting, time stretching, and adding noise.
**Optimize CNN Architecture:** Experiment with deeper networks and advanced architectures like LSTM-CNN hybrids.
**Hyperparameter Tuning:** Fine-tune model parameters using techniques like Bayesian Optimization.
**Ensemble Methods:** Combine multiple models to achieve better generalization.

## 📦 Installation and Usage
**Prerequisites**
**Python 3.x**
**Libraries:**
****numpy**
****pandas**
****librosa**
**matplotlib**
**scikit-learn**
**tensorflow / keras**

## 🤝 Contributors
**Naga Gayatri Bandaru****
****Shaik Faiz Ahamed**

I have shared this project on GitHub to help others who are interested in **music information retrieval**, **audio classification**, and **machine learning**.
