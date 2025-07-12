# EEG-Based Schizophrenia Detection: A Comparative Analysis of Machine Learning and Deep Learning Models

This repository contains the code and resources for the research paper titled, *"Comparative Analysis of Machine Learning and Deep Learning Models for EEG-Based Schizophrenia Detection"*. This project provides a comprehensive analysis of various machine learning and deep learning models for detecting schizophrenia using EEG data.

---

## 📜 Abstract

Psychiatric disorders demand enhanced diagnostics. This research conducts a comparative analysis of machine learning and deep learning models for the detection of a specific mental disorder through EEG analysis. Leveraging a dataset of EEG signals from schizophrenic patients, we employ eight distinct models, including Random Forest, K-Nearest Neighbors (KNN), XGBoost, Logistic Regression, Support Vector Machine (SVM), Decision Tree, and a Convolutional Neural Network (CNN). Our feature extraction process involves signal processing techniques, and the models are rigorously evaluated on their accuracy, precision, and recall. The study aims to provide comparative insights into the efficiency of different modeling approaches in the challenging task of mental disorder detection using EEG data.

---

## 🛠️ Methodology

The methodology for this project involves a comprehensive analysis of Electroencephalogram (EEG) data to distinguish between healthy individuals and those diagnosed with schizophrenia. The key steps are outlined below and visualized in the following flowchart:

![Methodology Flowchart](Methodology.jpg)

### Data Preprocessing
* **Data Collection**: The dataset was compiled from a peer-reviewed journal, incorporating a patient group of 14 individuals with paranoid schizophrenia and a control group of 14 healthy individuals.
* **Data Filtering**: EEG data was filtered to isolate five distinct frequency bands using a second-order Butterworth filter.
* **Epoching**: EEG signals were segmented into fixed-length epochs of 5 seconds with a 1-second overlap.
* **Labeling**: The dataset was categorized into healthy and patient groups for supervised learning.

### Feature Extraction
A set of 7201 features was derived from various mathematical functions applied to the EEG epochs. These features include:
* Mean
* Standard Deviation
* Peak-to-Peak
* Variance
* Minimum
* Maximum
* Argmin
* Argmax
* Mean Square
* Root Mean Square (RMS)
* Absolute Differences
* Skewness
* Kurtosis

### Model Training and Evaluation
* **Train-Test Split**: An 80:20 train-test split was implemented for robust model evaluation.
* **Machine Learning Models**:
    * Logistic Regression 
    * K-Nearest Neighbors (KNN) 
    * Support Vector Machine (SVM) with Linear and RBF kernels 
    * Random Forest 
    * XGBoost 
    * Decision Tree 
* **Deep Learning Model**:
    * A three-layered 1D Convolutional Neural Network (CNN) was implemented.
* **Evaluation Metrics**: Models were evaluated based on accuracy, precision, and recall.

---

## 📊 Results

The performance of the models was assessed based on their ability to predict and classify schizophrenia from the test dataset. The results are summarized below:

### Accuracy Comparison

| Model               | Accuracy (%) |
| :------------------ | :----------: |
| SVM (RBF)           | 96.56        |
| XGBoost             | 97.36        |
| Random Forest       | 96.04        |
| SVM (Linear)        | 92.01        |
| Logistic Regression | 90.40        |
| Decision Tree       | 88.34        |
| K-Nearest Neighbors | 83.92        |
| CNN                 | 95.00        |

### Model Performance Metrics

| Model               | Precision | Recall | Accuracy |
| :------------------ | :-------: | :----: | :------: |
| Logistic Regression |   0.917   | 0.907  |  0.904   |
| K-Nearest Neighbors |   0.839   | 0.874  |  0.839   |
| SVM (Linear)        |   0.932   | 0.922  |  0.920   |
| SVM (RBF)           |   0.975   | 0.962  |  0.966   |
| Random Forest       |   0.950   | 0.978  |  0.960   |
| XGBoost             |   0.976   | 0.976  |  0.974   |
| Decision Tree       |   0.898   | 0.884  |  0.883   |
| CNN                 |   0.993   | 0.747  |  0.950   |

---

## ⚙️ Dependencies

To run this project, you will need the following libraries:

* mne
* pandas
* numpy
* matplotlib
* scikit-learn
* xgboost
* tensorflow

You can install the necessary packages using pip:
```bash
!pip install mne
!pip install scikit-learn xgboost
