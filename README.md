<div align="center">

# 🧠 Parkinson's Disease Detection

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SVM](https://img.shields.io/badge/SVM-Linear_Kernel-9C27B0?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)

<br/>

> **Predicting Parkinson's Disease from voice measurements using Support Vector Machine**  
> with StandardScaler normalization and real-time custom input prediction.

<br/>

[🚀 Getting Started](#-getting-started) • [📊 Dataset](#-dataset) • [🤖 Model](#-model--evaluation) • [📈 Results](#-results) • [👤 Author](#-author)

---

</div>

<br/>

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [📁 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [🔬 How It Works](#-how-it-works)
- [🤖 Model & Evaluation](#-model--evaluation)
- [📈 Results](#-results)
- [🧪 Custom Prediction](#-custom-prediction)
- [⚠️ Known Issues & Notes](#️-known-issues--notes)
- [🔮 Future Improvements](#-future-improvements)
- [👤 Author](#-author)

<br/>

---

## 🔍 Overview

<table>
<tr>
<td>

Parkinson's Disease is a progressive neurological disorder affecting millions worldwide. Early detection is critical for effective treatment. This project builds a **Support Vector Machine (SVM)** classifier that detects Parkinson's Disease using **biomedical voice measurements** — analyzing jitter, shimmer, and other vocal features extracted from patient recordings.

The model achieves **87.17% accuracy** on both training and test data, and supports **real-time prediction** from custom voice input values.

</td>
</tr>
</table>

### ✨ Key Highlights

| 🏆 Feature | 📋 Detail |
|---|---|
| 🧠 Algorithm | Support Vector Machine (Linear Kernel) |
| 📦 Dataset Size | 195 voice recordings |
| 🎯 Task | Binary Classification (Parkinson's vs Healthy) |
| ⚖️ Class Distribution | 147 Parkinson's — 48 Healthy |
| 📐 Accuracy | 87.17% (Train & Test) |

<br/>

---

## 📊 Dataset

<div align="center">

```
📂 dataset.csv
├── 195 total voice recordings
├── 147 Parkinson's cases   (75.4%)
└── 48  Healthy cases       (24.6%)
```

</div>

> 📥 Download from [Kaggle — Parkinson's Disease Dataset](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)

### 🧾 Feature Description

| Feature | Description |
|---------|-------------|
| `MDVP:Fo(Hz)` | Average vocal fundamental frequency |
| `MDVP:Fhi(Hz)` | Maximum vocal fundamental frequency |
| `MDVP:Flo(Hz)` | Minimum vocal fundamental frequency |
| `MDVP:Jitter(%)` | Variation in fundamental frequency |
| `MDVP:Shimmer` | Variation in amplitude |
| `NHR / HNR` | Noise-to-harmonics ratio measures |
| `RPDE / DFA` | Nonlinear dynamical complexity measures |
| `spread1/2, D2, PPE` | Nonlinear measures of fundamental frequency |
| `status` | **Target** — `1` = Parkinson's, `0` = Healthy |

### 📉 Class Distribution

```
Parkinson's  ████████████████████████████████████████  75.4%
Healthy      ████████████░░                            24.6%
```

> ⚠️ **Note:** Dataset is imbalanced — Parkinson's cases dominate. Consider class weighting for improved recall on healthy patients.

<br/>

---

## 📁 Project Structure

```
📦 parkinsons-disease-detection/
│
├── 📄 dataset.csv                 ← Voice measurements dataset
├── 📓 parkinsons_detection.ipynb  ← Main Jupyter Notebook
└── 📝 README.md                   ← Project documentation
```

<br/>

---

## ⚙️ Tech Stack

<div align="center">

| Library | Version | Purpose |
|---------|---------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.8+ | Core language |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | 1.24+ | Array operations & reshaping |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | 2.0+ | Data loading & exploration |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | 1.0+ | SVM model, scaler & metrics |

</div>

### 📦 Installation

```bash
pip install numpy pandas scikit-learn
```

<br/>

---

## 🚀 Getting Started

### Option 1 — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/Hasnain006-nain/parkinsons-disease-detection.git
cd parkinsons-disease-detection

# 2. Install dependencies
pip install numpy pandas scikit-learn

# 3. Add dataset
# Place dataset.csv in the project root directory

# 4. Launch Jupyter Notebook
jupyter notebook parkinsons_detection.ipynb
```

### Option 2 — Google Colab ☁️

```python
from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/MyDrive/dataset.csv")
```

<br/>

---

## 🔬 How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                       PIPELINE OVERVIEW                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 📥 Load Data         →  Read dataset.csv with Pandas         │
│                                                                  │
│  2. 🔎 Explore Data      →  Shape, null check, class balance     │
│                                                                  │
│  3. 🔧 Preprocess        →  Drop 'name', split X / y            │
│                                                                  │
│  4. ✂️  Split Data        →  80% Train | 20% Test                │
│                                                                  │
│  5. 📏 Normalize         →  StandardScaler on Train & Test       │
│                                                                  │
│  6. 🤖 Train Model       →  SVC(kernel='linear').fit()           │
│                                                                  │
│  7. 🔮 Predict           →  model.predict(X_test)                │
│                                                                  │
│  8. 📈 Evaluate          →  Accuracy Score (Train & Test)        │
│                                                                  │
│  9. 🩺 Custom Input      →  Predict from manual voice values     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

<br/>

---

## 🤖 Model & Evaluation

### 📐 Support Vector Machine (Linear Kernel)

SVM is a powerful classification algorithm that finds the **optimal hyperplane** to separate classes. It is well-suited for:

- ✅ Small to medium-sized datasets
- ✅ High-dimensional feature spaces
- ✅ Medical diagnosis classification tasks
- ✅ Clear margin of separation between classes

### 🔢 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)
```

> StandardScaler normalizes all features to zero mean and unit variance — essential for SVM performance.

<br/>

---

## 📈 Results

<div align="center">

```
╔══════════════════════════════════════════════════════════╗
║              MODEL PERFORMANCE SUMMARY                   ║
╠══════════════════════════════════════════════════════════╣
║  Model           │  SVM  (Linear Kernel)                 ║
║  Train Accuracy  │  87.17%  █████████████████   ✅       ║
║  Test Accuracy   │  87.17%  █████████████████   ✅       ║
╚══════════════════════════════════════════════════════════╝
```

</div>

> 🔎 **Interpretation:** Equal train and test accuracy suggests a well-generalized model with no overfitting. Further improvement can be achieved by tuning the SVM kernel or handling class imbalance.

<br/>

---

## 🧪 Custom Prediction

The model supports live prediction from custom voice input values:

```python
input_data = (122.4, 148.65, 113.819, 0.00968, 0.00008, 0.00465,
              0.00696, 0.01394, 0.06134, 0.626, 0.03134, 0.04518,
              0.04368, 0.09403, 0.01929, 19.085, 0.458359, 0.819521,
              -4.075192, 0.33559, 2.486855, 0.368674)

# Output: Parkinson Disease ✅
```

```
if prediction[0] == 0:
    print("Healthy")
else:
    print("Parkinson Disease")
```

<br/>

---

## ⚠️ Known Issues & Notes

### ⚠️ Feature Names Warning
```
UserWarning: X does not have valid feature names,
but StandardScaler was fitted with feature names
```
**Fix:** Convert input to a DataFrame before transforming:
```python
input_df = pd.DataFrame([input_data], columns=X.columns)
std_data = scaler.transform(input_df)
```

### ⚖️ Class Imbalance
With 75% Parkinson's cases, the model may be biased. Consider:
```python
# Class weighting
model = SVC(kernel='linear', class_weight='balanced')
```

<br/>

---

## 🔮 Future Improvements

- [ ] 🔧 Try **RBF / Polynomial** SVM kernels for better accuracy
- [ ] 🔄 Apply **SMOTE** to handle class imbalance
- [ ] 🧪 Compare with **Random Forest** and **XGBoost**
- [ ] 📊 Add **ROC-AUC** curve and confusion matrix visualization
- [ ] 🌐 Deploy as a **web app** using Flask or Streamlit
- [ ] 💾 Save the trained model with `joblib` for production use

<br/>

---

## 👤 Author

<div align="center">

<br/>

```
╔════════════════════════════════════╗
║                                    ║
║         Hasnain Haider             ║
║                                    ║
║   Machine Learning Enthusiast      ║
║   Data Science | AI | Python       ║
║                                    ║
╚════════════════════════════════════╝
```

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/hasnain-machinelearning-engineer/)

<br/>

---

*© 2024 Hasnain Haider — Built for educational purposes in Medical ML & Healthcare AI*

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
