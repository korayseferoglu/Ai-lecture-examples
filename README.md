# 🧬 Breast Cancer Classification (Colab)

A complete machine learning workflow for **Breast Cancer (Wisconsin Diagnostic)** dataset.  
This project demonstrates the full pipeline from **data exploration (EDA)** to **model training and prediction** —  
replicable both on **Google Colab** and **locally (VS Code / terminal)**.

---

## 📚 About the Dataset

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569  
- **Features:** 30 numerical attributes derived from cell nucleus morphology  
- **Target:**
  - `0` → Malignant (cancerous)
  - `1` → Benign (non-cancerous)

Each sample represents measurements taken from a fine needle aspirate (FNA) of a breast mass.  
The goal is to predict whether the mass is **malignant** or **benign** based on its features.

---

## 🧠 Project Overview

| Stage | Script | Description |
|--------|--------|-------------|
| 🔍 **EDA** | `src/eda.py` | Generates visualizations and descriptive statistics (`reports/figures/`) |
| 🧩 **Training** | `src/train.py` | Trains multiple models and saves the best one as `models/best_model.joblib` |
| 🎯 **Prediction** | `src/predict.py` | Loads the trained model and performs sample predictions via CLI |

---

## ▶️ Run on Google Colab

If you want to test or experiment quickly, you can run the project entirely in **Google Colab** — no installation required.

**Step 1.** Open a new Colab notebook and copy–paste the following cell:

```python
!git clone https://github.com/korayseferoglu/Ai-lecture-examples.git
%cd Ai-lecture-examples
!pip install -q -r requirements.txt

# 1️⃣ Generate EDA figures (saved to reports/figures/)
!python src/eda.py

# 2️⃣ Train models and save the best one (models/best_model.joblib)
!python src/train.py

# 3️⃣ Example prediction (0 = malignant, 1 = benign)
!python src/predict.py --input "mean radius=14.0,mean texture=20.0,mean perimeter=90.0,mean area=600.0,mean smoothness=0.1"
