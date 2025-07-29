# 🧬 Breast Cancer Prediction App

This Streamlit-based web application predicts whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** using a **Logistic Regression model** trained on the **Breast Cancer Wisconsin Diagnostic Dataset**. The app offers real-time prediction, visualizations, and correlation insights to help understand the relationships among features.

---

## 🔍 Project Overview

Breast cancer is one of the most common cancers affecting women worldwide. Early detection and diagnosis significantly increase the chances of survival. This app:

- Accepts tumor measurement inputs.
- Applies machine learning (Logistic Regression).
- Predicts the tumor type in real-time.
- Shows prediction probability.
- Displays correlation and feature visualizations.

The goal is to provide an easy-to-use diagnostic assistant for educational, research, or awareness purposes.

---

## 🚀 Features

- 🎛️ **User-Friendly Sidebar:** Input five tumor characteristics using sliders.
- 🔮 **Real-Time Prediction:** Instant output showing whether the tumor is malignant or benign.
- 📈 **Prediction Probability:** Displays confidence for both classes.
- 📊 **Correlation Heatmaps:** Understand relationships between features and diagnosis.
- 📉 **Model Accuracy Plot:** Bar plot to visualize how well the model performs.
- 🔍 **Pairplot Visualization:** Displays scatter plots to visualize feature separability.

---

## 🧠 Machine Learning Workflow

### 📁 Dataset Used
- **Source:** Breast Cancer Wisconsin Diagnostic Dataset (UCI)
- **Rows:** 569 samples
- **Features:** 30 numeric features (only 5 used here for clarity)

### 🔄 Preprocessing Steps:
1. Removed `Unnamed: 32` and `id` columns.
2. Converted categorical target: `M` → 1, `B` → 0.
3. Selected understandable features:
   - `radius_mean`
   - `texture_mean`
   - `perimeter_mean`
   - `area_mean`
   - `smoothness_mean`
4. Applied `LocalOutlierFactor` to remove outliers.
5. Used `StandardScaler` for feature normalization.

### 🧪 Model Training
- Algorithm: **Logistic Regression**
- `train_test_split`: 80% training, 20% testing
- Accuracy scored and plotted
- Model saved using `joblib`

---

## 📊 Visualizations

1. **Correlation with Target Variable:**
   - Bar chart showing each feature's correlation with the diagnosis column.
2. **Full Heatmap:**
   - Shows feature-to-feature and feature-to-target correlations.
3. **Selected Feature Heatmap:**
   - Focuses only on the 5 selected features and their relationship with diagnosis.
4. **Pairplot:**
   - Scatterplot matrix differentiating benign and malignant tumors.
5. **Model Accuracy Plot:**
   - Bar plot showing performance of the logistic regression model.

---

## 💻 Technologies Used

| Tool            | Purpose                      |
|-----------------|------------------------------|
| Python          | Programming language         |
| Streamlit       | Web app development          |
| Pandas, NumPy   | Data manipulation            |
| Seaborn, Matplotlib | Data visualization        |
| Scikit-learn    | Machine Learning             |
| Joblib          | Model serialization          |


