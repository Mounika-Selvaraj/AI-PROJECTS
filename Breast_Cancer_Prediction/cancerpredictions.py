import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🧬 Breast Cancer Prediction App")
st.markdown("This ML-powered app predicts whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset.")

# --- Load & Preprocess Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})

    y = df["diagnosis"]

    # Select a few understandable features
    selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    X = df[selected_features]

    # Remove outliers
    lof = LocalOutlierFactor()
    y_pred = lof.fit_predict(X)
    x_score = lof.negative_outlier_factor_
    lof_threshold = -2.5
    outlier_index = np.where(x_score < lof_threshold)[0]
    X = X.drop(index=outlier_index)
    y = y.drop(index=outlier_index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler, df, selected_features

X, y, scaler, full_df, selected_features = load_data()

# --- Train Model ---
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "model.pkl")
    return model, accuracy

model, accuracy = train_model()

# --- Sidebar Input ---
st.sidebar.header("Enter Tumor Measurements")

def get_user_input(features):
    input_dict = {}
    for feature in features:
        min_val = float(full_df[feature].min())
        max_val = float(full_df[feature].max())
        mean_val = float(full_df[feature].mean())
        input_dict[feature] = st.sidebar.slider(
            label=feature.replace("_", " ").capitalize(),
            min_value=round(min_val, 2),
            max_value=round(max_val, 2),
            value=round(mean_val, 2)
        )
    return pd.DataFrame(input_dict, index=[0])

user_input_df = get_user_input(selected_features)

# --- Scale & Predict User Input ---
input_scaled = scaler.transform(user_input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# --- Section: Data Points ---
st.subheader("📌 Data Points Summary")
st.write(full_df.describe())

# --- Section: Correlation with Target ---
st.subheader("📈 Correlation with Target Variable (Diagnosis)")
plt.figure(figsize=(12, 10))
full_df.drop('diagnosis', axis=1).corrwith(full_df['diagnosis']).plot(
    kind='bar', grid=True, color="green", title="Correlation with Target"
)
st.pyplot(plt.gcf())

# --- Section: Full Correlation Heatmap ---
st.subheader("🧠 Full Feature Correlation Heatmap")
plt.figure(figsize=(20, 20))
sns.heatmap(full_df.corr(), cmap='YlGnBu', annot=True)
plt.title("Correlation Map", fontweight="bold", fontsize=16)
st.pyplot(plt.gcf())

# --- Section: Selected Feature Correlation ---
st.subheader("📊 Heatmap of Selected Features")
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
corr = full_df[['diagnosis'] + selected_features].corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax_corr)
st.pyplot(fig_corr)

# --- Section: Pairplot of Selected Features ---
st.subheader("🔍 Visualizing Input Features (Pairplot)")
fig_pair = sns.pairplot(full_df[['diagnosis'] + selected_features], hue="diagnosis", vars=selected_features, palette="Set2")
st.pyplot(fig_pair)

# --- Section: Prediction ---
st.subheader("🔮 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ The model predicts **Malignant (Cancer)**.")
else:
    st.success("✅ The model predicts **Benign (No Cancer)**.")

st.subheader("📊 Prediction Probability")
st.write(f"Malignant: **{prediction_proba[0][1]*100:.2f}%**")
st.write(f"Benign: **{prediction_proba[0][0]*100:.2f}%**")

# --- Section: Model Accuracy ---
st.subheader("✅ Model Accuracy")
st.write(f"Logistic Regression Accuracy: **{accuracy*100:.2f}%**")

# --- Section: Model Accuracy Plot ---
st.subheader("📉 Model Accuracy Plot")
predicted = ['Logistic Regression']
accuracy_values = [accuracy * 100]

plt.figure(figsize=(10, 5))
sns.barplot(x=predicted, y=accuracy_values, palette='pastel')
plt.title("Plotting the Model Accuracies", fontsize=16, fontweight="bold")
plt.ylabel("Accuracy (%)")
st.pyplot(plt.gcf())
