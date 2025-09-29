# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------
# 1️⃣ Title
# --------------------------
st.title("💓 Heart Disease Risk Predictor")
st.markdown("Predict heart disease risk for single patients or multiple patients via CSV upload.")

# --------------------------
# 2️⃣ Dataset (Training)
# --------------------------
data = {
    "age": [52, 58, 46, 54, 61, 42, 65, 49, 57, 60],
    "sex": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    "cp": [0, 2, 1, 3, 0, 2, 1, 0, 2, 3],
    "trestbps": [140, 130, 120, 150, 138, 125, 160, 135, 145, 155],
    "chol": [260, 230, 204, 284, 236, 228, 294, 210, 278, 240],
    "fbs": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "restecg": [1, 0, 1, 2, 1, 1, 0, 1, 2, 0],
    "thalach": [172, 165, 180, 150, 160, 175, 148, 168, 155, 152],
    "exang": [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    "oldpeak": [0.0, 2.3, 1.4, 3.6, 1.8, 0.0, 2.5, 0.5, 2.0, 3.2],
    "slope": [2, 1, 2, 0, 1, 2, 0, 2, 1, 0],
    "ca": [0, 2, 0, 1, 0, 0, 2, 0, 1, 3],
    "thal": [2, 3, 2, 3, 2, 2, 3, 2, 3, 3],
    "target": [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# --------------------------
# 3️⃣ Preprocessing
# --------------------------
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop("target", axis=1)
y = df["target"]

# --------------------------
# 4️⃣ Train Model
# --------------------------
model = LogisticRegression()
model.fit(X, y)

# --------------------------
# 5️⃣ Single Patient Input
# --------------------------
st.header("🔹 Predict Single Patient")

def user_input_features():
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    cp_map = {0: "Typical", 1: "Atypical", 2: "Non-anginal", 3: "Asymptomatic"}
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: cp_map[x])
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=130)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
    
    restecg_map = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: restecg_map[x])
    
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression induced by Exercise", value=1.0, step=0.1)
    
    slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2], format_func=lambda x: slope_map[x])
    
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    
    thal_map = {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: thal_map[x])
    
    sex_val = 1 if sex == "Male" else 0

    data = {
        "age": age,
        "sex": sex_val,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame([data])

if st.checkbox("Predict Single Patient"):
    input_df = user_input_features()
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)[:, 1]

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠ High risk of Heart Disease (Probability: {prediction_prob[0]:.2f})")
    else:
        st.success(f"✅ Low risk of Heart Disease (Probability: {prediction_prob[0]:.2f})")

# --------------------------
# 6️⃣ CSV Upload for Multiple Patients
# --------------------------
st.header("🔹 Predict Multiple Patients via CSV Upload")
st.markdown("Upload a CSV file with columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")

uploaded_file = st.file_uploader("Choose CSV", type="csv")
if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    
    required_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
    if all(col in df_upload.columns for col in required_cols):
        # Preprocess numerical columns
        df_upload[num_cols] = scaler.transform(df_upload[num_cols])
        
        # Predict
        predictions = model.predict(df_upload)
        prediction_probs = model.predict_proba(df_upload)[:, 1]
        df_upload["Prediction"] = ["High Risk" if p==1 else "Low Risk" for p in predictions]
        df_upload["Probability"] = prediction_probs
        
        # Display CSV without pyarrow
        st.table(df_upload.to_dict(orient="records"))
    else:
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")

# --------------------------
# 7️⃣ Model Performance
# --------------------------
st.header("📊 Model Performance (Training Data)")
train_pred = model.predict(X)
acc = accuracy_score(y, train_pred)
st.write(f"**Accuracy:** {acc:.2f}")

cm = confusion_matrix(y, train_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
st.pyplot(fig)

