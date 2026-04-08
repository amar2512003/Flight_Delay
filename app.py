import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================================
# Load model and encoders
# ==========================================================
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ==========================================================
# Load dataset
# ==========================================================
df = pd.read_csv("2018.csv")

# ==========================================================
# Original notebook preprocessing
# ==========================================================

delay_cols = [
    'CARRIER_DELAY',
    'WEATHER_DELAY',
    'NAS_DELAY',
    'SECURITY_DELAY',
    'LATE_AIRCRAFT_DELAY'
]

for col in delay_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

if 'CANCELLATION_CODE' in df.columns:
    df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].fillna('Not Cancelled')

if 'Unnamed: 27' in df.columns:
    df = df.drop(columns=['Unnamed: 27'])

# Remove cancelled and diverted flights
df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

# Create target variable
df['delay'] = (df['ARR_DELAY'] > 15).astype(int)

# ==========================================================
# Original notebook features
# ==========================================================
features = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DISTANCE",
    "CRS_ELAPSED_TIME"
]

X = df[features].copy()
y = df["delay"]

# ==========================================================
# Keep only original encoder classes (important fix)
# ==========================================================
for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
    X = X[X[col].isin(encoders[col].classes_)]

# Align target
y = y.loc[X.index]

# ==========================================================
# Safe encoding
# ==========================================================
for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
    X[col] = encoders[col].transform(X[col])

# ==========================================================
# Real test split for original metrics
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# Predictions
# ==========================================================
y_pred = model.predict(X_test)

# ==========================================================
# Real metrics
# ==========================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

st.title("✈️ Flight Delay Prediction System")

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "EDA",
    "Preprocessing Pipeline",
    "Model Metrics"
])

# ==========================================================
# TAB 1 Prediction
# ==========================================================
with tab1:

    st.header("Flight Delay Prediction")

    carrier = st.selectbox("Airline", encoders["OP_CARRIER"].classes_)
    origin = st.selectbox("Origin Airport", encoders["ORIGIN"].classes_)
    dest = st.selectbox("Destination Airport", encoders["DEST"].classes_)

    crs_dep_time = st.number_input("Scheduled Departure Time (HHMM)", 0, 2359)
    distance = st.number_input("Distance", 0)
    crs_elapsed = st.number_input("Scheduled Duration (minutes)", 0)

    if st.button("Predict"):

        carrier_encoded = encoders["OP_CARRIER"].transform([carrier])[0]
        origin_encoded = encoders["ORIGIN"].transform([origin])[0]
        dest_encoded = encoders["DEST"].transform([dest])[0]

        input_data = np.array([[
            carrier_encoded,
            origin_encoded,
            dest_encoded,
            crs_dep_time,
            distance,
            crs_elapsed
        ]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ Flight will be DELAYED\n\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ Flight will be ON TIME\n\nDelay Probability: {probability:.2%}")

# ==========================================================
# TAB 2 EDA
# ==========================================================
with tab2:

    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Arrival Delay Distribution")

    fig1, ax1 = plt.subplots()
    df['ARR_DELAY'].dropna().hist(bins=50, ax=ax1)
    ax1.set_xlabel("Arrival Delay")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    st.subheader("Top Airlines")

    top_airlines = df['OP_CARRIER'].value_counts().head(10)

    fig2, ax2 = plt.subplots()
    top_airlines.plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

# ==========================================================
# TAB 3 Preprocessing Pipeline
# ==========================================================
with tab3:

    st.header("Preprocessing Pipeline")

    st.markdown("""
    ### Original Notebook Pipeline

    1. Filled missing delay values  
    2. Removed cancelled flights  
    3. Removed diverted flights  
    4. Created target variable:
       Delay = ARR_DELAY > 15 min  
    5. Selected core operational features  
    6. Applied label encoding  
    7. Trained Random Forest model  
    """)

    st.subheader("Encoder Classes Sample")

    sample = pd.DataFrame({
        "Carrier": encoders["OP_CARRIER"].classes_[:10],
        "Origin": encoders["ORIGIN"].classes_[:10],
        "Destination": encoders["DEST"].classes_[:10]
    })

    st.dataframe(sample)

# ==========================================================
# TAB 4 Metrics
# ==========================================================
with tab4:

    st.header("Real Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig3, ax3 = plt.subplots()
    ax3.imshow(cm)

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, cm[i, j], ha='center', va='center')

    st.pyplot(fig3)

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    fig4, ax4 = plt.subplots()
    ax4.bar(features, importance)
    plt.xticks(rotation=45)

    st.pyplot(fig4)