import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Score Prediction", layout="wide")
st.title("📊 Student Score Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

data = load_data()
st.write("### First 5 Rows", data.head())

# EDA
st.subheader("EDA")
fig, ax = plt.subplots()
sns.scatterplot(x="Study_Hours", y="Marks", data=data, ax=ax)
st.pyplot(fig)

# Model
X = data.drop("Marks", axis=1)
y = data["Marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.write("**Model Performance:**")
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

# Prediction
st.subheader("🎯 Predict Marks for New Student")
study_hours = st.number_input("Study Hours", min_value=1, max_value=120, value=6)
attendance = st.number_input("Attendance", min_value=0, max_value=100, value=85)
assignments = st.number_input("Assignments Completed", min_value=0, max_value=20, value=7)
previous_score = st.number_input("Previous Score", min_value=0, max_value=100, value=65)

if st.button("Predict"):
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Assignments_Completed": [assignments],
        "Previous_Score": [previous_score]
    })
    prediction = rf.predict(new_student)[0]
    st.success(f"Predicted Marks: {round(prediction, 2)}")


