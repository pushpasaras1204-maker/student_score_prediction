import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Student Score Prediction", layout="wide")
st.title("📊 EDA + Student Score Prediction using Random Forest")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

data = load_data()

st.subheader("🔍 First 5 Rows of Dataset")
st.write(data.head())

# -----------------------------
# EDA Section
# -----------------------------
st.subheader("📈 Exploratory Data Analysis")

with st.expander("Dataset Info"):
    buffer = []
    data.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

st.write("**Summary Statistics**")
st.write(data.describe())

st.write("**Missing Values**")
st.write(data.isnull().sum())

st.write("**Correlation Matrix**")
st.write(data.corr(numeric_only=True))

# Scatter plot
st.write("### Scatter Plot: Study Hours vs Marks")
fig, ax = plt.subplots()
sns.scatterplot(x="Study_Hours", y="Marks", data=data, color="blue", s=70, ax=ax)
ax.set_title("Study Hours vs Marks")
st.pyplot(fig)

# Histogram
st.write("### Distribution of Marks")
fig, ax = plt.subplots()
ax.hist(data["Marks"], bins=5, color="green", edgecolor="black")
ax.set_xlabel("Marks")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Boxplot
st.write("### Boxplot of Marks")
fig, ax = plt.subplots()
sns.boxplot(x=data["Marks"], color="orange", ax=ax)
st.pyplot(fig)

# Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Pairplot (wrapped to avoid crash)
st.write("### Pairwise Relationships")
try:
    fig = sns.pairplot(data)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Pairplot could not be displayed: {e}")

# -----------------------------
# Random Forest Model
# -----------------------------
st.subheader("🤖 Model Training & Evaluation")

X = data.drop("Marks", axis=1)
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.write("**Model Evaluation Metrics:**")
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🎯 Predict Marks for New Student")

study_hours = st.number_input("Study Hours", min_value=1, max_value=120, value=6)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
assignments = st.number_input("Assignments Completed", min_value=0, max_value=20, value=7)
previous_score = st.number_input("Previous Score", min_value=0, max_value=100, value=65)

if st.button("Predict"):
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Assignments_Completed": [assignments],
        "Previous_Score": [previous_score]
    })
    predicted_score = rf.predict(new_student)
    st.success(f"✅ Predicted Marks: {round(predicted_score[0], 2)}")
