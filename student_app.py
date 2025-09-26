import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Student Score Prediction", layout="wide")
st.title("📊 EDA and Student Score Prediction using Random Forest")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

data = load_data()
st.write("### First 5 Rows of Dataset")
st.dataframe(data.head())

# -------------------------------
# Exploratory Data Analysis
# -------------------------------
st.header("🔎 Exploratory Data Analysis (EDA)")

plot_type = st.selectbox(
    "Choose a Plot",
    ["Scatter Plot", "Histogram", "Boxplot", "Correlation Heatmap", "Pairplot"]
)

if plot_type == "Scatter Plot":
    x_axis = st.selectbox("X-axis", data.columns, index=0)
    y_axis = st.selectbox("Y-axis", data.columns, index=len(data.columns) - 1)
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, data=data, ax=ax, s=70, color="blue")
    st.pyplot(fig)

elif plot_type == "Histogram":
    column = st.selectbox("Select Column", data.columns, index=len(data.columns) - 1)
    fig, ax = plt.subplots()
    ax.hist(data[column], bins=6, color="green", edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)

elif plot_type == "Boxplot":
    column = st.selectbox("Select Column", data.columns, index=len(data.columns) - 1)
    fig, ax = plt.subplots()
    sns.boxplot(x=data[column], ax=ax, color="orange")
    ax.set_title(f"Boxplot of {column}")
    st.pyplot(fig)

elif plot_type == "Correlation Heatmap":
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif plot_type == "Pairplot":
    st.info("⚠ Pairplot may take a little time for larger datasets.")
    fig = sns.pairplot(data)
    st.pyplot(fig)

# -------------------------------
# Model Training
# -------------------------------
st.header("🤖 Model Training & Evaluation")

X = data.drop("Marks", axis=1)
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.write("**Model Performance:**")
st.write(f"📉 Mean Squared Error: `{mean_squared_error(y_test, y_pred):.2f}`")
st.write(f"📈 R² Score: `{r2_score(y_test, y_pred):.2f}`")

# -------------------------------
# Prediction Section
# -------------------------------
st.header("🎯 Predict Marks for New Student")

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
    prediction = rf.predict(new_student)[0]
    st.success(f"✅ Predicted Marks: {round(prediction, 2)}")




