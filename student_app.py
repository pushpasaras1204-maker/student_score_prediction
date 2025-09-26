import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("students.csv")   
print("\n---- FIRST 5 ROWS ----")
print(data.head())
print("\n---- DATA INFO ----")
print(data.info())
print("\n---- SUMMARY STATISTICS ----")
print(data.describe())
print("\n---- MISSING VALUES ----")
print(data.isnull().sum())
print("\n---- CORRELATION MATRIX ----")
print(data.corr(numeric_only=True))
plt.figure(figsize=(6,4))
sns.scatterplot(x="Study_Hours", y="Marks", data=data, color="blue", s=70)
plt.title("Study Hours vs Marks")
plt.grid(True)
plt.show()
plt.figure(figsize=(6,4))
plt.hist(data["Marks"], bins=5, color="green", edgecolor="black")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(6,4))
sns.boxplot(x=data["Marks"], color="orange")
plt.title("Boxplot of Marks")
plt.show()
plt.figure(figsize=(5,4))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
sns.pairplot(data)
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()
X = data.drop("Marks", axis=1)
y = data["Marks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\n---- MODEL EVALUATION ----")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [85],
    "Assignments_Completed": [7],
    "Previous_Score": [65]
})

predicted_score = rf.predict(new_student)
print("\nPredicted Marks for new student:", round(predicted_score[0], 2))