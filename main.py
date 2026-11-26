
# Student Performance Analysis Using Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load Dataset

data = pd.read_csv("Student_Performance_Analysis/dataset/students.csv")

print("\nSTUDENT DATASET")
print(data)

# Step 2: Data Preprocessing

print("\nMissing Values:")
print(data.isnull().sum())

data = data.drop_duplicates()

subjects = [
    "Math",
    "Science",
    "English",
    "Computer",
    "Social_Science"
]

data[subjects] = data[subjects].apply(
    pd.to_numeric, errors="coerce"
)

data[subjects] = data[subjects].fillna(
    data[subjects].mean()
)

# Step 3: Calculate Total and Average

data["Total"] = data[subjects].sum(axis=1)

data["Average"] = data[subjects].mean(axis=1)

# Step 4: Performance Categories

def performance(avg):
    if avg >= 85:
        return "Excellent"
    elif avg >= 70:
        return "Good"
    elif avg >= 50:
        return "Average"
    else:
        return "Needs Improvement"

data["Performance"] = data["Average"].apply(performance)

# Step 5: Statistical Analysis

print("\nCLASS STATISTICS")

print("Class Average:",
      round(data["Average"].mean(), 2))

print("Highest Average:",
      data["Average"].max())

print("Lowest Average:",
      data["Average"].min())

print("\nSubject-wise Average:")
print(data[subjects].mean())

# Step 6: Highest and Lowest Performing Students

highest = data.loc[data["Average"].idxmax()]

lowest = data.loc[data["Average"].idxmin()]

print("\nHighest Performing Student:")
print(highest[["Name", "Average"]])

print("\nLowest Performing Student:")
print(lowest[["Name", "Average"]])

# Step 7: Machine Learning - KMeans Clustering

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data[subjects])

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = model.fit_predict(scaled_data)

print("\nStudent Performance Groups:")
print(data[["Name", "Average", "Cluster"]])

# Step 8: Save Analysed Dataset

data.to_csv(
    "Student_Performance_Analysis/outputs/analysed_students.csv",
    index=False
)

# Step 9: Create Charts

sns.set_theme(style="whitegrid")

# Chart 1: Subject-wise Average

subject_average = data[subjects].mean()

plt.figure(figsize=(9, 5))

subject_average.plot(
    kind="bar",
    title="Subject-wise Average Marks"
)

plt.ylabel("Average Marks")
plt.xlabel("Subjects")
plt.tight_layout()

plt.savefig("Student_Performance_Analysis/charts/subject_average.png")
plt.show()
plt.close()

# Chart 2: Marks Distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    data["Average"],
    bins=6,
    kde=True
)

plt.title("Student Marks Distribution")
plt.xlabel("Average Marks")

plt.tight_layout()

plt.savefig("Student_Performance_Analysis/charts/marks_distribution.png")
plt.show()
plt.close()

# Chart 3: Student Performance Comparison

plt.figure(figsize=(10, 5))

sns.barplot(
    data=data,
    x="Name",
    y="Average"
)

plt.title("Student Performance Comparison")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("Student_Performance_Analysis/charts/student_comparison.png")
plt.show()
plt.close()

# Chart 4: Correlation Heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(
    data[subjects].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Subject Marks Correlation")

plt.tight_layout()

plt.savefig("Student_Performance_Analysis/charts/correlation_heatmap.png")
plt.show()
plt.close()

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Results saved in outputs folder.")
print("Charts saved in charts folder.")
