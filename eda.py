import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("edupro_prepared_dataset.csv")

#1
print("\nShape:", data.shape)
print("\nColumns:\n", data.columns)

print("\nInfo")
print(data.info())
#2
print("\nDescribe")
print(data.describe())
#3
print("\nMissing values:")
print(data.isna().sum())

plt.figure()
sns.histplot(data["EnrollmentCount"], kde=True)
plt.title("Enrollment Distribution")
plt.show()

plt.figure()
sns.histplot(data["CourseRevenue"], kde=True)
plt.title("Revenue Distribution")
plt.show()
#4
plt.figure()
data.groupby("CourseCategory")["EnrollmentCount"].mean().sort_values().plot(kind="bar")
plt.title("Average Enrollment by Category")
plt.show()

plt.figure()
data.groupby("CourseCategory")["CourseRevenue"].mean().sort_values().plot(kind="bar")
plt.title("Average Revenue by Category")
plt.show()
#5
plt.figure()
sns.scatterplot(x=data["CoursePrice"], y=data["EnrollmentCount"])
plt.title("Price vs Enrollment")
plt.show()

plt.figure()
sns.scatterplot(x=data["CoursePrice"], y=data["CourseRevenue"])
plt.title("Price vs Revenue")
plt.show()
#6
plt.figure()
sns.boxplot(x="RatingTier", y="EnrollmentCount", data=data)
plt.title("Rating Impact on Enrollment")
plt.show()

# 7
if "CourseLevel" in data.columns:
    plt.figure()
    sns.boxplot(x="CourseLevel", y="EnrollmentCount", data=data)
    plt.title("Course Level vs Enrollment")
    plt.show()

# 8
num_data = data.select_dtypes(include="number")

plt.figure(figsize=(10,6))
sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#9
print("\nTop 10 courses by revenue:")
print(data.sort_values("CourseRevenue", ascending=False)[["CourseName","CourseRevenue"]].head(10))

print("\nTop 10 courses by enrollment:")
print(data.sort_values("EnrollmentCount", ascending=False)[["CourseName","EnrollmentCount"]].head(10))