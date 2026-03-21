import pandas as pd
import numpy as np

file_path = "EduPro Online Platform.xlsx"

courses = pd.read_excel(file_path, sheet_name="Courses")
transactions = pd.read_excel(file_path, sheet_name="Transactions")

try:
    teachers = pd.read_excel(file_path, sheet_name="Teachers")
except:
    teachers = None

print("Courses columns:", courses.columns)
print("Transactions columns:", transactions.columns)
if teachers is not None:
    print("Teachers columns:", teachers.columns)

#clean
courses = courses.drop_duplicates(subset="CourseID")

num_cols = ["CoursePrice", "CourseDuration", "CourseRating"]
for c in num_cols:
    if c in courses.columns:
        courses[c] = pd.to_numeric(courses[c], errors="coerce")

courses["CoursePrice"] = courses["CoursePrice"].fillna(0)
courses["CourseDuration"] = courses["CourseDuration"].fillna(courses["CourseDuration"].median())
courses["CourseRating"] = courses["CourseRating"].fillna(courses["CourseRating"].median())


transactions = transactions.drop_duplicates()

transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
transactions["Amount"] = pd.to_numeric(transactions["Amount"], errors="coerce")

transactions = transactions[transactions["Amount"] >= 0]


course_perf = (
    transactions
    .groupby("CourseID")
    .agg(
        EnrollmentCount=("TransactionID", "count"),
        CourseRevenue=("Amount", "sum"),
        AvgRevenuePerEnrollment=("Amount", "mean")
    )
    .reset_index()
)

#merge
data = courses.merge(course_perf, on="CourseID", how="left")


data["EnrollmentCount"] = data["EnrollmentCount"].fillna(0)
data["CourseRevenue"] = data["CourseRevenue"].fillna(0)
data["AvgRevenuePerEnrollment"] = data["AvgRevenuePerEnrollment"].fillna(0)


if teachers is not None:

    if "TeacherID" in courses.columns and "TeacherID" in teachers.columns:

        teachers = teachers.drop_duplicates(subset="TeacherID")

        if "YearsOfExperience" in teachers.columns:
            teachers["YearsOfExperience"] = pd.to_numeric(
                teachers["YearsOfExperience"], errors="coerce"
            ).fillna(0)

        if "TeacherRating" in teachers.columns:
            teachers["TeacherRating"] = pd.to_numeric(
                teachers["TeacherRating"], errors="coerce"
            ).fillna(teachers["TeacherRating"].median())

        data = data.merge(teachers, on="TeacherID", how="left")

#feature
# 1
data["PriceBand"] = pd.cut(
    data["CoursePrice"],
    bins=[0, 50, 200, 1000],
    labels=["Low", "Medium", "High"]
)

# 2
data["DurationBucket"] = pd.cut(
    data["CourseDuration"],
    bins=[0, 5, 20, 100],
    labels=["Short", "Medium", "Long"]
)

# 3
data["RatingTier"] = pd.cut(
    data["CourseRating"],
    bins=[0, 3.5, 4.3, 5],
    labels=["Low", "Good", "Top"]
)


if "CourseLevel" in data.columns:
    level_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    data["CourseLevelEncoded"] = data["CourseLevel"].map(level_map)

#4
category_revenue = (
    data.groupby("CourseCategory")["CourseRevenue"]
    .sum()
    .reset_index()
    .rename(columns={"CourseRevenue": "CategoryRevenue"})
)

data = data.merge(category_revenue, on="CourseCategory", how="left")


transactions["Month"] = transactions["TransactionDate"].dt.month

monthly = (
    transactions.groupby("CourseID")
    .size()
    .reset_index(name="MonthlyEnrollments")
)

data = data.merge(monthly, on="CourseID", how="left")
data["MonthlyEnrollments"] = data["MonthlyEnrollments"].fillna(0)

#final
print("\nFinal dataset shape:", data.shape)
print(data.head())

data.to_csv("edupro_prepared_dataset.csv", index=False)
print("\nSaved → edupro_prepared_dataset.csv")