import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

file_path = "data/EduPro Online Platform.xlsx"

users = pd.read_excel(file_path, sheet_name='Users')
courses = pd.read_excel(file_path, sheet_name='Courses')
teachers = pd.read_excel(file_path, sheet_name='Teachers')
transactions = pd.read_excel(file_path, sheet_name='Transactions')

#merge
df = transactions.merge(courses, on='CourseID', how='left')
df = df.merge(teachers, on='TeacherID', how='left')

print("Merged Dataset Shape:", df.shape)
print(df.head())


#feature eng
#1.price
df['PriceBand'] = pd.cut(
    df['CoursePrice'],
    bins=[-1, 50, 150, 500],
    labels=['Low', 'Medium', 'High']
)

#2. duration
df['DurationBucket'] = pd.cut(
    df['CourseDuration'],
    bins=[0, 15, 35, 60],
    labels=['Short', 'Medium', 'Long']
)

# 3. rating
df['RatingTier'] = pd.cut(
    df['CourseRating'],
    bins=[0, 3, 4, 4.5, 5],
    labels=['Poor', 'Average', 'Good', 'Excellent']
)

# 4. level
level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
df['CourseLevelEncoded'] = df['CourseLevel'].map(level_map)

# 5.exp
df['ExperienceBucket'] = pd.cut(
    df['YearsOfExperience'],
    bins=[0, 2, 5, 10, 20],
    labels=['Junior', 'Mid', 'Senior', 'Expert']
)

# 6.rev 
df['RevenuePerEnrollment'] = df['Amount']  
df['ValueScore'] = df['CourseRating'] / (df['CoursePrice'] + 1)

#add

agg_df = df.groupby('CourseID').agg({
    'Amount': ['sum', 'mean', 'count'],
    'CoursePrice': 'first',
    'CourseDuration': 'first',
    'CourseRating': 'first',
    'CourseCategory': 'first',
    'CourseType': 'first',
    'CourseLevel': 'first',
    'CourseLevelEncoded': 'first',
    'PriceBand': 'first',
    'DurationBucket': 'first',
    'RatingTier': 'first',
    'TeacherRating': 'mean',
    'YearsOfExperience': 'mean'
}).reset_index()


agg_df.columns = [
    'CourseID',
    'CourseRevenue',
    'AvgRevenuePerEnrollment',
    'EnrollmentCount',
    'CoursePrice',
    'CourseDuration',
    'CourseRating',
    'CourseCategory',
    'CourseType',
    'CourseLevel',
    'CourseLevelEncoded',
    'PriceBand',
    'DurationBucket',
    'RatingTier',
    'AvgTeacherRating',
    'AvgExperience'
]

#rev
category_revenue = agg_df.groupby('CourseCategory')['CourseRevenue'].transform('sum')
agg_df['CategoryRevenue'] = category_revenue

# enrollments
monthly_enrollments = df.groupby(['CourseID', pd.Grouper(key='TransactionDate', freq='M')]).size().reset_index(name='MonthlyEnrollments')

monthly_avg = monthly_enrollments.groupby('CourseID')['MonthlyEnrollments'].mean().reset_index()

agg_df = agg_df.merge(monthly_avg, on='CourseID', how='left')

#missing values
agg_df['PriceBand'] = agg_df['PriceBand'].fillna('Medium')
agg_df['DurationBucket'] = agg_df['DurationBucket'].fillna('Medium')
agg_df['RatingTier'] = agg_df['RatingTier'].fillna('Average')
agg_df['AvgTeacherRating'] = agg_df['AvgTeacherRating'].fillna(agg_df['AvgTeacherRating'].mean())
agg_df['AvgExperience'] = agg_df['AvgExperience'].fillna(agg_df['AvgExperience'].mean())


print("\nFinal Aggregated Dataset Shape:", agg_df.shape)
print(agg_df.head())

# Save 
agg_df.to_csv("processed_edupro_data.csv", index=False)

print("\n✅ Phase 1 & 2 Completed Successfully!")
