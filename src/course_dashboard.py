import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "edupro_prepared_dataset.csv")
enroll_model_path = os.path.join(BASE_DIR, "model", "enrollment_model.pkl")
rev_model_path = os.path.join(BASE_DIR, "model", "revenue_model.pkl")

data = pd.read_csv(data_path)
enroll_model = joblib.load(enroll_model_path)
rev_model = joblib.load(rev_model_path)

st.title("EduPro — Course Demand & Revenue Forecast")

#sidebar
st.sidebar.header("Course Inputs")

course_price = st.sidebar.number_input("Course Price", 0.0, 1000.0, 100.0)
duration = st.sidebar.number_input("Duration (hours)", 1.0, 100.0, 10.0)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.2)

category = None
level = None

if "CourseCategory" in data.columns:
    category = st.sidebar.selectbox(
        "Category",
        sorted(data["CourseCategory"].dropna().unique())
    )

if "CourseLevel" in data.columns:
    level = st.sidebar.selectbox(
        "Course Level",
        sorted(data["CourseLevel"].dropna().unique())
    )


def build_input_row():

    
    row = data.iloc[[0]].copy()

    
    if "CoursePrice" in row.columns:
        row["CoursePrice"] = course_price

    if "CourseDuration" in row.columns:
        row["CourseDuration"] = duration

    if "CourseRating" in row.columns:
        row["CourseRating"] = rating

    if category is not None and "CourseCategory" in row.columns:
        row["CourseCategory"] = category

    if level is not None and "CourseLevel" in row.columns:
        row["CourseLevel"] = level

    
    zero_cols = [
        "EnrollmentCount",
        "CourseRevenue",
        "AvgRevenuePerEnrollment",
        "MonthlyEnrollments",
        "CategoryRevenue"
    ]

    for c in zero_cols:
        if c in row.columns:
            row[c] = 0

    return row

#prediction
if st.button("Predict"):

    input_df = build_input_row()

    enroll_pred = enroll_model.predict(input_df)[0]
    revenue_pred = rev_model.predict(input_df)[0]

    st.subheader("Predictions")

    st.metric("Predicted Enrollments", int(enroll_pred))
    st.metric("Predicted Revenue", f"${revenue_pred:,.2f}")


st.subheader("Category Revenue Comparison")

if "CourseCategory" in data.columns:
    cat_rev = data.groupby("CourseCategory")["CourseRevenue"].mean().sort_values()
    st.bar_chart(cat_rev)


st.subheader("Top Courses")

cols = [c for c in ["CourseName", "CourseRevenue"] if c in data.columns]

if cols:
    top = data.sort_values("CourseRevenue", ascending=False)[cols].head(10)
    st.dataframe(top)
