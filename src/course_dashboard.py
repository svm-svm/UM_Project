import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#models
enroll_data = joblib.load("model/enrollment_model.pkl")
rev_data = joblib.load("model/revenue_model.pkl")

enroll_model = enroll_data["model"]
rev_model = rev_data["model"]
features = rev_data["features"]

df = pd.read_csv("processed_edupro_data.csv")

st.title("EduPro Course Demand & Revenue Predictor")

#sidebarr
st.sidebar.header("Course Inputs")

course_type = st.sidebar.selectbox("Course Type", df["CourseType"].unique())

if course_type == "Free":
    price = 0
    st.sidebar.warning("Free course → Price fixed at ₹0")
else:
    price = st.sidebar.slider("Course Price", 0, 500, 100)

duration = st.sidebar.slider("Course Duration (hours)", 1, 50, 20)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.0)

category = st.sidebar.selectbox("Course Category", df["CourseCategory"].unique())
level = st.sidebar.selectbox("Course Level", df["CourseLevel"].unique())

teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.0)
experience = st.sidebar.slider("Years of Experience", 0, 25, 5)

#input
input_data = pd.DataFrame({
    "CoursePrice": [float(price)],
    "CourseDuration": [float(duration)],
    "CourseRating": [float(rating)],
    "CourseCategory": [str(category)],
    "CourseType": [str(course_type)],
    "CourseLevel": [str(level)],
    "TeacherRating": [float(teacher_rating)],
    "YearsOfExperience": [float(experience)]
})

for col in features:
    if col not in input_data.columns:
        if col in df.select_dtypes(include="object").columns:
            input_data[col] = "Unknown"
        else:
            input_data[col] = 0.0

input_data = input_data[features]

for col in input_data.columns:
    if col in df.select_dtypes(include="object").columns:
        input_data[col] = input_data[col].astype(str)
    else:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

#predict
if st.button("Predict"):

    enroll_pred = enroll_model.predict(input_data)[0]
    if price == 0:
        revenue_pred = 0
    else:
        revenue_pred = rev_model.predict(input_data)[0]

    st.subheader("Predictions")

    st.metric("Predicted Enrollment", f"{int(enroll_pred)} students")
    st.metric("Predicted Revenue", f"₹ {int(revenue_pred)}")

    st.info("Note: Enrollment prediction is less accurate due to missing behavioral data.")

    st.subheader("Revenue vs Price Analysis")

    price_range = np.linspace(10, 500, 30)
    revenues = []

    for p in price_range:
        temp = input_data.copy()
        temp["CoursePrice"] = p

        if p == 0:
            pred = 0
        else:
            pred = rev_model.predict(temp)[0]

        revenues.append(pred)

    fig, ax = plt.subplots()
    ax.plot(price_range, revenues)
    ax.set_xlabel("Price")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue vs Price")

    st.pyplot(fig)

    #best price
    best_price = price_range[np.argmax(revenues)]
    best_revenue = max(revenues)

    st.subheader("Optimal Pricing Insight")
    st.success(f"Optimal Price: ₹ {int(best_price)}")
    st.success(f"Max Revenue: ₹ {int(best_revenue)}")

    # features
    try:
        model = rev_model.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Feature Importance")

            importances = model.feature_importances_
            feat_names = rev_model.named_steps["prep"].get_feature_names_out()

            imp_df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig2, ax2 = plt.subplots()
            ax2.barh(imp_df["Feature"], imp_df["Importance"])
            ax2.invert_yaxis()
            ax2.set_title("Top Features")

            st.pyplot(fig2)

    except:
        pass

# compare
st.subheader("Category-Level Comparison")

metric = st.selectbox("Compare by:", ["Revenue", "Enrollment"])

cat_data = df.groupby("CourseCategory").agg({
    "CourseRevenue": "mean",
    "EnrollmentCount": "mean"
}).reset_index()

if metric == "Revenue":
    values = cat_data["CourseRevenue"]
else:
    values = cat_data["EnrollmentCount"]

fig3, ax3 = plt.subplots()
ax3.bar(cat_data["CourseCategory"], values)
ax3.set_title(f"Category Comparison: {metric}")
ax3.set_xticklabels(cat_data["CourseCategory"], rotation=45)

st.pyplot(fig3)
