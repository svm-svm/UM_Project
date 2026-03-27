import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv("processed_edupro_data.csv")

loaded = joblib.load("model/revenue_model.pkl")
model = loaded["model"]

preprocessor = model.named_steps["prep"]
estimator = model.named_steps["model"]
#feature names
X = data.drop(columns=[
    "CourseID",
    "CourseName",
    "EnrollmentCount",
    "CourseRevenue",
    "CategoryRevenue",
    "MonthlyEnrollments",
    "AvgRevenuePerEnrollment"
], errors="ignore")

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

ohe = preprocessor.named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_cols)

feature_names = np.concatenate([num_cols, cat_names])

if hasattr(estimator, "feature_importances_"):
    importances = estimator.feature_importances_
elif hasattr(estimator, "coef_"):
    importances = np.abs(estimator.coef_)
else:
    raise ValueError("Model does not support importance extraction")

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(imp_df.head(20))

plt.figure(figsize=(10,6))
plt.barh(imp_df.head(15)["Feature"][::-1], imp_df.head(15)["Importance"][::-1])
plt.title("Feature Importance")
plt.show()
