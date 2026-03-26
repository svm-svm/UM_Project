import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("processed_edupro_data.csv")

target_enroll = "EnrollmentCount"
target_revenue = "CourseRevenue"

#remove useless column
drop_cols = [
    "CourseID",
    "CourseName",
    "EnrollmentCount",
    "CourseRevenue",
    "CategoryRevenue",        
    "MonthlyEnrollments",     
    "AvgRevenuePerEnrollment" 
]

X = data.drop(columns=drop_cols, errors="ignore")
y_enroll = data[target_enroll]
y_revenue = data[target_revenue]


X_train, X_test, yE_train, yE_test, yR_train, yR_test = train_test_split(
    X, y_enroll, y_revenue, test_size=0.2, random_state=42
)

#preprocess
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
}

#evaluate
def evaluate(model, X_train, X_test, y_train, y_test):

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return pipe, mae, rmse, r2


#enrollment
print("\n=== ENROLLMENT PREDICTION ===")

best_enroll_model = None
best_enroll_name = None
best_rmse = float("inf")

for name, model in models.items():

    pipe, mae, rmse, r2 = evaluate(model, X_train, X_test, yE_train, yE_test)

    print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")

    if rmse < best_rmse and r2 < 0.999:
        best_rmse = rmse
        best_enroll_model = pipe
        best_enroll_name = name

print(f"\n✅ Best Enrollment Model: {best_enroll_name}")


#revenue
print("\n=== REVENUE PREDICTION ===")

best_rev_model = None
best_rev_name = None
best_rmse = float("inf")

for name, model in models.items():

    pipe, mae, rmse, r2 = evaluate(model, X_train, X_test, yR_train, yR_test)

    print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_rev_model = pipe
        best_rev_name = name

print(f"\n✅ Best Revenue Model: {best_rev_name}")

joblib.dump({
    "model": best_enroll_model,
    "model_name": best_enroll_name,
    "features": X.columns.tolist()
}, "enrollment_model.pkl")

joblib.dump({
    "model": best_rev_model,
    "model_name": best_rev_name,
    "features": X.columns.tolist()
}, "revenue_model.pkl")

