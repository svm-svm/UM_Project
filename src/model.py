import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("edupro_prepared_dataset.csv")

target_enroll = "EnrollmentCount"
target_revenue = "CourseRevenue"


drop_cols = [
    "CourseID",
    "CourseName",
    "EnrollmentCount",
    "CourseRevenue"
]

X = data.drop(columns=drop_cols, errors="ignore")
y_enroll = data[target_enroll]
y_revenue = data[target_revenue]


X_train, X_test, yE_train, yE_test = train_test_split(
    X, y_enroll, test_size=0.2, random_state=42
)

_, _, yR_train, yR_test = train_test_split(
    X, y_revenue, test_size=0.2, random_state=42
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
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor()
}

#train
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



print("\n=== ENROLLMENT PREDICTION ===")

best_enroll_model = None
best_score = -999

for name, model in models.items():

    pipe, mae, rmse, r2 = evaluate(model, X_train, X_test, yE_train, yE_test)

    print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")

    if r2 > best_score:
        best_score = r2
        best_enroll_model = pipe


print("\n=== REVENUE PREDICTION ===")

best_rev_model = None
best_score = -999

for name, model in models.items():

    pipe, mae, rmse, r2 = evaluate(model, X_train, X_test, yR_train, yR_test)

    print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")

    if r2 > best_score:
        best_score = r2
        best_rev_model = pipe

#save
import joblib

joblib.dump(best_enroll_model, "enrollment_model.pkl")
joblib.dump(best_rev_model, "revenue_model.pkl")

print("\nSaved models ✔")