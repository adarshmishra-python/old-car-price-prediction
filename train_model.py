
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "dataset" / "car_data.csv")

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

categorical = ["brand", "fuel", "transmission", "owner"]
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=250, random_state=42, max_depth=14, min_samples_leaf=2
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print(f"MAE: ₹{mean_absolute_error(y_test, pred):,.0f}")
print(f"R2 Score: {r2_score(y_test, pred):.3f}")

joblib.dump(pipeline, BASE / "model" / "car_price_model.pkl")
print("Model saved successfully.")
