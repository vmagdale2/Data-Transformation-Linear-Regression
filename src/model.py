import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Feature Engineered Data
X_train = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/X_train_feature_engineered.pkl")
X_test = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/X_test_feature_engineered.pkl")
y_train = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/y_train.pkl")
y_test = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/y_test.pkl")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

print(f"🔍 Training MSE (Linear): {mean_squared_error(y_train, y_train_pred):.4f} | R-squared: {r2_score(y_train, y_train_pred):.4f}")
print(f"🔍 Testing MSE (Linear): {mean_squared_error(y_test, y_test_pred):.4f} | R-squared: {r2_score(y_test, y_test_pred):.4f}")

# Ridge Regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
y_train_ridge = ridge_model.predict(X_train)
y_test_ridge = ridge_model.predict(X_test)

print(f"🔍 Training MSE (Ridge): {mean_squared_error(y_train, y_train_ridge):.4f} | R-squared: {r2_score(y_train, y_train_ridge):.4f}")
print(f"🔍 Testing MSE (Ridge): {mean_squared_error(y_test, y_test_ridge):.4f} | R-squared: {r2_score(y_test, y_test_ridge):.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=1)
lasso_model.fit(X_train, y_train)
y_train_lasso = lasso_model.predict(X_train)
y_test_lasso = lasso_model.predict(X_test)

print(f"🔍 Training MSE (Lasso): {mean_squared_error(y_train, y_train_lasso):.4f} | R-squared: {r2_score(y_train, y_train_lasso):.4f}")
print(f"🔍 Testing MSE (Lasso): {mean_squared_error(y_test, y_test_lasso):.4f} | R-squared: {r2_score(y_test, y_test_lasso):.4f}")

print("✅ Modeling complete!")
