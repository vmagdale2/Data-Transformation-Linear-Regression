import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

# Load Feature Engineered Data
X_train = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/X_train_feature_engineered.pkl")
X_test = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/X_test_feature_engineered.pkl")
y_train = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/y_train.pkl")
y_test = joblib.load("/content/drive/MyDrive/Professional/Portfolio/Transformations/y_test.pkl")

# Load Model
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='r2')
print(f"🔍 Cross-Validation R-squared scores: {cv_scores}")
print(f"✅ Average CV R-squared: {cv_scores.mean():.4f}")

# Fold Investigation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    ridge_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    val_pred = ridge_model.predict(X_train.iloc[val_idx])
    val_r2 = r2_score(y_train.iloc[val_idx], val_pred)
    print(f"🔍 Fold {fold} R-squared: {val_r2:.4f}")

print("✅ Evaluation complete!")
