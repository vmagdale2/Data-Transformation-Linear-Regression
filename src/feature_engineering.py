import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

# Load data
X_train = pd.read_pickle("/content/drive/MyDrive/Professional/Portfolio/Transformations/X_train.pkl")
X_test = pd.read_pickle("/content/drive/MyDrive/Professional/Portfolio/Transformations/X_test.pkl")

# Interaction Terms
def create_interaction_terms(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(interaction_features, columns=feature_names)

X_train_inter = create_interaction_terms(X_train)
X_test_inter = create_interaction_terms(X_test)

# Binning Example
X_train['AGE_Bin'] = pd.qcut(X_train['AGE'], q=4, labels=['Child', 'Young Adult', 'Middle Age', 'Senior'])
X_train['DIS_Bin'] = pd.qcut(X_train['DIS'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
X_train['CRIM_Bin'] = pd.qcut(X_train['CRIM'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

X_test['AGE_Bin'] = pd.qcut(X_test['AGE'], q=4, labels=['Child', 'Young Adult', 'Middle Age', 'Senior'])
X_test['DIS_Bin'] = pd.qcut(X_test['DIS'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
X_test['CRIM_Bin'] = pd.qcut(X_test['CRIM'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Imputation for new bins
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Save Feature Engineered Data
joblib.dump(X_train, "/content/drive/MyDrive/Professional/Portfolio/Transformations/X_train_feature_engineered.pkl")
joblib.dump(X_test, "/content/drive/MyDrive/Professional/Portfolio/Transformations/X_test_feature_engineered.pkl")

print("✅ Feature Engineering complete and data saved!")
