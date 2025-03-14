# 📃 Data Transformation

## 🛠️ Objectives
- Apply data transformation techniques to improve feature distributions
- Address outliers to improve model performance
- Ensure data consistency across train and test sets

## 🔢 Steps
1. **Log Transformation:**
   - Applied log transformation to `CRIM`, `DIS`, and `AGE` to reduce skewness.
2. **Square Root Transformation:**
   - Applied square root transformation for features that required milder adjustments.
3. **Box-Cox Transformation:**
   - Applied Box-Cox transformation for features requiring stronger correction.
4. **Data Splitting:**
   - Split data into `X_train`, `X_test`, `y_train`, and `y_test` to ensure effective model evaluation.
5. **Data Export:**
   - Saved transformed data as `.pkl` files to ensure reproducibility and easy loading.

## 📊 Key Results
- **Skewness Reduced:** Transformation improved data distribution significantly.
- **Outlier Impact Mitigated:** Outlier effects reduced, especially for `CRIM`, `DIS`, and `AGE`.

## 🔍 Next Steps
- Perform feature engineering to introduce additional predictive features for modeling.

