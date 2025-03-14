# 📃 Feature Engineering

## 🛠️ Objectives
- Create new features to enhance model performance
- Generate interaction terms for improved predictive power
- Apply binning strategies for critical features

## 🔢 Steps
1. **Interaction Terms:**
   - Applied polynomial interaction terms for key numeric features to enhance model complexity.
2. **Binning:**
   - Binned `AGE` into categories: `Child`, `Young Adult`, `Middle-Aged`, and `Senior`.
   - Binned `DIS` into categories: `Near`, `Moderate`, and `Far`.
   - Binned `CRIM` into categories: `Low`, `Medium`, and `High`.
3. **Feature Scaling:**
   - Standardized numeric features to ensure consistent scale for effective modeling.
4. **Data Export:**
   - Saved engineered features as `.pkl` files for seamless integration into the modeling phase.

## 📊 Key Results
- **Interaction Terms Added:** Expanded dataset improved model complexity.
- **Binning Enhanced Stability:** Enhanced feature stability by grouping continuous data.

## 🔍 Next Steps
- Train baseline and regularized models to evaluate model performance.

