# 🏠 Predicting Real Estate Prices in the Greater Toronto Area Using Machine Learning

This project uses machine learning techniques to predict real estate prices in the Greater Toronto Area (GTA). By analyzing key property features such as location, size, and number of bedrooms/bathrooms, the project compares the performance of **Random Forest**, **Support Vector Machine (SVM)**, and **Polynomial Regression** models.

## 📄 Abstract

The GTA real estate market is complex and influenced by numerous variables. Traditional pricing methods often fall short in capturing non-linear relationships among factors. In this study, we build predictive models using modern ML algorithms and evaluate them using Mean Squared Error (MSE) and R² scores. **Random Forest** achieved the best performance with an MSE of **0.0103** and an R² score of **0.9830**.

---

## 📊 Models Compared

| Model                  | MSE     | R² Score |
|-----------------------|---------|----------|
| Random Forest         | 0.0103  | 0.9830   |
| Support Vector Machine (SVM) | 0.0162  | 0.9732   |
| Polynomial Regression | 0.0168  | 0.9722   |

---

## 🧰 Features Used

- Property price (target)
- Region (categorical, one-hot encoded)
- Number of bedrooms
- Number of bathrooms
- Square footage (or proxy size)
- Engineered features:
  - Price per bedroom
  - Price per bathroom

---

## 🧼 Data Preprocessing

- Removed outliers (±3σ)
- One-hot encoding for categorical regions
- Replaced missing bedroom/bathroom values with 0
- Normalized numeric features

---

## ⚙️ Model Implementation

### 1. Random Forest Regressor
- Best performance model
- Tuned with:
  - `n_estimators = 100`
  - `max_depth = 30`
  - `min_samples_split = 2`
  - `min_samples_leaf = 1`

### 2. Support Vector Regressor (SVR)
- Tuned with:
  - `C = 1000`
  - `gamma = 0.01`
  - `epsilon = 0.1`
  - Kernel: RBF

### 3. Polynomial Regression
- Polynomial degree: 3
- Regularization: Ridge with `alpha = 1`

---

## 🧪 Evaluation Metrics

- **Mean Squared Error (MSE):** Measures average squared difference between actual and predicted values
- **R² Score (Coefficient of Determination):** Measures proportion of variance explained by the model

---

## 📈 Visualizations

- MSE and R² bar charts across models
- Scatter plot of actual vs predicted prices
  - Random Forest closely aligns with ideal predictions
  - SVM and Polynomial Regression show more deviation

---

## 🏁 Conclusion

Random Forest is the most suitable model for real estate price prediction in GTA due to its:
- High accuracy
- Robustness to outliers
- Ability to capture non-linear relationships

SVM and Polynomial Regression are viable alternatives, though slightly less accurate in this setting.

---

## 🔮 Future Work

- Add features: neighborhood amenities, proximity to transit, crime rate, school ratings
- Use deep learning models (e.g., MLPs, XGBoost)
- Time-series modeling for trend prediction
- Geographic visualization of predictions with maps

---

