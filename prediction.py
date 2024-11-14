import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
data_path = '/Users/zactseng/Documents/ECE 1513/Project/GTA-Real-Estate-Price-Prediction/clean_combined_toronto_property_data.xlsx'
data = pd.read_excel(data_path)

# Data preprocessing
# Select relevant features
features = ['location', 'bedrooms', 'bathrooms', 'property_size']
target = 'price'
data = data[features + [target]].dropna()

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Separate features and target
X = data.drop(columns=target)
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model initialization
models = {
    "Polynomial Regression": PolynomialFeatures(degree=2),
    "SVM": SVR(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lin_reg = RandomForestRegressor(random_state=42)
lin_reg.fit(X_train_poly, y_train)
y_pred_poly = lin_reg.predict(X_test_poly)

# SVM
svm_params = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100]}
svr = GridSearchCV(SVR(), svm_params, cv=5, scoring='neg_mean_squared_error')
svr.fit(X_train, y_train)
y_pred_svm = svr.best_estimator_.predict(X_test)

# Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error')
rf.fit(X_train, y_train)
y_pred_rf = rf.best_estimator_.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name}: MSE = {mse:.2f}, RMSE = {rmse:.2f}")

evaluate_model(y_test, y_pred_poly, "Polynomial Regression")
evaluate_model(y_test, y_pred_svm, "SVM")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, label="Random Forest Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.legend()
plt.show()
