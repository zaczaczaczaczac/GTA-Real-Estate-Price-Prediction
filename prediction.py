import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data_path = '/Users/zactseng/Documents/ECE 1513/Project/GTA-Real-Estate-Price-Prediction/clean_combined_toronto_property_data.xlsx'
data = pd.read_excel(data_path)

# Data preprocessing
# Drop unnecessary columns
data = data.drop(columns=['address', 'pricem'])

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Define features and target
X = data.drop(columns=['price'])
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model initialization and training
models = {
    "SVM": SVR(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    if name == "SVM":
        params = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100]}
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    
    # Predict
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results[name] = {"MSE": mse, "RMSE": rmse}

# Display results
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.2f}, RMSE = {metrics['RMSE']:.2f}")

# Visualization of the best model
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.legend()
plt.show()
