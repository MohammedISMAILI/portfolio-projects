from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Split dataset into features and target variable
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = lin_reg.predict(X_test_scaled)

# Calculate and print the performance metrics
print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
print("R^2 Score: ", r2_score(y_test, y_pred))


from sklearn.decomposition import PCA

# Initialize PCA, let's say we want to keep 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit the linear regression model on the reduced dataset
lin_reg_pca = LinearRegression()
lin_reg_pca.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred_pca = lin_reg_pca.predict(X_test_pca)
print("RMSE (PCA): ", mean_squared_error(y_test, y_pred_pca, squared=False))
print("R^2 Score (PCA): ", r2_score(y_test, y_pred_pca))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name="PRICE")

# Data Preprocessing
# Handling missing values (if any)
X.fillna(X.mean(), inplace=True)

# Normalizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection
# Correlation matrix
corr_matrix = pd.DataFrame(X_scaled, columns=boston.feature_names).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Eigenvalues and Eigenvectors Analysis
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Principal Component Analysis (PCA)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance by Principal Components:\n", pca.explained_variance_ratio_)

# Model Development: Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction and Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)

# Visualizations
# Scatter plot of actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices')
plt.show()

# PCA Components
plt.figure(figsize=(8, 6))
plt.bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 6), np.cumsum(pca.explained_variance_ratio_), where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.show()

# Feature Importance (based on coefficients in PCA-transformed space)
plt.figure(figsize=(8, 6))
plt.bar(range(X_train.shape[1]), model.coef_)
plt.xlabel('PCA Components')
plt.ylabel('Coefficients')
plt.title('Feature Importance in PCA-transformed Space')
plt.show()
