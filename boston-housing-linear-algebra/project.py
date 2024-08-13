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
