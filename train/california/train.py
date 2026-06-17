"""
# Components:
#  mlflow.start_run() # Start a new run
#  mlflow.log_param() # Log hyperparameters
#  mlflow.log_metric() # Log metrics like RMSE and R²
#  mlflow.sklearn.log_model() # Save the model
#  mlflow.end_run() # End the run
# Uses scikit-learn's California Housing dataset to train a Ridge Regression model.
# Logs hyperparameters, metrics, and the trained model using MLflow.

# pip install mlflow
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
# gives you a fully functional local instance with the UI at localhost:5000
"""
import mlflow
from mlflow import sklearn
from pandas.core.common import random_state
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# Load the dataset; 20,640 observations evaluating median house values across California districts
california = fetch_california_housing()
X, y = california.data, california.target


# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=43,
)


# Scale the features, ridge is a penalized model, so scaling is important.
# Standardize features by removing the mean and scaling to unit variance.
# Because Ridge applies a penalty to coefficient weights, diffiring features scales will disproportionately effect the model.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# Initalize and train the ridge regression
# a higher value penalizes large coefficients more heavily
# alpha=1.0 the model applies L1 penalty to eliminate the least important features setting their coefficients to exactly zero.
# alpha=0.0 the model applies L2 penalty, smooth scaling down all coefficients proportionally without discarding entire features.
ridge_model = Ridge(alpha=0.1)

# Make predictions and evaluate
y_pred = ridge_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"RSquared: {r2:.4f}")
