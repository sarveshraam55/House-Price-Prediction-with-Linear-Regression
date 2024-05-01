import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns



train_df = pd.read_csv('train.csv')
print(train_df.head())

train_df.info()

test_df = pd.read_csv('test.csv')
print(test_df.head())

test_df.info()

y_test = pd.read_csv('sample_submission.csv')
del y_test['Id']
print(y_test)


#train data
numeric_cols =train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_df.select_dtypes(include=['object']).columns

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

encoder = OneHotEncoder(drop='first', sparse=False)


#test data
numeric_cols =test_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = test_df.select_dtypes(include=['object']).columns

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

encoder = OneHotEncoder(drop='first', sparse=False)

# Additional features related to square footage, number of bedrooms, and number of bathrooms
#train_df['TotalRooms'] = train_df['BedroomAbvGr'] + train_df['FullBath']
#train_df['BedBathRatio'] = train_df['BedroomAbvGr'] / (train_df['FullBath'] + 1)  # Adding 1 to prevent division by zero

# Select features and target variable
X_train = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath' ]]
y_train = train_df['SalePrice']

 # Adding 1 to prevent division by zero
X_test = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath',
                   ]]


# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
y_train_reshaped = y_train.values.reshape(-1, 1)
y_train_scaled = scaler.fit_transform(y_train_reshaped)

X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = scaler.fit_transform(y_test)

"""**Apply Linear Regression model**"""

# Create a linear regression model
model = LinearRegression()

#Use the GridSearchCV technique to increase the accuracy of the model
# Define hyperparameters for Grid Search
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False],
    'copy_X': [True, False],
    'n_jobs': [-1, 1, 2]
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train_scaled)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

"""**Evaluate the model and make predictions on the test set**"""

# Evaluate the best model and make predictions on the test set
best_model = grid_search.best_estimator_
test_predictions= best_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test_scaled, test_predictions)

"""**Print predicted values in Excel_Sheet**"""

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
test_predictions= best_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, test_predictions)



# Flatten the multi-dimensional array test_predictions
flattened_predictions = test_predictions.flatten()

# Create the DataFrame
predictions_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': flattened_predictions})

# Write the predictions to an Excel file
predictions_df.to_excel('predictions.xlsx', index=False)

print("Predictions saved to predictions.xlsx")

"""**Compare between predicted values and true values**"""

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_scaled, test_predictions, alpha=0.5)
plt.title('True vs. Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()
