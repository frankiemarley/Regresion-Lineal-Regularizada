import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv')

# non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# numeric_data = data.select_dtypes(include=[np.number])

# print("Non-numeric columns:", non_numeric_columns)

# non_numeric_columns = ['COUNTY_NAME', 'STATE_NAME']

# # Exclude non-numeric columns from correlation calculation
# numeric_columns = [col for col in data.columns if col not in non_numeric_columns]
# numeric_data = data[numeric_columns]

# # Calculate the correlation matrix for numeric columns
# correlation_matrix = numeric_data.corr()

# # Display correlations with the target variable
# correlations_with_target = correlation_matrix['diabetes_prevalence'].sort_values(ascending=False)
# print(correlations_with_target.head(10))


# # Top ten most correlated variables with 'diabetes_prevalence'
# top_corr_features = correlations_with_target.index[:10]

# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix.loc[top_corr_features, top_corr_features], annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Top 10 Correlated Variables Heatmap')
# plt.show()

# Select predictor variables and target
X = data[['diabetes_Upper 95% CI', 'CKD_prevalence', 'CKD_Upper 95% CI',
          'anycondition_prevalence', 'Heart disease_Lower 95% CI']]
y = data['diabetes_prevalence']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Lasso model
lasso_model = Lasso(alpha=0.1, max_iter=300)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate Lasso model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("Lasso Regression:")
print(f"Mean Squared Error: {mse_lasso}")
print(f"R-squared: {r2_lasso}")
print(f"Mean Absolute Error: {mae_lasso}")
print(f"Coefficients: {lasso_model.coef_}")

# Fit Ridge model
ridge_model = Ridge(alpha=0.1, max_iter=300)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate Ridge model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print("\nRidge Regression:")
print(f"Mean Squared Error: {mse_ridge}")
print(f"R-squared: {r2_ridge}")
print(f"Mean Absolute Error: {mae_ridge}")
print(f"Coefficients: {ridge_model.coef_}")


# #-------------------------------------------------------------------------------------------------
# # Define a color palette for scatterplots
# colors = ['blue', 'green', 'orange', 'red']

# # Plot histogram of diabetes prevalence
# plt.figure(figsize=(10, 6))
# sns.histplot(data['diabetes_prevalence'], kde=True, color='purple')
# plt.title('Distribution of Diabetes Prevalence')
# plt.xlabel('Diabetes Prevalence')
# plt.ylabel('Frequency')
# plt.show()

# # Plot scatterplots of numeric features vs. diabetes prevalence
# numeric_features = ['diabetes_Upper 95% CI', 'diabetes_Lower 95% CI', 'CKD_Lower 95% CI', 'CKD_prevalence']
# fig, axes = plt.subplots(1, len(numeric_features), figsize=(20, 6))

# for i, feature in enumerate(numeric_features):
#     sns.scatterplot(x=data[feature], y=data['diabetes_prevalence'], ax=axes[i], color=colors[i])
#     axes[i].set_title(f'{feature} vs. Diabetes Prevalence')
#     axes[i].set_xlabel(f'{feature}')
#     axes[i].set_ylabel('Diabetes Prevalence')

# plt.tight_layout()
# plt.show()

# # Plot boxplot of urban vs. rural diabetes prevalence
# if 'Urban_rural_code' in data.columns:
#     palette = sns.color_palette("husl", len(data['Urban_rural_code'].unique()))

#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x='Urban_rural_code', y='diabetes_prevalence', data=data, hue='Urban_rural_code', palette=palette, legend=False)
#     plt.title('Urban vs. Rural Diabetes Prevalence')
#     plt.xlabel('Urban Rural Code')
#     plt.ylabel('Diabetes Prevalence')
#     plt.show()
# else:
#     print("Urban_rural_code column not found in the dataset.")
# #-------------------------------------------------------------------------------------------------------

# # Select numeric features for scatterplots
# numeric_features = ['diabetes_Upper 95% CI', 'diabetes_Lower 95% CI', 'CKD_Lower 95% CI', 'CKD_prevalence', 
#                     'CKD_Upper 95% CI', 'anycondition_Lower 95% CI', 'Heart disease_Lower 95% CI', 
#                     'anycondition_prevalence', 'anycondition_Upper 95% CI']

# # Define a color palette for scatterplots
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'pink', 'brown', 'cyan', 'magenta']

# # Calculate number of rows and columns for subplots
# num_rows = (len(numeric_features) + 1) // 2  # Ensure at least 1 row
# num_cols = min(2, len(numeric_features))   # Maximum 2 columns

# # Create subplots for each numeric feature
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# for i, feature in enumerate(numeric_features):
#     row = i // num_cols
#     col = i % num_cols
    
#     # Define colors
#     scatter_color = colors[i]
#     line_color = 'black'  # Get a darker shade
    
#     sns.regplot(x=X[feature], y=y, ax=axes[row, col], scatter_kws={'color': scatter_color}, line_kws={'color': line_color})
#     axes[row, col].set_title(f'{feature} vs. Diabetes Prevalence')
#     axes[row, col].set_xlabel(f'{feature}')
#     axes[row, col].set_ylabel('Diabetes Prevalence')

# # Remove any unused subplots
# for j in range(len(numeric_features), num_rows * num_cols):
#     fig.delaxes(axes.flatten()[j])

# plt.tight_layout()
# plt.show()


# # Create a function to plot residuals
# def plot_residuals(model, X, y):
#     y_pred = model.predict(X)
#     residuals = y - y_pred
#     plt.figure(figsize=(10, 6))
#     sns.residplot(x=y_pred, y=residuals, lowess=True, color='green')
#     plt.title('Residual Plot')
#     plt.xlabel('Fitted values')
#     plt.ylabel('Residuals')
#     plt.show()

# # Plot residuals for Lasso model
# plot_residuals(lasso_model, X_test, y_test)

# # Plot residuals for Ridge model
# plot_residuals(ridge_model, X_test, y_test)

# # Plot distribution of predicted vs. actual values
# plt.figure(figsize=(10, 6))
# sns.kdeplot(y_test, label='Actual', color='blue', linewidth=2)
# sns.kdeplot(y_pred_lasso, label='Predicted (Lasso)', color='green', linewidth=2)
# sns.kdeplot(y_pred_ridge, label='Predicted (Ridge)', color='orange', linewidth=2)
# plt.title('Distribution of Actual vs. Predicted Diabetes Prevalence')
# plt.xlabel('Diabetes Prevalence')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

# # Plot coefficients for Lasso and Ridge models
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# axes[0].bar(X.columns, lasso_model.coef_, color='blue')
# axes[0].set_title('Lasso Coefficients')
# axes[0].set_xlabel('Features')
# axes[0].set_ylabel('Coefficient Value')
# axes[0].tick_params(axis='x', rotation=45)

# axes[1].bar(X.columns, ridge_model.coef_, color='green')
# axes[1].set_title('Ridge Coefficients')
# axes[1].set_xlabel('Features')
# axes[1].set_ylabel('Coefficient Value')
# axes[1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.show()


# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Plot feature importances
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Plot residuals
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=y_test-y_pred, lowess=True, scatter_kws={'alpha': 0.5})
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"Cross-validation R2 scores: {cv_scores}")
print(f"Mean R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

