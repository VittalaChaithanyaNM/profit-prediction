import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset# Load dataset
data = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Codes\project1\mlproject\business_data.csv")


# Display the first few rows of the dataset
print(data.head())

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=["BusinessCategory", "Location", "Month"], drop_first=True)

# Display the transformed dataset
print("\nTransformed DataFrame:")
print(data.head())

# Split data into features and target variable
X = data.drop(columns=["Profit"])  # Features
y = data["Profit"]                 # Target

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")

# Plot actual vs predicted profit
diff = np.abs(y_test - y_pred)
colors = plt.cm.viridis(diff / diff.max())
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_test, y_pred, c=diff, cmap="viridis", alpha=0.8, edgecolor="k")
plt.colorbar(scatter, label="Absolute Difference")
plt.title("Actual vs Predicted Profit with Color Difference")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.legend()
plt.show()
# Feature importance
feature_importances = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Save the model feature importances
feature_importances.to_csv("feature_importances.csv", index=False)

