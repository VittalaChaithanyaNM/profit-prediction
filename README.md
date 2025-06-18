Profit Prediction using Machine Learning

This is a beginner-level machine learning project that predicts business profits based on various features like business category, location, and month using a Random Forest Regressor.

🔍 Project Overview

The aim of this project is to build a regression model that can predict profits for different businesses by learning patterns from historical data. It includes data preprocessing, model training, evaluation, and visualization of results.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib

📂 Dataset

The dataset used in this project is a CSV file named `business_data.csv` which contains the following columns:
- `Profit` (Target variable)
- `BusinessCategory`
- `Location`
- `Month`
- Other business-related numerical features

> 📌 Note: Make sure to place the `business_data.csv` file in the appropriate directory as referenced in the script.

⚙️ Workflow

1. **Data Loading**: Read the dataset using `pandas`.
2. **Preprocessing**:
   - Convert categorical columns (`BusinessCategory`, `Location`, `Month`) to dummy variables using one-hot encoding.
3. **Model Building**:
   - Split the data into training and testing sets.
   - Use `RandomForestRegressor` to train the model.
4. **Evaluation**:
   - Predict on test data.
   - Calculate Mean Squared Error (MSE) as the evaluation metric.
5. **Visualization**:
   - Plot actual vs predicted profits using color gradients to represent prediction error.
6. **Feature Importance**:
   - Analyze and save feature importances as a CSV.

📊 Output

- **Mean Squared Error (MSE)** is printed after model evaluation.
- A **scatter plot** shows actual vs predicted profits.
- A CSV file `feature_importances.csv` is created with ranked feature contributions.

🚀 How to Run

1. Clone this repository.
2. Ensure the required libraries are installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
