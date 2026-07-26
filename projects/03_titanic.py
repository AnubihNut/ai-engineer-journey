# Titanic Sex Survival Prediction
#Objetive:
#Build my first Machine Learning model using Scikit-Learn to predict passenger sex survival on the Titanic.
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("This is a simple example of building a machine learning model using Scikit-Learn to predict passenger sex survival on the Titanic dataset. The model is evaluated using various metrics such as MAE, MSE, RMSE, and R2.")
# Load the Titanic dataset from seaborn
df = sns.load_dataset("titanic")

#Show the first few rows of the dataset to get an overview of the data
print("First few rows of the dataset:")
print(df.head())

# Check for missing values in the dataset
print("Missing values in the dataset:")
print(df.isnull().sum())

# Calculate the mean age of passengers in the dataset
print("Mean age:", df["age"].mean())

# Fill missing values in the "age" column with the mean age
df_clean = df.copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].mean())
df_clean["age"].isnull().sum()
df_clean["age"].mean()

# Drop columns that are not needed for the model
df_clean = df_clean.drop(columns=["alive"])
df_clean = df_clean.drop(columns=["adult_male"])
df_clean = df_clean.drop(columns=["embark_town"])
df_clean = df_clean.drop(columns=["class"])
df_clean = df_clean.drop(columns=["who"])
df_clean = df_clean.drop(columns=["deck"])

print("Columns after dropping:", df_clean.columns)
print("Missing values after cleaning:")
print(df_clean.isnull().sum())

# Change alpha values into numeric
df_clean["sex"] = df_clean["sex"].map({"male": 0, "female": 1})
df_clean["embarked"] = df_clean["embarked"].fillna("S").map({"C": 0, "Q": 1, "S": 2})
df_clean["alone"] = df_clean["alone"].map({False: 0, True: 1})

print("Data types after conversion:")
print(df_clean.dtypes)

# Split the dataset into features (X) and target variable (y)
X = df_clean.drop(columns=["sex"])
y = df_clean["sex"]

# Create the model and split the data using train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42 #This value is optional and can be set to any integer. It is used to ensure that the split of the data into training and testing sets is reproducible. By setting a random_state, you can get the same split every time you run the code, which is useful for debugging and comparing results across different runs of the model.
)

#Create a Logistic Regression model
model = LogisticRegression() 

# Fit the model to the training data
model.fit(X_train, y_train)

#Make predictions on the test data
predictions = model.predict(X_test)
print("Predictions:", predictions[:10])
print("Actual values:", y_test.head(10).values)

for real, pred in zip(y_test[:10], predictions[:10]):
    print(f"Real: {real} | Predicción: {pred:.3f}")

# Calculate evaluation metrics
print("\nEvaluation Metrics:")

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse:.3f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# Calculate R-squared (R2)
r2 = r2_score(y_test, predictions)
print(f"R-squared (R2): {r2:.3f}")

print("This is the end of the Titanic Sex Survival Prediction model evaluation. The model has been trained and evaluated using various metrics to assess its performance.")
