import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("House_Prices_Advanced.csv")

# Select required columns
df = df[['Location','Area_sqft','BHK','Furnishing','Property_Type',
         'City_Zone','Transaction_Type','Property_Age_Years',
         'Floor','Distance_from_City_Center_km','Price']]

# Numeric columns
num_cols = ['Area_sqft','BHK','Property_Age_Years','Floor','Distance_from_City_Center_km']

# Scale numeric features
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encode categorical columns
cat_cols = ['Location','Furnishing','Property_Type','City_Zone','Transaction_Type']

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df[['Location','Area_sqft','BHK','Furnishing','Property_Type',
        'City_Zone','Transaction_Type','Floor','Property_Age_Years',
        'Distance_from_City_Center_km']]

y = df['Price']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train smaller Random Forest model (important for GitHub size)
model = RandomForestRegressor(
    n_estimators=30,   # smaller model
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Save files
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("Model saved successfully!")