import streamlit as st
import numpy as np
import joblib

# Load saved model, scaler, encoders
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter the property details to predict the house price")

# -------- Categorical Inputs --------
location = st.selectbox("Location", encoders['Location'].classes_)
furnishing = st.selectbox("Furnishing", encoders['Furnishing'].classes_)
property_type = st.selectbox("Property Type", encoders['Property_Type'].classes_)
city_zone = st.selectbox("City Zone", encoders['City_Zone'].classes_)
transaction = st.selectbox("Transaction Type", encoders['Transaction_Type'].classes_)

# -------- Numeric Inputs --------
area = st.number_input("Area (sqft)", min_value=100)
bhk = st.number_input("BHK", min_value=1)
floor = st.number_input("Floor", min_value=0)
age = st.number_input("Property Age (years)", min_value=0)
distance = st.number_input("Distance from City Center (km)", min_value=0.0)

# -------- Prediction --------
if st.button("Predict Price"):

    # Encode categorical values
    location = encoders['Location'].transform([location])[0]
    furnishing = encoders['Furnishing'].transform([furnishing])[0]
    property_type = encoders['Property_Type'].transform([property_type])[0]
    city_zone = encoders['City_Zone'].transform([city_zone])[0]
    transaction = encoders['Transaction_Type'].transform([transaction])[0]

    # Scale numeric features (same as training)
    numeric_data = np.array([[area, bhk, age, floor, distance]])

    scaled_numeric = scaler.transform(
        np.hstack([numeric_data, [[0]]])
    )[:, :-1]

    area, bhk, age, floor, distance = scaled_numeric[0]

    # Final model input
    input_data = np.array([[location, area, bhk, furnishing,
                            property_type, city_zone,
                            transaction, floor, age, distance]])

    # Predict scaled price
    prediction = model.predict(input_data)
    scaled_price = prediction[0]

    # Convert back to real rupees
    real_price = scaler.inverse_transform([[0,0,0,0,0,scaled_price]])[0][-1]

    st.success(f"💰 Predicted House Price: ₹ {real_price:,.2f}")