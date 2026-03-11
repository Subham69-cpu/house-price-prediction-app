import streamlit as st
import numpy as np
import joblib

# Load saved model and preprocessing objects
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter the house details below")

# Dropdown inputs
location = st.selectbox("Location", encoders['Location'].classes_)
furnishing = st.selectbox("Furnishing", encoders['Furnishing'].classes_)
property_type = st.selectbox("Property Type", encoders['Property_Type'].classes_)
city_zone = st.selectbox("City Zone", encoders['City_Zone'].classes_)
transaction = st.selectbox("Transaction Type", encoders['Transaction_Type'].classes_)

# Numeric inputs
area = st.number_input("Area (sqft)", min_value=100)
bhk = st.number_input("BHK", min_value=1)
floor = st.number_input("Floor", min_value=0)
age = st.number_input("Property Age (Years)", min_value=0)
distance = st.number_input("Distance from City Center (km)", min_value=0.0)

if st.button("Predict Price"):

    location = encoders['Location'].transform([location])[0]
    furnishing = encoders['Furnishing'].transform([furnishing])[0]
    property_type = encoders['Property_Type'].transform([property_type])[0]
    city_zone = encoders['City_Zone'].transform([city_zone])[0]
    transaction = encoders['Transaction_Type'].transform([transaction])[0]

    numeric_data = np.array([[area, bhk, age, floor, distance]])

    scaled_numeric = scaler.transform(numeric_data)

    area, bhk, age, floor, distance = scaled_numeric[0]

    input_data = np.array([[location, area, bhk, furnishing,
                            property_type, city_zone,
                            transaction, floor, age, distance]])

    prediction = model.predict(input_data)

    price = prediction[0]

    st.success(f"💰 Predicted House Price: ₹ {price:,.2f}")
