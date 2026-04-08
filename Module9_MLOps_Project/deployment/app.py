import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="chaitram/tourism-package-prediction", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package based on their demographic details and interaction history.
Please provide the required customer and interaction inputs below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=500000)
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, value=30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=10)
PreferredPropertyStar = st.selectbox("Preferred Hotel Star Rating", [1, 2, 3, 4, 5])
Passport = st.selectbox("Passport", [0, 1])
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=10)

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 2,3,4,5)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=10)
DurationOfPitch = st.number_input("Duration of Pitch (seconds)", min_value=0, value=300)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': Type of Contact,
    'CityTier': City Tier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': Marital Status,
    'Designation': Designation,
    'Age': Age,
    'MonthlyIncome': Monthly Income,
    'NumberOfTrips': Number Of Trips,
    'NumberOfPersonVisiting': Number Of Person Visiting,
    'PreferredPropertyStar': Preferred Property Star,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': Number Of Children Visiting,
    'PitchSatisfactionScore': Pitch Satisfaction Score,
    'ProductPitched': Product Pitched,
    'NumberOfFollowups': Number Of Followups,
    'DurationOfPitch': Duration Of Pitch
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
