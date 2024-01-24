import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_classifier_model.pkl")

# Create a Streamlit app
st.title("Credit Card Fraud Detection App")


# Add user input features
st.sidebar.header("User Input Features")

# Example input fields, you can customize
limit_bal = st.sidebar.number_input("X1", value=0.00, key="1")
Pay_0 = st.sidebar.selectbox("X6", [-2,-1,0,1,2,3,4,5,6,7,8,9])
Pay_2 = st.sidebar.selectbox("X7", [-2,-1,0,1,2,3,4,5,6,7,8,9])
Pay_3 = st.sidebar.selectbox("X8", [-2,-1,0,1,2,3,4,5,6,7,8,9])
Pay_4 = st.sidebar.selectbox("X9", [-2,-1,0,1,2,3,4,5,6,7,8,9])
Pay_5 = st.sidebar.selectbox("X10", [-2,-1,0,1,2,3,4,5,6,7,8,9])
Pay_6 = st.sidebar.selectbox("X11", [-2,-1,0,1,2,3,4,5,6,7,8,9])
Pay_7 = st.sidebar.number_input("X18", value=0.00, key="2")
Pay_8 = st.sidebar.number_input("X19", value=0.00, key="3")
Pay_9 = st.sidebar.number_input("X19", value=0.00, key="4")


# Create a dictionary with user input
user_input = {
    "X1": limit_bal,
    "X6": Pay_0,
    "X7": Pay_2,
    "X8": Pay_3,
    "X9": Pay_4,
    "X10": Pay_5,
    "X11": Pay_6,
    "X18": Pay_7,
    "X19": Pay_8,
    "X20": Pay_9
    # Include the features here
}

# When the user clicks the 'Predict' button
if st.button("Predict"):
    # Preprocess user input and make a prediction
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    st.write("Prediction:", prediction)
