import streamlit as st
import numpy as np
import tensorflow  as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

##Loading the trained model
model = tf.keras.models.load_model('model.h5')

## Load the encoders and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)

## Steamlit app
st.title("Customer Churn Prediction App")

#User input
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.slider('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

##Prepare the input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

## On hot encode the geography column
geo_encoded = one_hot_encoder_geography.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
## Combine one hot encoded columns with the rest of the input data
input_data_encoded = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

##Scaling the input data
input_data_scaled = scaler.transform(input_data_encoded)

## Predecting the output using the trained model   
prediction = model.predict(input_data_scaled)
## Get probability of churn
churn_probability = prediction[0][0]

if churn_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")






