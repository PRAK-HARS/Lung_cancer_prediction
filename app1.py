import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_lungs.pkl', 'rb') as file:
    label_encoder_lungs = pickle.load(file)

st.title('Lung Cancer Risk Prediction')

# Function to preprocess input data
def preprocess_input(gender, age, smoking, yellow_finger, anxiety, peer_pressure, chronic_disease,
                     fatigue, allergy, wheezing, alcohol_consumption, coughing,
                     shortness_of_breath, swallowing_difficulty, chest_pain):
    # Convert categorical inputs ('Yes'/'No') to binary (0/1)
    categorical_features = [smoking, yellow_finger, anxiety, peer_pressure, chronic_disease,
                            fatigue, allergy, wheezing, alcohol_consumption, coughing,
                            shortness_of_breath, swallowing_difficulty, chest_pain]
    
    categorical_features_binary = [1 if feature == 'Yes' else 0 for feature in categorical_features]
    
    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Construct input array
    input_data = np.array([[gender_encoded, age] + categorical_features_binary])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

# Streamlit app inputs
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 97)
smoking = st.selectbox('Smoking', ['No', 'Yes'])
yellow_finger = st.selectbox('Yellow Fingers', ['No', 'Yes'])
anxiety = st.selectbox('Anxiety', ['No', 'Yes'])
peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'])
chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'])
fatigue = st.selectbox('Fatigue', ['No', 'Yes'])
allergy = st.selectbox('Allergy', ['No', 'Yes'])
wheezing = st.selectbox('Wheezing', ['No', 'Yes'])
alcohol_consumption = st.selectbox('Alcohol Consumption', ['No', 'Yes'])
coughing = st.selectbox('Coughing', ['No', 'Yes'])
shortness_of_breath = st.selectbox('Shortness of Breath', ['No', 'Yes'])
swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['No', 'Yes'])
chest_pain = st.selectbox('Chest Pain', ['No', 'Yes'])

# Predict function
def predict_lung_cancer_risk(gender, age, smoking, yellow_finger, anxiety, peer_pressure, chronic_disease,
                             fatigue, allergy, wheezing, alcohol_consumption, coughing,
                             shortness_of_breath, swallowing_difficulty, chest_pain):
    input_data_scaled = preprocess_input(gender, age, smoking, yellow_finger, anxiety, peer_pressure, chronic_disease,
                                         fatigue, allergy, wheezing, alcohol_consumption, coughing,
                                         shortness_of_breath, swallowing_difficulty, chest_pain)
    
    prediction = model.predict(input_data_scaled)
    return prediction

if st.button('Predict'):
    prediction = predict_lung_cancer_risk(gender, age, smoking, yellow_finger, anxiety, peer_pressure, chronic_disease,
                                          fatigue, allergy, wheezing, alcohol_consumption, coughing,
                                          shortness_of_breath, swallowing_difficulty, chest_pain)
    
    st.write(f'The predicted risk of lung cancer is: {prediction[0][0]}')

    if prediction >= 0.7:
        st.write('Recommendation: High risk (Yes)')
    elif prediction <= 0.4:
        st.write('Recommendation: Low risk (No)')
    else :
        st.write('Recommendation: Moderate Risk')
