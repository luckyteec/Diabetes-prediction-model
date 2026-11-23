import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
import streamlit as st


# 1. Load your saved model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')


def diabetes_prediction (input_data):

    input_data_as_numpy =np.asarray(input_data, dtype =float)
    data_reshaped = input_data_as_numpy.reshape(1,-1)
    data_scaled = scaler.transform(data_reshaped)
    prediction = model.predict(data_reshaped)
    
    if  (prediction [0]==1):
        st.error ("Patient is diabetic")
    else:
       st.success ("patient is not diabetic")


def main ():
    st.set_page_config(
        page_title='Diabetes Diagnostic Predictor',
        layout='wide',
    )
    
    st.title('Diabetes Diagnostic Predictor')
    st.write(
        'welcome to L-diagnostic. this is our Diabetes prediction model'
    )
    
    st.header('Medical Report')
    st.write('Please enter your diabetes lab report here')
    #user input
    Pregnancies = st.text_input('Pregnancy number')
    Glucose = st.text_input ('Glucose level' )
    Diastolic = st.text_input('Diastolic level')
    Triceps = st.text_input ('Triceps level' )
    Insulin = st.text_input('Insulin level')
    bmi = st.text_input ('BMI value')
    dpf = st.text_input('DPF value')
    Age = st.text_input ('Age of patient' )

    #prediction
    # diagnosis = ""

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, Diastolic, Triceps, Insulin, bmi, dpf, Age])
    
    
    # st.success(diagnosis)

if __name__ == '__main__':
    main()


