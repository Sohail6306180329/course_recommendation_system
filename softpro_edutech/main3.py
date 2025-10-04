# student Performance Analysis
import streamlit as st
import pandas as pd
import joblib

# load model and encoder 
model = joblib.load('student_performance_model.pkl')
technology_encoder = joblib.load('technology_encoder.pkl')
grade_encoder = joblib.load('grade_encoder.pkl')

st.subheader("Student Performance Analysis and Prediction App")
st.write('Fill the student score below to predict thier final grade')

# user input
technology=st.selectbox('Technology',technology_encoder.classes_)
welcome_test=st.slider('Welcome Test Score', 30,50,40)
presentation=st.slider('Presentation Score', 90,150,120)
mini_project=st.slider('Mini Project Score', 60,100,80)
hrskills=st.slider('HR Skills Score', 90,150,120)
project_presentation=st.slider('Project Presentation Score', 160,250,205)
project_submission=st.slider('Project Submission Score', 200,300,251)
attendance=st.slider('Attendance Score', 70,100,85)
discipline=st.slider('Discipline Score',60,100,80)