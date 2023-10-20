import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
from streamlit_player import st_player
import lime
import lime.lime_tabular

# Load the trained models and preprocessor
with open('model_common.pkl', 'rb') as f:
    clf_common = pickle.load(f)
with open('model_rare.pkl', 'rb') as f:
    clf_rare = pickle.load(f)
with open('model_adverse.pkl', 'rb') as f:
    clf_adverse = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the data
df = pd.read_csv('synthetic_drug_data.csv')

# Define the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Adverse event predictor", "About us", "Doctor's speak"])

    if page == "Home":
        st.title("Safety AI: Adverse Event Prediction. Done Responsibly!")
        st.image("landingImage.jpg", caption='Safety AI', use_column_width=True)
        st.subheader("This app uses AI to predict adverse events based on patient data.")    

    elif page == "Adverse event predictor":
        st.title("Adverse Event Predictor V1.0")

        # User inputs
        st.subheader("Patient Data")
        age = st.number_input('Age', min_value=0, max_value=100, value=25)
        sex = st.selectbox('Gender', options=np.append(df['Sex'].unique(), 'Other'))
        indications = st.selectbox('Indicated for', options=df['Indicated for'].unique())
        contraindications = st.selectbox('Concomitant medications', options=df['Concmittant medications'].dropna().unique())
        medical_history = st.selectbox('Medical History', options=df['Medical History'].unique())
        st.subheader("Prescribing Medication")
        medications = st.selectbox('Prescribing Medication', options=df['Prescribing Medication'].unique())

        # Transform the inputs using the preprocessor
        X = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'Indicated for': [indications],
            'Prescribing Medication': [medications],
            'Concmittant medications': [contraindications],
            'Medical History': [medical_history]
        })

        X_transformed = preprocessor.transform(X)

        # Make predictions and measure the time it takes
        if st.button("Predict"):
            st.markdown("**Predict**")
            start_time = time.time()
            y_pred_common = clf_common.predict(X_transformed)
            y_pred_rare = clf_rare.predict(X_transformed)
            y_pred_adverse = clf_adverse.predict(X_transformed)
            end_time = time.time()
            prediction_time = end_time - start_time

            # Display the predictions in a formatted way
            st.markdown(f"**Common Side effects:** {y_pred_common[0]}")
            st.markdown(f"**Rare Side effects:** {y_pred_rare[0]}")
            
            if y_pred_adverse[0] == 1:
                st.markdown("<div style='background-color: #FF7F7F; padding: 10px; border-radius: 5px; color: black'><strong>The patient is likely to experience adverse event(s) of severe life-threatening pancreatitis and/or low blood sugar.</strong></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color: #90EE90; padding: 10px; border-radius: 5px'><strong>The patient is unlikely to experience an adverse event.</strong></div>", unsafe_allow_html=True)
            
            st.markdown(f"**Data Source: drugs.com**")
            st.markdown(f"**Time taken to predict: {prediction_time} seconds**") 
            st.markdown(f"**Faster than a lightning bolt**")

            # Create a bar chart
            chart_data = df[['Age', 'Indicated for']]
            st.bar_chart(chart_data)

            # Get user feedback 
        feedback = st.text_input("Please provide your feedback:")
        if feedback:
            st.write("Thank you for your feedback!")

    elif page == "About us":
        st.title("About Us")
        st.subheader("We are a team of healthcare professionals and data scientists committed to improving patient safety through the power of AI.")
        st.balloons()

    elif page == "Doctor's speak":
        st.title("Doctor's Speak")
        # Embed a YouTube video of testimonials
        st_player("https://youtu.be/smMzxeT7Jws")

if __name__ == "__main__":
    main()
