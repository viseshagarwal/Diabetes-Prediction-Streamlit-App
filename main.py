import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model
filename = "Diabetes.pkl"
classifier = pickle.load(open(filename, "rb"))


def predict():
    st.sidebar.header("Team 5 - Diabetes Prediction")

    st.title("🩺 Diabetes Prediction: Team 5")
    st.markdown(
        """
    ## Diabetes Prediction (Only for Females Above 21 Years of Age)
    This application predicts whether a patient has diabetes based on diagnostic measurements.
    The dataset used is from the National Institute of Diabetes and Digestive and Kidney Diseases.
    """
    )

    st.markdown(
        """
    ### Patient Information:
    """
    )
    name = st.text_input("Name:")
    pregnancy = st.number_input("Number of times pregnant:", min_value=0, max_value=20)
    glucose = st.number_input(
        "Plasma Glucose Concentration (mg/dL):", min_value=0.0, max_value=300.0
    )
    bp = st.number_input(
        "Diastolic Blood Pressure (mm Hg):", min_value=0.0, max_value=200.0
    )
    skin = st.number_input(
        "Triceps Skin Fold Thickness (mm):", min_value=0.0, max_value=100.0
    )
    insulin = st.number_input(
        "2-Hour Serum Insulin (mu U/ml):", min_value=0.0, max_value=900.0
    )
    bmi = st.number_input("Body Mass Index (BMI):", min_value=0.0, max_value=70.0)
    dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, max_value=3.0)
    age = st.number_input("Age:", min_value=21, max_value=120)

    st.markdown("**Outcome:** Class variable (0 or 1)")

    submit = st.button("Predict")

    if submit:
        prediction = classifier.predict(
            [[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]]
        )
        if prediction == 0:
            st.success(f"🎉 Congratulation {name}! You are not diabetic.")
        else:
            st.error(f"⚠️ {name}, it seems like you are diabetic. But don't lose hope!")
            st.markdown(
                """
            Here are some tips to help you manage your health:
            [Diabetes Prevention Tips](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)
            """
            )

    # Add a footer
    st.markdown(
        """
    ---
    **Made with ❤️ by Team 5**  
    Muskan Kumari Gupta [2347130]
    Vansh Shah [2347152]
    Visesh Agarwal [2347164]
    Arunoth Symen A [2347215]
    """
    )


def main():
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("MODE", ("About", "Predict Diabetes"))

    if choice == "Predict Diabetes":
        st.markdown(
            '<p style="font-size: 42px; color: #6A5ACD;">Welcome to the Diabetes Prediction App!</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        This application uses a predictive model to help determine the likelihood of a patient having diabetes based on various health metrics.
        """
        )
        predict()
    elif choice == "About":
        st.markdown(
            """
        ### About this App
        This application is designed to assist in predicting diabetes in patients using a machine learning model.
        The model has been trained on data from the National Institute of Diabetes and Digestive and Kidney Diseases.
        """
        )


if __name__ == "__main__":
    main()
