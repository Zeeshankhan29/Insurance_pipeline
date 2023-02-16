import streamlit as st
import pandas as pd
import pickle

st.title("Insurance Prediction")

st.write("Enter the following information to get a personalized insurance quote:")

age = st.number_input("Age",step=1)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI")
children = st.number_input("Children",step=1)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

if st.button("Get Quote"):
    data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }
    df = pd.DataFrame(data, index=[0])
    st.write("Here is your input data:")
    st.write(df)

    # Load the saved pipeline object from the pickle file
    with open('pipeline','rb') as f:
        pipeline_obj = pickle.load(f)
    # Transform the input data using the pipeline object
    transformed_data = pipeline_obj.transform(df)

    # Make a prediction using the transformed data and the loaded model
    with open('best_model','rb') as f1:
        ob = pickle.load(f1)
    prediction = ob.predict(transformed_data)
    st.write("Your personalized insurance quote is:")
    st.write(prediction[0])



