import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = pickle.load(open("titanic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details below to predict survival.")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
cabin_deck = st.selectbox("Cabin Deck", ['A','B','C','D','E','F','G','T','U'])  # 'U' = unknown

# Convert input to model format
def preprocess_input():
    sex_val = 1 if sex == "female" else 0
    embarked_vals = {"C": 0, "Q": 0}
    if embarked in embarked_vals:
        embarked_vals[embarked] = 1
    cabin_vals = {key: 0 for key in ['B', 'C', 'D', 'E', 'F', 'G', 'T', 'U']}
    if cabin_deck in cabin_vals:
        cabin_vals[cabin_deck] = 1

    input_dict = {
        "Pclass": pclass,
        "Sex": sex_val,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_Q": embarked_vals["Q"],
        "Embarked_S": 1 if embarked == "S" else 0,  # drop_first was True
        **{f"Cabin_Deck_{k}": v for k, v in cabin_vals.items() if k != "A"}  # drop_first for A
    }

    input_df = pd.DataFrame([input_dict])
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    return input_df

# Predict
if st.button("Predict Survival"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"🎉 Survived with probability: {prob:.2f}")
    else:
        st.error(f"☠️ Did not survive (probability of survival: {prob:.2f})")
