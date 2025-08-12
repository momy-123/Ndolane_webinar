import streamlit as st
import joblib
import numpy as np
model = joblib.load("iris_model.pkl")

st.title("Prediction de l'espèce Iris")
st.write("Entrz les caractéristiques de la fleur pour prédire son espèce")

sepal_length = st.slider("La longueur du sepale (cm)", 4.0,8.0,5.0)
sepal_width = st.slider("La largeur du sepale (cm)", 2.0,4.5,3.0)
petal_length = st.slider("La longueur du sepale (cm)", 1.0,7.0,4.0)
petal_width = st.slider("La largeur du sepale (cm)", 0.1,2.5,1.0)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("prédire"):
    prediction = model.predict(input_data)[0]
    species = ['Setosa', 'Versicolor', 'Verginica']
    st.success(f"L'espèce prédite est: {species[prediction]}")
