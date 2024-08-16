import streamlit as st
import pandas as pd
import plotly.express as px 
import joblib
import tensorflow as tf
import numpy as np
import sklearn
import requests
import plotly.express as px 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler():
    model = load_model("E:\\COISAS\\FIAP DATA ANALYTICS\\FASE 5\\Streamlit\\Modelos\\multilayer-perceptron.keras")
    scaler = joblib.load("E:\\COISAS\\FIAP DATA ANALYTICS\\FASE 5\\Streamlit\\Modelos\\scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Previsão de nota média para bolsa com input dos indicadores da Passos Mágicos")

# Interface para o usuário inserir os inputs
with st.form("input_form"):
    inputs = {}
    inputs["IAA"] = st.number_input("Digite o valor para IAA", step=0.1)
    inputs["IAN"] = st.number_input("Digite o valor para IAN", step=0.1)
    inputs["IDA"] = st.number_input("Digite o valor para IDA", step=0.1)
    inputs["IEG"] = st.number_input("Digite o valor para IEG", step=0.1)
    inputs["INDE"] = st.number_input("Digite o valor para INDE", step=0.1)
    inputs["IPP"] = st.number_input("Digite o valor para IPP", step=0.1)
    inputs["IPS"] = st.number_input("Digite o valor para IPS", step=0.1)
    inputs["IPV"] = st.number_input("Digite o valor para IPV", step=0.1)
    submit = st.form_submit_button("Prever média para bolsa")

if submit:
    # Convertendo os inputs para DataFrame com os nomes corretos das colunas
    input_df = pd.DataFrame([inputs])

    # Escalonar os inputs
    input_scaled = scaler.transform(input_df)

    # Fazer a previsão
    prediction = model.predict(input_scaled)
    
    # Se o modelo foi treinado para prever valores normalizados, você pode precisar "desnormalizar" a saída
    # Aqui estou assumindo que o target foi normalizado entre 0 e 10
    min_val = 0.0  # substitua pelo valor mínimo do target nos dados de treinamento
    max_val = 10.0  # substitua pelo valor máximo do target nos dados de treinamento
    prediction_original_scale = prediction[0][0] * (max_val - min_val) + min_val
    
    # Exibindo o resultado da previsão
    st.write(f"Nota média do aluno referente aos indicadore: {prediction_original_scale}")

