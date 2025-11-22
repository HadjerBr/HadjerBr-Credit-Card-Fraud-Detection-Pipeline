# app/app.py

import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_data
from src.model_utils import train_and_evaluate


st.title("Kredi Kartı Sahtekarlık Tespiti - Hiperparametre Deney Arayüzü")
st.write(
    "Bu arayüzde lojistik regresyon modelinin **öğrenme oranı (learning rate)**, "
    "**iterasyon sayısı** ve **sınıflandırma eşiği (threshold)** hiperparametrelerini "
    "değiştirerek modelin performans metriklerini gözlemleyebilirsiniz.\n\n"
    "Model, dengelenmiş train verisi üzerinde eğitilip, orijinal dengesiz test verisi "
    "üzerinde değerlendirilmiştir."
)


@st.cache_resource
def get_data():
    X_train, y_train, X_test, y_test = load_data("notebook/preprocessed_data.npz")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_data()

# Sidebar: Hyperparameters 
st.sidebar.header("Hiperparametre Seçimi")

lr = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=1)
iters = st.sidebar.slider("Iterasyon Sayısı", min_value=100, max_value=3000, step=100, value=1000)
threshold = st.sidebar.slider("Sınıflandırma Eşiği (Threshold)", 0.1, 0.9, 0.5, 0.05)

if st.button("Modeli Eğit ve Değerlendir"):
    with st.spinner("Model eğitiliyor..."):
        model, metrics = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            learning_rate=lr,
            iterations=iters,
            threshold=threshold,
        )

    st.subheader("Performans Metrikleri (Test Set - Dengesiz)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Accuracy**: {metrics['accuracy']:.4f}")
        st.write(f"**Precision**: {metrics['precision']:.4f}")
        st.write(f"**Recall**: {metrics['recall']:.4f}")
    with col2:
        st.write(f"**F1-score**: {metrics['f1']:.4f}")
        st.write(f"**ROC-AUC**: {metrics['auc']:.4f}")

    st.markdown("---")
    st.subheader("Confusion Matrix Değerleri")
    st.write(f"TP (Fraud doğru): **{metrics['TP']}**")
    st.write(f"TN (Normal doğru): **{metrics['TN']}**")
    st.write(f"FP (Normal → Fraud): **{metrics['FP']}**")
    st.write(f"FN (Fraud → Normal): **{metrics['FN']}**")

    cm = np.array(
        [
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]],
        ]
    )

    st.write("Confusion Matrix (tablo):")
    st.table(
        pd.DataFrame(
            cm,
            index=["Gerçek Normal", "Gerçek Fraud"],
            columns=["Tahmin Normal", "Tahmin Fraud"],
        )
    )

    st.markdown("---")
    st.subheader("ROC Eğrisi")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(metrics["fprs"], metrics["tprs"], label=f"AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)
else:
    st.info("Soldan hiperparametreleri seçip **'Modeli Eğit ve Değerlendir'** butonuna basın.")
