# eda.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")

def run_eda(df, max_hist_cols=6):
    st.subheader("🚀 Базовая информация и статистика")
    st.write("Строк, столбцов:", df.shape)
    st.write(df.dtypes)
    st.write("Пропуски:")
    st.write(df.isna().sum())

    st.subheader("Описательная статистика")
    st.dataframe(df.describe().T)

    st.subheader("Рассеяние XY, цвет — Z")
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(df['X'], df['Y'], c=df['Z'], s=25, cmap='terrain')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    fig.colorbar(sc, ax=ax, label='Z')
    st.pyplot(fig)

    st.subheader("Распределения")
    cols = ['X','Y','Z']
    for c in cols:
        fig, ax = plt.subplots(figsize=(5,2.5))
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(c)
        st.pyplot(fig)

    st.subheader("Парные корреляции")
    corr = df[['X','Y','Z']].corr()
    st.write(corr)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    st.pyplot(fig)

    # Дополнительно: показать первые строки
    st.subheader("Первые 10 строк")
    st.dataframe(df.head(10))