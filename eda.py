# eda.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")

def run_eda(df, max_hist_cols=6):
    st.subheader("🚀 Basic info & statistics")
    st.write("Rows, cols:", df.shape)
    st.write(df.dtypes)
    st.write("Missing values:")
    st.write(df.isna().sum())

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe().T)

    st.subheader("Scatter XY colored by Z")
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(df['X'], df['Y'], c=df['Z'], s=25, cmap='terrain')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    fig.colorbar(sc, ax=ax, label='Z')
    st.pyplot(fig)

    st.subheader("Distributions")
    cols = ['X','Y','Z']
    for c in cols:
        fig, ax = plt.subplots(figsize=(5,2.5))
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(c)
        st.pyplot(fig)

    st.subheader("Pairwise correlations")
    corr = df[['X','Y','Z']].corr()
    st.write(corr)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    st.pyplot(fig)

    # Optional: show sample points
    st.subheader("Sample points (first 10 rows)")
    st.dataframe(df.head(10))