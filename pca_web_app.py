import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
import io
from scipy.stats import ttest_ind

def benjamini_hochberg(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n+1)
    qvals = pvals * n / ranked
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    return np.clip(qvals, 0, 1)

st.title("PCA + Clustering + Volcano")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:

    df = pd.read_excel(uploaded_file, header=0, index_col=0)
    data = df.T.copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # PCA
    n_pc = st.slider("PCA components", 2, min(10, data.shape[1]), 2)
    pca = PCA(n_components=n_pc)
    pca_res = pca.fit_transform(scaled)

    explained_var = pca.explained_variance_ratio_ * 100

    # Scree plot
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(range(1, n_pc+1), explained_var, marker='o')
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance (%)")

    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(colors='black')

    st.pyplot(fig)

    # Clustering
    k = st.slider("Clusters", 2, min(10, len(data)), 3)
    km = KMeans(n_clusters=k, random_state=42)
    clusters = km.fit_predict(pca_res)

    scores = pd.DataFrame(pca_res, index=data.index, columns=[f'PC{i+1}' for i in range(n_pc)])
    scores['Cluster'] = clusters

    centroids = scores.groupby('Cluster').mean().sort_values(by='PC1')
    mapping = {old: f'Cluster_{i+1}' for i, old in enumerate(centroids.index)}
    scores['Cluster_Label'] = scores['Cluster'].map(mapping)

    # PCA plot
    fig = px.scatter(scores, x='PC1', y='PC2', color='Cluster_Label')

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black")
    )

    st.plotly_chart(fig)

    # Differential
    eps = 1e-6
    rows = []

    for cl in scores['Cluster_Label'].unique():
        mask = scores['Cluster_Label'] == cl
        g1 = data[mask]
        g2 = data[~mask]

        for col in data.columns:
            v1 = g1[col].dropna()
            v2 = g2[col].dropna()

            if len(v1) > 0 and len(v2) > 0:
                _, p = ttest_ind(v1, v2, equal_var=False)
                fc = np.log2((v1.mean()+eps)/(v2.mean()+eps))
                rows.append([cl, col, fc, p, v1.mean(), v2.mean()])

    diff = pd.DataFrame(rows, columns=["Cluster","Crosslink","log2FC","p","mean_cluster","mean_other"])

    diff['q'] = np.nan
    for cl in diff['Cluster'].unique():
        mask = diff['Cluster']==cl
        diff.loc[mask,'q'] = benjamini_hochberg(diff.loc[mask,'p'])

    diff['-log10p'] = -np.log10(diff['p'])

    # Volcano
    fc_thresh = st.slider("log2FC threshold", 0.0, 5.0, 1.0)
    q_thresh = st.slider("FDR threshold", 0.0001, 0.2, 0.05)

    fig = px.scatter(diff, x="log2FC", y="-log10p", color="Cluster")

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        xaxis_title="log2FC",
        yaxis_title="-log10(p-value)"
    )

    fig.add_vline(x=fc_thresh)
    fig.add_vline(x=-fc_thresh)
    fig.add_hline(y=-np.log10(q_thresh))

    st.plotly_chart(fig)
