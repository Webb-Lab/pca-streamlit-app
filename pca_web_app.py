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

# -----------------------------
# FDR correction
# -----------------------------
def benjamini_hochberg(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n+1)
    qvals = pvals * n / ranked
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    return np.clip(qvals, 0, 1)

# -----------------------------
# App
# -----------------------------
st.title("PCA + Clustering + Volcano + Export")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="file")

if uploaded_file:

    df = pd.read_excel(uploaded_file, header=0, index_col=0)
    st.subheader("Raw Data")
    st.dataframe(df)

    # -----------------------------
    # Data prep
    # -----------------------------
    data = df.T.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # -----------------------------
    # PCA
    # -----------------------------
    max_pc = min(10, data.shape[1])
    n_pc = st.slider("PCA components", 2, max_pc, 2, key="pc")

    pca = PCA(n_components=n_pc)
    pca_res = pca.fit_transform(scaled)

    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = explained_var.cumsum()

    # Scree plot
    st.subheader("Scree Plot")
    fig_scree, ax = plt.subplots()
    ax.plot(range(1, n_pc+1), explained_var, marker='o', label="Individual")
    ax.plot(range(1, n_pc+1), cumulative_var, marker='s', linestyle='--', label="Cumulative")
    ax.legend(); ax.grid(True)
    st.pyplot(fig_scree)

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=data.columns,
        columns=[f'PC{i+1}' for i in range(n_pc)]
    )

    # -----------------------------
    # Clustering diagnostics
    # -----------------------------
    max_k = min(10, len(data))

    st.subheader("Elbow Plot")
    inertia = []
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(pca_res)
        inertia.append(km.inertia_)

    fig_elbow, ax = plt.subplots()
    ax.plot(range(1, max_k+1), inertia, marker='o')
    ax.grid(True)
    st.pyplot(fig_elbow)

    st.subheader("Silhouette Plot")
    sil_scores = []
    for k in range(2, max_k):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(pca_res)
        sil_scores.append(silhouette_score(pca_res, labels))

    fig_sil, ax = plt.subplots()
    ax.plot(range(2, max_k), sil_scores, marker='o')
    ax.grid(True)
    st.pyplot(fig_sil)

    k = st.slider("Number of clusters", 2, max_k, min(3, max_k), key="cluster")

    km = KMeans(n_clusters=k, random_state=42)
    clusters = km.fit_predict(pca_res)

    scores = pd.DataFrame(
        pca_res,
        index=data.index,
        columns=[f'PC{i+1}' for i in range(n_pc)]
    )
    scores['Cluster'] = clusters

    # Auto label
    centroids = scores.groupby('Cluster').mean().sort_values(by='PC1')
    mapping = {old: f'Cluster_{i+1}' for i, old in enumerate(centroids.index)}
    scores['Cluster_Label'] = scores['Cluster'].map(mapping)

    # PCA plot
    st.subheader("PCA Plot")
    fig = px.scatter(scores, x='PC1', y='PC2', color='Cluster_Label')
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Differential analysis
    # -----------------------------
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
                rows.append([cl, col, fc, p])

    diff = pd.DataFrame(rows, columns=["Cluster","Crosslink","log2FC","p"])

    # FDR
    diff['q'] = np.nan
    for cl in diff['Cluster'].unique():
        mask = diff['Cluster']==cl
        diff.loc[mask,'q'] = benjamini_hochberg(diff.loc[mask,'p'])

    diff['-log10p'] = -np.log10(diff['p'])

    # -----------------------------
    # Cluster summaries
    # -----------------------------
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = scores['Cluster_Label']

    cluster_means = data_with_clusters.groupby('Cluster').mean()
    global_mean = data.mean()
    fold_change = cluster_means / global_mean

    top_enriched = {}
    for c in fold_change.index:
        top_enriched[c] = fold_change.loc[c].sort_values(ascending=False).head(10)
    top_enriched_df = pd.DataFrame(top_enriched)

    # -----------------------------
    # Volcano plot
    # -----------------------------
    st.subheader("Volcano Plot")

    fc_thresh = st.slider("log2FC threshold", 0.0, 5.0, 1.0)
    q_thresh = st.slider("FDR threshold", 0.0001, 0.2, 0.05)
    y_cap = st.slider("Y-axis max (-log10 p)", 5, 20, 10)

    x_cap = 10

    plot_df = diff.copy()
    plot_df['x'] = np.clip(plot_df['log2FC'], -x_cap, x_cap)
    plot_df['y'] = np.clip(plot_df['-log10p'], 0, y_cap)

    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='Cluster',
        hover_data=['Crosslink']
    )

    fig.add_vline(x=fc_thresh)
    fig.add_vline(x=-fc_thresh)
    fig.add_hline(y=-np.log10(q_thresh))

    fig.update_xaxes(range=[-x_cap-1, x_cap+1])
    fig.update_yaxes(range=[0, y_cap+1])

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Excel export
    # -----------------------------
    output = io.BytesIO()

    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_pc)],
        'Explained Variance (%)': explained_var,
        'Cumulative (%)': cumulative_var
    })

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        scores.to_excel(writer, sheet_name="Scores")
        loadings.to_excel(writer, sheet_name="Loadings")
        variance_df.to_excel(writer, sheet_name="Variance")
        diff.to_excel(writer, sheet_name="Differential")
        cluster_means.to_excel(writer, sheet_name="Cluster Means")
        fold_change.to_excel(writer, sheet_name="Fold Change")
        top_enriched_df.to_excel(writer, sheet_name="Top Enriched")
        df.to_excel(writer, sheet_name="Original Data")

    st.download_button(
        "Download Full Excel Output",
        data=output.getvalue(),
        file_name="pca_analysis_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
