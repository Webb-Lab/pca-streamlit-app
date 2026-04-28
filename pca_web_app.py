import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
st.title("PCA + Clustering + Publication Volcano Figures")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:

    df = pd.read_excel(uploaded_file, header=0, index_col=0)
    data = df.T.copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # PCA
    n_pc = st.slider("PCA components", 2, min(10, data.shape[1]), 2, key="pc")
    pca = PCA(n_components=n_pc)
    pca_res = pca.fit_transform(scaled)

    # Clustering
    k = st.slider("Clusters", 2, min(10, len(data)), 3, key="k")
    km = KMeans(n_clusters=k, random_state=42)
    clusters = km.fit_predict(pca_res)

    scores = pd.DataFrame(pca_res, index=data.index, columns=[f'PC{i+1}' for i in range(n_pc)])
    scores['Cluster'] = clusters

    centroids = scores.groupby('Cluster').mean().sort_values(by='PC1')
    mapping = {old: f'Cluster_{i+1}' for i, old in enumerate(centroids.index)}
    scores['Cluster_Label'] = scores['Cluster'].map(mapping)

    # PCA plot
    st.subheader("PCA")
    fig = px.scatter(scores, x='PC1', y='PC2', color='Cluster_Label')
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Differential
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
    diff['q'] = np.nan

    for cl in diff['Cluster'].unique():
        mask = diff['Cluster']==cl
        diff.loc[mask,'q'] = benjamini_hochberg(diff.loc[mask,'p'])

    diff['-log10p'] = -np.log10(diff['p'])

    # -----------------------------
    # Settings
    # -----------------------------
    fc_thresh = st.slider("log2FC threshold", 0.0, 5.0, 1.0)
    q_thresh = st.slider("FDR threshold", 0.0001, 0.2, 0.05)

    x_lim = 10
    y_lim = 10

    # -----------------------------
    # MULTI-PANEL FIGURE
    # -----------------------------
    st.subheader("Publication Figure")

    clusters_unique = diff['Cluster'].unique()
    n = len(clusters_unique)

    cols = min(3, n)
    rows_fig = int(np.ceil(n/cols))

    fig_multi, axes = plt.subplots(rows_fig, cols, figsize=(5*cols, 5*rows_fig))

    if n == 1:
        axes = np.array([[axes]])

    axes = axes.flatten()

    for i, cl in enumerate(clusters_unique):
        ax = axes[i]
        sub = diff[diff['Cluster']==cl].copy()

        sub['x'] = np.clip(sub['log2FC'], -x_lim, x_lim)
        sub['y'] = np.clip(sub['-log10p'], 0, y_lim)

        sig = (abs(sub['log2FC']) > fc_thresh) & (sub['q'] < q_thresh)

        ax.scatter(sub['x'], sub['y'],
                   c=sig.map({True:'red',False:'gray'}),
                   s=10)

        # Threshold lines
        ax.axvline(fc_thresh, linestyle='--')
        ax.axvline(-fc_thresh, linestyle='--')
        ax.axhline(-np.log10(q_thresh), linestyle='--')

        ax.set_title(cl)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(0, y_lim)

        # ---- Smart-ish label spacing ----
        top_hits = sub.sort_values('q').head(8)

        used_positions = []

        for _, r in top_hits.iterrows():
            x = r['x']
            y = r['y']

            # shift if too close to existing labels
            for (px, py) in used_positions:
                if abs(x - px) < 0.5 and abs(y - py) < 0.5:
                    y += 0.5

            ax.text(x, y, r['Crosslink'], fontsize=6)
            used_positions.append((x, y))

    # Remove empty panels
    for j in range(i+1, len(axes)):
        fig_multi.delaxes(axes[j])

    plt.tight_layout()

    st.pyplot(fig_multi)

    # Save combined figure
    buf = io.BytesIO()
    fig_multi.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)

    st.download_button(
        "Download Multi-Panel Volcano Figure",
        data=buf.getvalue(),
        file_name="volcano_figure.png",
        mime="image/png"
    )

    plt.close(fig_multi)
