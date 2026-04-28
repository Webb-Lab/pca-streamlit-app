import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy.stats import ttest_ind

# -----------------------------
# App setup
# -----------------------------
st.title("Interactive PCA Visualization with K-Means Clustering")
st.write("Upload an Excel file: rows = crosslinks, columns = replicates. Column A should contain crosslink IDs.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=0, index_col=0, engine='openpyxl')
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    # -----------------------------
    # Data prep (transpose)
    # -----------------------------
    data = df.T.copy()  # rows = samples, columns = crosslinks

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # -----------------------------
    # PCA
    # -----------------------------
    max_components = min(data.shape[1], 10)
    num_pca_components = st.slider("Select number of PCA components", 2, max_components, 2)

    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(scaled_data)

    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = explained_variance.cumsum()

    # -----------------------------
    # Scree plot
    # -----------------------------
    st.subheader("Scree Plot")
    fig_scree, ax = plt.subplots()
    ax.plot(range(1, num_pca_components + 1), explained_variance, marker='o', label='Individual')
    ax.plot(range(1, num_pca_components + 1), cumulative_variance, marker='s', linestyle='--', label='Cumulative')
    ax.legend(); ax.grid()
    st.pyplot(fig_scree)

    # -----------------------------
    # Loadings
    # -----------------------------
    loadings = pd.DataFrame(
        pca.components_.T,
        index=data.columns,
        columns=[f'PC{i+1}' for i in range(num_pca_components)]
    )

    st.subheader("Loadings")
    st.dataframe(loadings)

    # -----------------------------
    # Clustering
    # -----------------------------
    max_clusters = min(10, len(data))
    num_clusters = st.slider("Number of clusters", 2, max_clusters, min(3, max_clusters))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    # -----------------------------
    # Auto-label clusters (ordered)
    # -----------------------------
    scores_df = pd.DataFrame(pca_result, index=data.index, columns=[f'PC{i+1}' for i in range(num_pca_components)])
    scores_df['Cluster'] = clusters

    centroids = scores_df.groupby('Cluster').mean()
    centroids_sorted = centroids.sort_values(by='PC1')

    cluster_map = {old: f'Cluster_{i+1}' for i, old in enumerate(centroids_sorted.index)}
    scores_df['Cluster_Label'] = scores_df['Cluster'].map(cluster_map)

    # -----------------------------
    # PCA plot
    # -----------------------------
    plot_df = scores_df.copy()

    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Cluster_Label',
        hover_name=plot_df.index,
        title='PCA Plot'
    )
    st.plotly_chart(fig, width='stretch')

    # -----------------------------
    # Crosslink summaries
    # -----------------------------
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = scores_df['Cluster_Label']

    cluster_means = data_with_clusters.groupby('Cluster').mean()
    global_mean = data.mean()
    fold_change = cluster_means / global_mean

    # Top enriched
    top_enriched = {}
    for c in fold_change.index:
        top_enriched[c] = fold_change.loc[c].sort_values(ascending=False).head(10)
    top_enriched_df = pd.DataFrame(top_enriched)

    # -----------------------------
    # Differential analysis
    # -----------------------------
    diff_results = []

    for cluster in scores_df['Cluster_Label'].unique():
        mask = scores_df['Cluster_Label'] == cluster
        group1 = data[mask]
        group2 = data[~mask]

        for col in data.columns:
            vals1 = group1[col].dropna()
            vals2 = group2[col].dropna()

            if len(vals1) > 1 and len(vals2) > 1:
                stat, pval = ttest_ind(vals1, vals2, equal_var=False)
                log2fc = np.log2(vals1.mean() / vals2.mean()) if vals2.mean() != 0 else np.nan

                diff_results.append([cluster, col, log2fc, pval])

    diff_df = pd.DataFrame(diff_results, columns=['Cluster', 'Crosslink', 'log2FC', 'p-value'])

    # -----------------------------
    # Excel export
    # -----------------------------
    output = io.BytesIO()

    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(num_pca_components)],
        'Explained Variance (%)': explained_variance,
        'Cumulative (%)': cumulative_variance
    })

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        scores_df.to_excel(writer, sheet_name='Scores')
        loadings.to_excel(writer, sheet_name='Loadings')
        variance_df.to_excel(writer, sheet_name='Variance')
        cluster_means.to_excel(writer, sheet_name='Cluster Means')
        fold_change.to_excel(writer, sheet_name='Fold Change')
        top_enriched_df.to_excel(writer, sheet_name='Top Enriched')
        diff_df.to_excel(writer, sheet_name='Differential')
        df.to_excel(writer, sheet_name='Original Data')

    st.download_button(
        label="Download Full Analysis Excel",
        data=output.getvalue(),
        file_name="pca_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
