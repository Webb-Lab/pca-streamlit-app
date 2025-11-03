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

# Set up the Streamlit app
st.title("Interactive PCA Visualization with K-Means Clustering")
st.write("Upload an Excel file: rows are experimental replicates/samples, columns are features (compound, peptide etc). The script expects the feature names in row 1 and the replicate names in column A of the Excel spreadsheet. This script does not include missing value imputation.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, header=0, index_col=0, engine='openpyxl')
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    data = df.copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    max_components = min(data.shape[1], 10)
    num_pca_components = st.slider("Select number of PCA components", min_value=2, max_value=max_components, value=2)
    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = explained_variance.cumsum()

    # Scree plot
    st.subheader("Scree Plot of Explained Variance")
    fig_scree, ax_scree = plt.subplots()
    ax_scree.plot(range(1, num_pca_components + 1), explained_variance, marker='o', linestyle='-', label='Individual')
    ax_scree.plot(range(1, num_pca_components + 1), cumulative_variance, marker='s', linestyle='--', label='Cumulative')
    ax_scree.set_xlabel("Principal Component")
    ax_scree.set_ylabel("Explained Variance (%)")
    ax_scree.set_title("Scree Plot")
    ax_scree.legend()
    ax_scree.grid(True)
    st.pyplot(fig_scree)

    # PCA Loadings
    st.subheader("PCA Loading Matrix")
    loadings = pd.DataFrame(pca.components_.T, index=data.columns, columns=[f'PC{i+1}' for i in range(num_pca_components)])
    st.dataframe(loadings)

    # PCA Loadings Heatmap
    st.subheader("PCA Loadings Heatmap")
    fig_loadings, ax_loadings = plt.subplots(figsize=(10, min(0.5 * len(data.columns), 12)))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', ax=ax_loadings)
    ax_loadings.set_title("PCA Loadings Heatmap")
    st.pyplot(fig_loadings)

    # Clustering
    max_possible_clusters = min(10, len(data))
    st.subheader("Elbow Method to Help Choose Optimal Number of Clusters")
    inertia_values = []
    cluster_range = range(1, max_possible_clusters + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_result)
        inertia_values.append(kmeans.inertia_)
    fig_elbow, ax = plt.subplots()
    ax.plot(cluster_range, inertia_values, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.grid(True)
    st.pyplot(fig_elbow)

    st.subheader("Silhouette Score for Cluster Counts")
    silhouette_scores = []
    silhouette_range = range(2, min(max_possible_clusters, len(data) - 1) + 1)
    for k in silhouette_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pca_result)
        score = silhouette_score(pca_result, labels)
        silhouette_scores.append(score)
    fig_silhouette, ax2 = plt.subplots()
    ax2.plot(silhouette_range, silhouette_scores, marker='o', color='green')
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs Number of Clusters")
    ax2.grid(True)
    st.pyplot(fig_silhouette)

    num_clusters = st.slider("Select number of clusters for K-Means", min_value=2, max_value=max_possible_clusters, value=min(3, max_possible_clusters))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    plot_df = pd.DataFrame({
        f'PC1 ({explained_variance[0]:.2f}%)': pca_result[:, 0],
        f'PC2 ({explained_variance[1]:.2f}%)': pca_result[:, 1],
        'Feature': data.index,
        'Cluster': clusters
    })
    for i, col in enumerate(data.columns):
        plot_df[f'Replicate_{i+1}'] = data[col].values

    fig_2d = px.scatter(
        plot_df,
        x=f'PC1 ({explained_variance[0]:.2f}%)',
        y=f'PC2 ({explained_variance[1]:.2f}%)',
        color=plot_df['Cluster'].astype(str),
        hover_data=['Feature'] + [f'Replicate_{i+1}' for i in range(data.shape[1])],
        title='PCA Scatter Plot with K-Means Clustering (2D)',
        labels={'color': 'Cluster'}
    )
    fig_2d.update_layout(legend_title_text='Cluster', dragmode='pan', hovermode='closest')
    st.plotly_chart(fig_2d, width='stretch')

    fig_3d = None
    if num_pca_components >= 3:
        plot_df[f'PC3 ({explained_variance[2]:.2f}%)'] = pca_result[:, 2]
        fig_3d = px.scatter_3d(
            plot_df,
            x=f'PC1 ({explained_variance[0]:.2f}%)',
            y=f'PC2 ({explained_variance[1]:.2f}%)',
            z=f'PC3 ({explained_variance[2]:.2f}%)',
            color=plot_df['Cluster'].astype(str),
            hover_data=['Feature'] + [f'Replicate_{i+1}' for i in range(data.shape[1])],
            title='PCA Scatter Plot with K-Means Clustering (3D)',
            labels={'color': 'Cluster'}
        )
        fig_3d.update_layout(
            legend_title_text='Cluster',
            scene=dict(
                xaxis_title=f'PC1 ({explained_variance[0]:.2f}%)',
                yaxis_title=f'PC2 ({explained_variance[1]:.2f}%)',
                zaxis_title=f'PC3 ({explained_variance[2]:.2f}%)'
            )
        )
        st.plotly_chart(fig_3d, width='stretch')

    # Excel export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        plot_df.to_excel(writer, index=False, sheet_name='PCA + Clusters')
        pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(num_pca_components)],
            'Explained Variance (%)': explained_variance,
            'Cumulative Variance (%)': cumulative_variance
        }).to_excel(writer, index=False, sheet_name='Explained Variance')
        loadings.to_excel(writer, sheet_name='PCA Loadings')

    st.download_button(
        label="Download PCA + Cluster Data as Excel",
        data=output.getvalue(),
        file_name="pca_cluster_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
