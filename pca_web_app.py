import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io

st.title("PCA Visualization from before.xlsx")
st.write("Upload an Excel file with peptide labels in row 0, sample types in row 1, and intensity data in rows 2–4.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Extract metadata and data
    peptide_labels = df.iloc[0].tolist()
    sample_types = df.iloc[1].tolist()
    replicate_data = df.iloc[2:5].reset_index(drop=True).transpose()

    # Generate sample names
    replicate_counts = {}
    sample_names = []
    for sample_type in sample_types:
        replicate_counts[sample_type] = replicate_counts.get(sample_type, 0) + 1
        sample_names.append(f"{sample_type}{replicate_counts[sample_type]}")

    replicate_data.index = sample_names
    replicate_data.columns = [f"Peptide_{i+1}" for i in range(replicate_data.shape[1])]
    replicate_data = replicate_data.apply(pd.to_numeric, errors='coerce')

    # Impute missing values
    for sample_type in set(sample_types):
        indices = [i for i, name in enumerate(sample_names) if name.startswith(sample_type)]
        subset = replicate_data.iloc[indices]
        mean_vals = subset.mean(axis=0)
        replicate_data.iloc[indices] = subset.fillna(mean_vals)
    replicate_data = replicate_data.fillna(0)

    st.subheader("Transformed Intensity Matrix")
    st.dataframe(replicate_data)

    # Standardize and PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(replicate_data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_ * 100

    # Elbow method
    st.subheader("Elbow Method for Optimal Clusters")
    inertia_values = []
    cluster_range = range(1, 11)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_result)
        inertia_values.append(kmeans.inertia_)

    fig_elbow, ax = plt.subplots()
    ax.plot(cluster_range, inertia_values, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.grid(True)
    st.pyplot(fig_elbow)

    # Silhouette scores
    st.subheader("Silhouette Scores for Cluster Counts")
    silhouette_scores = []
    silhouette_range = range(2, 11)
    for k in silhouette_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pca_result)
        score = silhouette_score(pca_result, labels)
        silhouette_scores.append(score)

    fig_silhouette, ax2 = plt.subplots()
    ax2.plot(silhouette_range, silhouette_scores, marker='o', color='green')
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs Cluster Count")
    ax2.grid(True)
    st.pyplot(fig_silhouette)

    # Final clustering
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    plot_df = pd.DataFrame({
        f'PC1 ({explained_variance[0]:.2f}%)': pca_result[:, 0],
        f'PC2 ({explained_variance[1]:.2f}%)': pca_result[:, 1],
        'Sample': replicate_data.index,
        'Cluster': clusters,
        'Experiment': [name[:-1] for name in replicate_data.index]
    })

    fig = px.scatter(
        plot_df,
        x=f'PC1 ({explained_variance[0]:.2f}%)',
        y=f'PC2 ({explained_variance[1]:.2f}%)',
        color=plot_df['Cluster'].astype(str),
        symbol='Experiment',
        hover_data=['Sample'],
        title='PCA Scatter Plot with K-Means Clustering'
    )
    fig.update_layout(legend_title_text='Cluster and Experiment')
    st.plotly_chart(fig, use_container_width=True)

    # PDF download
    if st.button("Download PCA Plot as PDF"):
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            plt.figure(figsize=(8, 6))
            cmap = plt.get_cmap('tab10')
            cluster_colors = {c: cmap(i % 10) for i, c in enumerate(sorted(plot_df['Cluster'].unique()))}
            for cluster in sorted(plot_df['Cluster'].unique()):
                subset = plot_df[plot_df['Cluster'] == cluster]
                plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f'Cluster {cluster}', color=cluster_colors[cluster])
            for i, name in enumerate(plot_df['Sample']):
                plt.annotate(name, (pca_result[i, 0], pca_result[i, 1]))
            plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
            plt.title('PCA Plot with KMeans Clustering')
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()

        st.download_button(
            label="Download PCA Plot as PDF",
            data=pdf_buffer.getvalue(),
            file_name="pca_kmeans_plot.pdf",
            mime="application/pdf"
        )

    # Excel export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        replicate_data.to_excel(writer, sheet_name='Transformed Matrix')
        plot_df.to_excel(writer, index=False, sheet_name='PCA + Clusters')
        pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(2)],
            'Explained Variance (%)': explained_variance
        }).to_excel(writer, index=False, sheet_name='Explained Variance')
    st.download_button(
        label="Download Transformed Data as Excel",
        data=output.getvalue(),
        file_name="transformed_pca_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
