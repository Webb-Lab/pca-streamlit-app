import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io

# Set up the Streamlit app
st.title("Interactive PCA Visualization with K-Means Clustering")
st.write("Upload an Excel file: first row is the name (e.g., compound name, peptide sequence), second row is the experiment (e.g., control vs treated), rest are dependent variables (e.g., intensity for each replicate).")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Extract labels
    label_row_1 = df.iloc[0].astype(str).str.strip()
    label_row_2 = df.iloc[1].astype(str).str.strip()
    combined_labels = label_row_1 + ", " + label_row_2

    # Extract and clean numeric data
    raw_data = df.iloc[2:].reset_index(drop=True)
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
    filled_data = raw_data.fillna(raw_data.T.mean(axis=1))
    data = filled_data.transpose()

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA component slider
    max_components = min(data.shape[1], 10)
    num_pca_components = st.slider("Select number of PCA components", min_value=2, max_value=max_components, value=2)
    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage

    # Elbow method to help choose clusters
    st.subheader("Elbow Method to Help Choose Optimal Number of Clusters")
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
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.grid(True)
    st.pyplot(fig_elbow)

    # Silhouette score visualization
    st.subheader("Silhouette Score for Cluster Counts (2 to 10)")
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
    ax2.set_title("Silhouette Score vs Number of Clusters")
    ax2.grid(True)
    st.pyplot(fig_silhouette)

    # K-Means clustering
    num_clusters = st.slider("Select number of clusters for K-Means", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    # Create plot DataFrame
    plot_df = pd.DataFrame({
        f'PC1 ({explained_variance[0]:.2f}%)': pca_result[:, 0],
        f'PC2 ({explained_variance[1]:.2f}%)': pca_result[:, 1],
        'Label 1': label_row_1[:len(pca_result)],
        'Experiment': label_row_2[:len(pca_result)],
        'LegendLabel': combined_labels[:len(pca_result)],
        'Cluster': clusters
    })

    # Add original features for hover
    for i, col in enumerate(data.columns):
        plot_df[f'Feature_{i+1}'] = data[col]

    # Plot with clusters and shapes
    fig = px.scatter(
        plot_df,
        x=f'PC1 ({explained_variance[0]:.2f}%)',
        y=f'PC2 ({explained_variance[1]:.2f}%)',
        color=plot_df['Cluster'].astype(str),
        symbol='Experiment',
        hover_data=['LegendLabel'] + [f'Feature_{i+1}' for i in range(data.shape[1])],
        title='PCA Scatter Plot with K-Means Clustering',
        labels={'color': 'Cluster', 'symbol': 'Experiment'}
    )
    fig.update_layout(
        legend_title_text='Cluster and Experiment',
        dragmode='pan',
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Save plot as PDF using matplotlib
    if st.button("Save Plot as PDF"):
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            plt.figure(figsize=(8, 6))
            for cluster in sorted(plot_df['Cluster'].unique()):
                subset = plot_df[plot_df['Cluster'] == cluster]
                plt.scatter(
                    subset[f'PC1 ({explained_variance[0]:.2f}%)'],
                    subset[f'PC2 ({explained_variance[1]:.2f}%)'],
                    label=f'Cluster {cluster}',
                    alpha=0.7
                )
            plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
            plt.title('PCA Scatter Plot with K-Means Clustering')
            plt.legend(title='Cluster')
            plt.grid(True)
            pdf.savefig()
            plt.close()
        st.download_button(
            label="Download PCA Plot as PDF",
            data=pdf_buffer.getvalue(),
            file_name="pca_kmeans_plot.pdf",
            mime="application/pdf"
        )

    # Download PCA + Cluster data
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        plot_df.to_excel(writer, index=False, sheet_name='PCA + Clusters')
        pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(num_pca_components)],
            'Explained Variance (%)': explained_variance
        }).to_excel(writer, index=False, sheet_name='Explained Variance')
    st.download_button(
        label="Download PCA + Cluster Data as Excel",
        data=output.getvalue(),
        file_name="pca_cluster_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
