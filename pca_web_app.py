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
st.write("Upload an Excel file: rows are experimental replicates/samples, columns are features (compound, peptide etc). The script expects the feature names in row 1 and the replicate names in column A of the Excel spreadsheet. This script does not include missing value imputation.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file:
    # Load the Excel file with first column as index and first row as header
    df = pd.read_excel(uploaded_file, header=0, index_col=0, engine='openpyxl')

    # Display the raw data
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    # Leave missing values untouched
    data = df.copy()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA component slider
    max_components = min(data.shape[1], 10)
    num_pca_components = st.slider("Select number of PCA components", min_value=2, max_value=max_components, value=2)
    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage

# PCA Loadings
loadings = pd.DataFrame(pca.components_.T, index=data.columns, columns=[f'PC{i+1}' for i in range(num_pca_components)])
st.subheader("PCA Loading Matrix")
st.dataframe(loadings)
    cumulative_variance = explained_variance.cumsum()

    # Scree plot of explained variance
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

    # Determine max clusters based on sample size
    max_possible_clusters = min(10, len(data))

    # Elbow method to help choose clusters
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

    # Silhouette score visualization
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

    # K-Means clustering
    num_clusters = st.slider("Select number of clusters for K-Means", min_value=2, max_value=max_possible_clusters, value=min(3, max_possible_clusters))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    # Create plot DataFrame with swapped labels
    plot_df = pd.DataFrame({
        f'PC1 ({explained_variance[0]:.2f}%)': pca_result[:, 0],
        f'PC2 ({explained_variance[1]:.2f}%)': pca_result[:, 1],
        'Feature': data.index,
        'Cluster': clusters
    })

    # Add original replicates for hover
    for i, col in enumerate(data.columns):
        plot_df[f'Replicate_{i+1}'] = data[col].values

    # Plot with clusters (2D)
    fig_2d = px.scatter(
        plot_df,
        x=f'PC1 ({explained_variance[0]:.2f}%)',
        y=f'PC2 ({explained_variance[1]:.2f}%)',
        color=plot_df['Cluster'].astype(str),
        hover_data=['Feature'] + [f'Replicate_{i+1}' for i in range(data.shape[1])],
        title='PCA Scatter Plot with K-Means Clustering (2D)',
        labels={'color': 'Cluster'}
    )
    fig_2d.update_layout(
        legend_title_text='Cluster',
        dragmode='pan',
        hovermode='closest'
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    # Optional 3D PCA plot if 3 components are selected
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
        st.plotly_chart(fig_3d, use_container_width=True)

    # Save all plots as a single PDF
    if st.button("Download All Plots as PDF"):
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for fig in [fig_scree, fig_elbow, fig_silhouette]:
                pdf.savefig(fig)

            # Save 2D PCA plot
            fig_pca_2d, ax_pca_2d = plt.subplots()
            cluster_ids = sorted(plot_df['Cluster'].unique())
            cmap = plt.get_cmap('tab10')
            cluster_color_map = {cluster: cmap(i % 10) for i, cluster in enumerate(cluster_ids)}
            for cluster in cluster_ids:
                subset = plot_df[plot_df['Cluster'] == cluster]
                ax_pca_2d.scatter(
                    subset[f'PC1 ({explained_variance[0]:.2f}%)'],
                    subset[f'PC2 ({explained_variance[1]:.2f}%)'],
                    label=f'Cluster {cluster}',
                    alpha=0.7,
                    color=cluster_color_map[cluster]
                )
            ax_pca_2d.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
            ax_pca_2d.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
            ax_pca_2d.set_title('PCA Scatter Plot with K-Means Clustering (2D)')
            ax_pca_2d.legend(title='Cluster', fontsize='small', loc='best')
            ax_pca_2d.grid(True)
            pdf.savefig(fig_pca_2d)
            plt.close(fig_pca_2d)

            # Save 3D PCA plot as image if available
            if fig_3d:
            # Save the 3D plot as an HTML file and take a screenshot using selenium or skip it
               html_path = "temp_3d_plot.html"
               with open(html_path, "w") as f:
                   f.write(fig_3d.to_html(full_html=False, include_plotlyjs='cdn'))

               # Instead of screenshotting, just add a placeholder page in the PDF
               fig_pca_3d, ax_pca_3d = plt.subplots()
               ax_pca_3d.text(0.5, 0.5, "3D PCA plot saved separately as HTML", ha='center', va='center', fontsize=12)
               ax_pca_3d.axis('off')
               pdf.savefig(fig_pca_3d)
               plt.close(fig_pca_3d)

               # Optionally offer the HTML download
               with open(html_path, "rb") as f:
                   st.download_button(
                       label="Download 3D PCA Plot (HTML)",
                       data=f.read(),
                       file_name="pca_3d_plot.html",
                       mime="text/html"
                   )


        st.download_button(
            label="Download All Plots as PDF",
            data=pdf_buffer.getvalue(),
            file_name="all_pca_plots.pdf",
            mime="application/pdf"
        )

    # Download PCA + Cluster data
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        plot_df.to_excel(writer, index=False, sheet_name='PCA + Clusters')
        pd.DataFrame({
    loadings.to_excel(writer, sheet_name='PCA Loadings')
            'Principal Component': [f'PC{i+1}' for i in range(num_pca_components)],
            'Explained Variance (%)': explained_variance,
            'Cumulative Variance (%)': cumulative_variance
        }).to_excel(writer, index=False, sheet_name='Explained Variance')
    st.download_button(
        label="Download PCA + Cluster Data as Excel",
        data=output.getvalue(),
        file_name="pca_cluster_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
