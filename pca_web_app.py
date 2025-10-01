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

# Set up the Streamlit app
st.title("Transformed PCA Visualization with K-Means Clustering")
st.write("Upload an Excel file with raw data. The app will transform the matrix and visualize PCA clusters.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Step 1: Impute missing values using mean of two other points in the same row
    def impute_row(row):
        row = row.copy()
        for i in range(len(row)):
            if pd.isna(row[i]):
                non_nan = row[~pd.isna(row)]
                if len(non_nan) >= 2:
                    row[i] = non_nan.iloc[:2].mean()
                else:
                    row[i] = 0
        return row

    df.iloc[2:] = df.iloc[2:].apply(impute_row, axis=1)

    # Step 2: Transform matrix
    row1 = df.iloc[0].tolist()
    row2 = df.iloc[1].tolist()
    data = df.iloc[2:].reset_index(drop=True)

    new_columns = []
    new_data = []

    unique_row2 = pd.unique(row2)
    for label in unique_row2:
        indices = [i for i, x in enumerate(row2) if x == label]
        for j, idx in enumerate(indices):
            new_columns.append(f"{label}{j+1}")
            new_row = [row1[idx]] + data.iloc[:, idx].tolist()
            new_data.append(new_row)

    transformed_df = pd.DataFrame(new_data, columns=["Label"] + [f"Rep{i+1}" for i in range(data.shape[0])])

    # Step 3: Display transformed matrix
    st.subheader("Transformed Data Matrix")
    st.dataframe(transformed_df)

    # Step 4: PCA and clustering
    features = transformed_df.drop(columns=["Label"])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    num_pca_components = st.slider("Select number of PCA components", min_value=2, max_value=min(10, features.shape[1]), value=2)
    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_ * 100

    # Elbow method
    st.subheader("Elbow Method for Optimal Clusters")
    inertia_values = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_result)
        inertia_values.append(kmeans.inertia_)

    fig_elbow, ax = plt.subplots()
    ax.plot(range(1, 11), inertia_values, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    # Silhouette scores
    st.subheader("Silhouette Scores")
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pca_result)
        score = silhouette_score(pca_result, labels)
        silhouette_scores.append(score)

    fig_silhouette, ax2 = plt.subplots()
    ax2.plot(range(2, 11), silhouette_scores, marker='o', color='green')
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs Cluster Count")
    st.pyplot(fig_silhouette)

    # Final clustering
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    plot_df = pd.DataFrame({
        f'PC1 ({explained_variance[0]:.2f}%)': pca_result[:, 0],
        f'PC2 ({explained_variance[1]:.2f}%)': pca_result[:, 1],
        'Cluster': clusters,
        'Label': transformed_df['Label']
    })

    fig = px.scatter(
        plot_df,
        x=f'PC1 ({explained_variance[0]:.2f}%)',
        y=f'PC2 ({explained_variance[1]:.2f}%)',
        color=plot_df['Cluster'].astype(str),
        symbol='Label',
        hover_data=['Label'],
        title='PCA Scatter Plot with K-Means Clustering',
        labels={'color': 'Cluster', 'symbol': 'Label'}
    )
    fig.update_layout(legend_title_text='Cluster and Label')
    st.plotly_chart(fig, use_container_width=True)

    # PDF export
    if st.button("Download PCA Plot as PDF"):
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            plt.figure(figsize=(8, 6))
            marker_styles = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
            labels = plot_df['Label'].unique()
            label_marker_map = {label: marker_styles[i % len(marker_styles)] for i, label in enumerate(labels)}
            cluster_ids = sorted(plot_df['Cluster'].unique())
            cmap = plt.get_cmap('tab10')
            cluster_color_map = {cluster: cmap(i % 10) for i, cluster in enumerate(cluster_ids)}

            for cluster in cluster_ids:
                for label in labels:
                    subset = plot_df[(plot_df['Cluster'] == cluster) & (plot_df['Label'] == label)]
                    plt.scatter(
                        subset[f'PC1 ({explained_variance[0]:.2f}%)'],
                        subset[f'PC2 ({explained_variance[1]:.2f}%)'],
                        label=f'{label}, Cluster {cluster}',
                        alpha=0.7,
                        marker=label_marker_map[label],
                        color=cluster_color_map[cluster]
                    )

            plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
            plt.title('PCA Scatter Plot with K-Means Clustering')
            plt.legend(title='Cluster and Label', fontsize='small', loc='best')
            plt.grid(True)
            pdf.savefig()
            plt.close()

        st.download_button(
            label="Download PCA Plot as PDF",
            data=pdf_buffer.getvalue(),
            file_name="transformed_pca_plot.pdf",
            mime="application/pdf"
        )
