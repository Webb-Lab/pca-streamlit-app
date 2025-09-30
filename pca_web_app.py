import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up the app
st.title("PCA Visualization Web App")
st.write("Upload an Excel file where the first two rows are labels and the rest are numeric data.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    color_labels = df.iloc[0]
    shape_labels = df.iloc[1]
    data = df.iloc[2:].reset_index(drop=True)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Define color and marker maps
    unique_colors = sorted(color_labels.unique())
    unique_shapes = sorted(shape_labels.unique())
    color_map = {label: plt.cm.tab10(i % 10) for i, label in enumerate(unique_colors)}
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    marker_map = {label: marker_styles[i % len(marker_styles)] for i, label in enumerate(unique_shapes)}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = set()

    for i in range(len(pca_result)):
        color_val = color_labels[i]
        shape_val = shape_labels[i]
        color = color_map[color_val]
        marker = marker_map[shape_val]
        label = f"{color_val} | {shape_val}"
        if label not in plotted:
            ax.scatter(pca_result[i, 0], pca_result[i, 1], c=[color], marker=marker, label=label, edgecolors='black')
            plotted.add(label)
        else:
            ax.scatter(pca_result[i, 0], pca_result[i, 1], c=[color], marker=marker, edgecolors='black')

    # Custom legend
    handles = []
    for color_val in unique_colors:
        for shape_val in unique_shapes:
            if ((color_labels == color_val) & (shape_labels == shape_val)).any():
                color = color_map[color_val]
                marker = marker_map[shape_val]
                label = f"{color_val} | {shape_val}"
                handles.append(plt.Line2D([0], [0], marker=marker, color='w', label=label,
                                          markerfacecolor=color, markeredgecolor='black', markersize=10))

    ax.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("PCA Scatter Plot")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True)
    st.pyplot(fig)
