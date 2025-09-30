import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set up the app
st.title("Interactive PCA Visualization Web App")
st.write("Upload an Excel file where the first two rows are labels and the rest are numeric data.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Extract labels and data
    color_labels = df.iloc[0]
    shape_labels = df.iloc[1]
    data = df.iloc[2:].reset_index(drop=True)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Color': color_labels.values,
        'Shape': shape_labels.values
    })

    # Map shapes to Plotly symbols
    unique_shapes = sorted(plot_df['Shape'].unique())
    symbol_map = {val: symbol for val, symbol in zip(unique_shapes, ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'x'])}
    plot_df['Symbol'] = plot_df['Shape'].map(symbol_map)

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Color',
        symbol='Symbol',
        symbol_sequence=list(symbol_map.values()),
        title='PCA Scatter Plot',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    )

    fig.update_layout(
        legend_title_text='Color | Shape',
        dragmode='pan',
        hovermode='closest',
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Save plot as PNG
    if st.button("Save Plot as PNG"):
        fig.write_image("pca_plot.png")
        st.success("Plot saved as pca_plot.png")

