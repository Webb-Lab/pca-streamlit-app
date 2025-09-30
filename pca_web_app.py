import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

# Set up the app
st.title("Interactive PCA Visualization Web App")
st.write("Upload an Excel file where the first two rows are labels and the rest are numeric data.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the Excel file without headers
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Extract label rows
    color_labels_raw = df.iloc[0]
    shape_labels_raw = df.iloc[1]
    data_raw = df.iloc[2:].reset_index(drop=True)

    # Map unique string labels to numeric values for PCA processing
    color_map = {label: idx for idx, label in enumerate(pd.unique(color_labels_raw))}
    shape_map = {label: idx for idx, label in enumerate(pd.unique(shape_labels_raw))}

    # Create numeric label arrays for PCA input
    color_numeric = color_labels_raw.map(color_map)
    shape_numeric = shape_labels_raw.map(shape_map)

    # Convert data to numeric and fill missing values if any
    data_numeric = data_raw.apply(pd.to_numeric, errors='coerce')

    # Combine numeric labels with data for PCA
    data_combined = pd.concat([color_numeric.to_frame().T, shape_numeric.to_frame().T, data_numeric], ignore_index=True)

    # Drop rows with NaN values
    data_clean = data_combined.dropna()

    # Check if there's enough data for PCA
    if data_clean.shape[0] < 2 or data_clean.shape[1] < 2:
        st.error("Not enough valid numeric data to perform PCA. Please check your file and try again.")
    else:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_clean)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Color': color_labels_raw.values[:len(pca_result)],
            'Shape': shape_labels_raw.values[:len(pca_result)]
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

        # Provide download link for PCA plot data as Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            plot_df.drop(columns='Symbol').to_excel(writer, index=False, sheet_name='PCA Plot Data')
        st.download_button(
            label="Download PCA Plot Data as Excel",
            data=output.getvalue(),
            file_name="pca_plot_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
