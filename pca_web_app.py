import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

# Set up the app
st.title("Interactive PCA Visualization Web App")
st.write("Upload an Excel file where the first two rows are descriptive labels and the rest are numeric data.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the Excel file without headers
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Extract descriptive labels from first two rows and combine them
    label_row_1 = df.iloc[0].astype(str)
    label_row_2 = df.iloc[1].astype(str)
    combined_labels = label_row_1 + ", " + label_row_2

    # Extract numeric data starting from the third row
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
        'LegendLabel': combined_labels.values
    })

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='LegendLabel',
        title='PCA Scatter Plot',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    )

    fig.update_layout(
        legend_title_text='Combined Labels (Row 1, Row 2)',
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
        plot_df.to_excel(writer, index=False, sheet_name='PCA Plot Data')
    st.download_button(
        label="Download PCA Plot Data as Excel",
        data=output.getvalue(),
        file_name="pca_plot_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
