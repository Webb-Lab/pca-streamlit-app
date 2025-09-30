import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

# Set up the Streamlit app
st.title("Interactive PCA Visualization Web App")
st.write("This app visualizes PCA results from an Excel file. The first two rows are treated as descriptive labels and combined for legend entries.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file:
    # Read the Excel file without headers
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Extract labels from the first two rows (columns are samples)
    label_row_1 = df.iloc[0].astype(str).str.strip()  # Peptide
    label_row_2 = df.iloc[1].astype(str).str.strip()  # Experiment
    combined_labels = label_row_1 + ", " + label_row_2

    # Extract numeric data from row 3 onward and transpose
    raw_data = df.iloc[2:].reset_index(drop=True)

    # Convert all values to numeric and fill missing values using row-wise mean
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
    filled_data = raw_data.fillna(raw_data.T.mean(axis=1))

    data = filled_data.transpose()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform PCA to reduce to 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame for plotting with hover tooltips
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'LegendLabel': combined_labels[:len(pca_result)]  # Ensure alignment
    })

    # Add original numeric values as hover data
    for i, col in enumerate(data.columns):
        plot_df[f'Feature_{i+1}'] = data[col]

    # Create interactive PCA scatter plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='LegendLabel',
        hover_data=[f'Feature_{i+1}' for i in range(data.shape[1])],
        title='PCA Scatter Plot',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
    )
    fig.update_layout(
        legend_title_text='Peptide, Experiment',
        dragmode='pan',
        hovermode='closest'
    )

    # Display the plot
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
