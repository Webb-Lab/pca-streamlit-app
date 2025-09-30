import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

# Set up the app
st.title("Interactive PCA Visualization Web App")
st.write("Upload an Excel file where each pair of columns represents an array. The first two rows in each column are classifiers, and the remaining rows are numeric data.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the Excel file without headers
    df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

    # Determine number of columns
    num_cols = df.shape[1]

    # Prepare list to hold combined arrays and labels
    combined_arrays = []
    combined_labels = []

    # Process columns in pairs
    for i in range(0, num_cols, 2):
        col1 = df.iloc[:, i]
        col2 = df.iloc[:, i + 1]

        # Create label from first two rows of each column
        label1 = f"{col1.iloc[0]} + {col1.iloc[1]}"
        label2 = f"{col2.iloc[0]} + {col2.iloc[1]}"
        combined_label = f"{label1}, {label2}"
        combined_labels.append(combined_label)

        # Combine numeric data from both columns (excluding first two rows)
        values1 = col1.iloc[2:].astype(float).values
        values2 = col2.iloc[2:].astype(float).values
        combined_array = list(values1) + list(values2)
        combined_arrays.append(combined_array)

    # Create DataFrame from combined arrays
    data_df = pd.DataFrame(combined_arrays)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'LegendLabel': combined_labels
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
        legend_title_text='Combined Labels (Classifier 1 + Classifier 2)',
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
