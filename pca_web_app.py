import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

# Set up the Streamlit app
st.title("Interactive PCA Visualization Web App")
st.write("This app visualizes PCA results from an Excel file. The first two rows are treated as descriptive labels and combined for legend entries.")

# Read the Excel file without headers
df = pd.read_excel(excel_file, header=None, engine='openpyxl')

# Extract the first two rows as string labels and clean whitespace
label_row_1 = df.iloc[0].astype(str).str.strip()
label_row_2 = df.iloc[1].astype(str).str.strip()
combined_labels = label_row_1 + ", " + label_row_2

# Extract numeric data from row 3 onward
data = df.iloc[2:].reset_index(drop=True)

# Convert all values to numeric and fill missing values with column mean
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Perform PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Ensure the number of labels matches the number of PCA points
num_points = pca_result.shape[0]
num_labels = len(combined_labels)
repeated_labels = [combined_labels[i % num_labels] for i in range(num_points)]

# Create a DataFrame for plotting with hover tooltips
plot_df = pd.DataFrame({
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'LegendLabel': repeated_labels
})
# Add original numeric values as hover data
for col in data.columns:
    plot_df[f'Original_{col}'] = data[col]

# Create interactive PCA scatter plot
fig = px.scatter(
    plot_df,
    x='PC1',
    y='PC2',
    color='LegendLabel',
    hover_data=[f'Original_{col}' for col in data.columns],
    title='PCA Scatter Plot',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
)

fig.update_layout(
    legend_title_text='Combined Labels (Row 1, Row 2)',
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
