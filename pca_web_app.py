import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel("after.xlsx", header=0, index_col=0, engine="openpyxl")

# Drop non-numeric columns and convert to numeric
data = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with all NaNs
data.dropna(how='all', inplace=True)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Determine valid cluster range for silhouette score
num_samples = pca_result.shape[0]
valid_cluster_range = range(2, min(11, num_samples))

# Calculate silhouette scores
silhouette_scores = []
for k in valid_cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_result)
    score = silhouette_score(pca_result, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(valid_cluster_range, silhouette_scores, marker='o', color='green')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_score_plot.png")
