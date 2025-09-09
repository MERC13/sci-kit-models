import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_breast_cancer

# Data prep
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

if 'target' in df.columns:
    y = df['target']
    X = df.drop(columns=['target'])
else:
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    df = X.copy()
    df['target'] = y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['cluster'] = labels

centers_scaled = kmeans.cluster_centers_
centers_unscaled = pd.DataFrame(
    scaler.inverse_transform(centers_scaled),
    columns=X.columns
)
print("Cluster Centers (original units):\n", centers_unscaled.round(2))

# Eval
print("\nCluster Counts:")
print(pd.Series(labels).value_counts().sort_index())

sil = silhouette_score(X_scaled, labels)
print(f"\nSilhouette score: {sil:.3f}")

print("\nContingency table (true label vs cluster):")
print(pd.crosstab(df['target'], df['cluster'], rownames=['true'], colnames=['cluster']))

# Visualize
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
vis['cluster'] = labels
if set(pd.unique(y)) == {0, 1}:
    vis['target'] = y.replace({0: 'malignant', 1: 'benign'})
else:
    vis['target'] = y

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=vis, x='PC1', y='PC2', hue='cluster', palette='Set1', s=40, alpha=0.85)
plt.title('K-Means clusters (PCA space)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
sns.scatterplot(data=vis, x='PC1', y='PC2', hue='target', palette='Dark2', s=40, alpha=0.85)
plt.title('True labels (PCA space)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()
