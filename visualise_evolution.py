import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os

# Configuration
emb_path = "data/embeddings_clip_dior.npy"
emb_meta_path = "data/embeddings_metadata_dior.csv"
meta_base_path = "data/metadata_dior_womenswear.csv"
out_path = "data/dior_tsne_results.csv"


print("🔹 Running t-SNE pipeline...")

# ================================================
# Load data
# ================================================
if not os.path.exists(emb_path):
    raise FileNotFoundError(f"❌ Embedding file not found: {emb_path}")

X = np.load(emb_path)
meta_embed = pd.read_csv(emb_meta_path)
meta_base = pd.read_csv(meta_base_path)

# Merge metadata on collection_id
meta = meta_embed.merge(meta_base, on="collection_id", how="left")

print(f"Loaded {len(X)} embeddings across {meta['era'].nunique()} eras for Dior.")
print(meta.groupby('era')['collection_id'].nunique())

# ================================================
# Dimensionality Reduction
# ================================================
print("Performing PCA (50D) → t-SNE (2D)...")
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=2000)
X_tsne = tsne.fit_transform(X_pca)

meta["tsne_x"] = X_tsne[:, 0]
meta["tsne_y"] = X_tsne[:, 1]

# ================================================
# 2D Visualization — by Era
# ================================================
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=meta,
    x="tsne_x", y="tsne_y",
    hue="era", palette="deep", alpha=0.7
)
plt.title("Dior Runway Style Evolution (CLIP → t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Creative Era", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ================================================
# Chronological Visualization — by Year
# ================================================
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=meta,
    x="tsne_x", y="tsne_y",
    hue="year", palette="viridis", alpha=0.7
)
plt.title(f"Dior - Aesthetic Drift Over Time")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar = plt.colorbar
plt.tight_layout()
plt.show()

# ================================================
# 3D PCA Visualization
# ================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    c=meta["year"], cmap="plasma", s=15
)
ax.set_title(f"PCA 3D Projection - Dior Collections Over Time")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()

# ================================================
# Save Results
# ================================================
meta.to_csv(out_path, index=False)
print(f"### Success ###\nSaved Dior t-SNE coordinates → {out_path}")
