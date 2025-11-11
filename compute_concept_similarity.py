import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

# Configuration
meta_path = "data/metadata_dior_womenswear_full.csv"
output_path = "data/dior_concept_scores.csv"

print("🔹 Running CLIP concept similarity...")

# ====================================
# Load CLIP model
# ====================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ====================================
# Load metadata
# ====================================
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"❌ Metadata file not found: {meta_path}")

meta = pd.read_csv(meta_path)
print(f"Loaded {len(meta)} image entries from {meta_path}")

# ====================================
# Define style concepts
# ====================================
concepts = ["romantic", "minimalist", "avant-garde", "feminine", "futuristic"]

# Encode text concepts once
text_tokens = clip.tokenize(concepts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ====================================
# Compute image similarity for each concept
# ====================================
results = []

for i, row in tqdm(meta.iterrows(), total=len(meta), desc=f"Processing Dior images"):
    path = row["image_path"]
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity with each concept
        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

        result = {
            "image_path": path,
            "collection_id": row["collection_id"],
            "year": row["year"],
            "season": row["season"],
            "era": row["era"],
            "director": row.get("director", None),
        }
        for j, c in enumerate(concepts):
            result[c] = sims[j]

        results.append(result)
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

# ====================================
# Save to CSV
# ====================================
df_results = pd.DataFrame(results)
df_results.to_csv(output_path, index=False)

print(f"### Success ###\nSaved {len(df_results)} concept similarity scores → {output_path}")
