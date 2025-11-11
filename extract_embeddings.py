from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Define input and output file paths
meta_path = "data/metadata_dior_womenswear_full.csv"
emb_out = "data/embeddings_clip_dior.npy"
meta_out = "data/embeddings_metadata_dior.csv"

# ======================================
# LOAD MODEL
# ======================================
print("🔹 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ======================================
# LOAD METADATA
# ======================================
print(f"Reading metadata from {meta_path}...")
meta = pd.read_csv(meta_path)

# ======================================
# COMPUTE EMBEDDINGS
# ======================================
embeddings = []

for row in tqdm(meta.itertuples(), total=len(meta), desc=f"Processing Dior images"):
    image_path = row.image_path
    if not os.path.exists(image_path):
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        features = features / features.norm(p=2)  # normalize
        embeddings.append({
            "collection_id": row.collection_id,
            "image_path": image_path,
            "embedding": features[0].cpu().numpy()
        })
    except Exception as e:
        print(f"Skipped {image_path}: {e}")

# ======================================
# SAVE RESULTS
# ======================================
if embeddings:
    df = pd.DataFrame(embeddings)
    np.save(emb_out, np.stack(df["embedding"].values))
    df.drop(columns=["embedding"]).to_csv(meta_out, index=False)
    print(f"### Success ###\nSaved {len(df)} embeddings → {emb_out}")
else:
    print(f"❌ No embeddings created for Dior - check your image paths.")
