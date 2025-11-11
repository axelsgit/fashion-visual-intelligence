import os
import pandas as pd


# Define paths
base_csv = "data/metadata_dior_womenswear.csv"
output_csv = "data/metadata_dior_womenswear_full.csv"

# === LOAD METADATA ===
base_meta = pd.read_csv(base_csv)

rows = []

# === PROCESS EACH FOLDER ===
for _, row in base_meta.iterrows():
    folder = row["folder_path"]
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for fname in image_files:
        img_path = os.path.join(folder, fname)
        rows.append({
            "collection_id": row["collection_id"],
            "year": row["year"],
            "season": row["season"],
            "director": row["director"],
            "collection_name": row["collection_name"],
            "era": row["era"],
            "location": row["location"],
            "category": row["category"],
            "folder_path": folder,
            "image_path": img_path,
            "notes": row["notes"]
        })

# === CREATE AND SAVE FULL METADATA ===
df_full = pd.DataFrame(rows)
df_full.to_csv(output_csv, index=False)

print(f"### Success ###\nSaved {len(df_full)} image entries to {output_csv}")
