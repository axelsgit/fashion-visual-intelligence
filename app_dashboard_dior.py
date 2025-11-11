import streamlit as st
import pandas as pd
import plotly.express as px
import os
import random
from PIL import Image
import torch
import clip
import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# streamlit run ai_dior_venv/app_dashboard_dior.py  

# ================================================
# Page setup
# ================================================
st.set_page_config(
    page_title="Dior Style Evolution",
    page_icon="data/dior-logo.jpg",
    layout="wide"
)

st.title("Dior Style Evolution Map")
st.markdown("""
Explore the **visual DNA of Dior** across creative eras using AI embeddings (CLIP + t-SNE).  
Each dot represents a **runway look** positioned by aesthetic similarity.
Use the filters on the left to focus on specific eras or years.
""")

# ================================================
# Load data
# ================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/dior_tsne_results.csv")
    return df

df = load_data()

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()
    return model, preprocess

model, preprocess = load_clip_model()

@st.cache_data
def load_clip_embeddings():
    """Load precomputed CLIP image embeddings (512-dim)"""
    emb = np.load("data/embeddings_clip_dior.npy")  
    df_meta = pd.read_csv("data/metadata_dior_womenswear_full.csv")
    emb = normalize(emb)  # normalize for cosine similarity
    return emb, df_meta

clip_embeddings, df_meta = load_clip_embeddings()

# Extract available filters
eras = sorted(df["era"].dropna().unique())
years = sorted(df["year"].dropna().unique())

# ================================================
# Sidebar filters
# ================================================
st.sidebar.header("Filters")
selected_eras = st.sidebar.multiselect("Creative Era", eras, default=eras)
selected_years = st.sidebar.slider("Year Range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))

filtered = df[(df["era"].isin(selected_eras)) & (df["year"].between(*selected_years))]

# ================================================
# 🧭 Dior Aesthetic Landscape (Combined Map + Clusters)
# ================================================
st.markdown("## 🧭 Dior Aesthetic Landscape")

st.markdown("""
This visualization shows how **AI perceives Dior's style evolution** over time.  
Each point represents a **runway look**, placed based on visual similarity using **CLIP** (an AI model trained to understand image-text relationships) and **t-SNE** dimensionality reduction.

- **When colored by era**, you can see the *creative transitions* between Galliano, Simons, and Chiuri.  
- **When colored by AI clusters**, the model groups looks by *visual resemblance*, independent of the designer.  
You can switch between these two perspectives below.
""")

# Toggle color mode
color_mode = st.radio(
    "Color the map by:",
    ["Creative Era", "AI-Detected Cluster"],
    horizontal=True
)

@st.cache_data
def compute_style_clusters(df, n_clusters=6):
    features = df[["tsne_x", "tsne_y"]].values
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(features)
    return df

df = compute_style_clusters(df)

# Select color variable
if color_mode == "Creative Era":
    color_col = "era"
    title = "Dior Aesthetic Map (Colored by Creative Era)"
    color_seq = px.colors.qualitative.Vivid
else:
    color_col = "cluster"
    title = "AI-Detected Style Clusters (K-Means on Embeddings)"
    color_seq = px.colors.qualitative.Safe

# Filtered data (already defined earlier)
filtered = df[(df["era"].isin(selected_eras)) & (df["year"].between(*selected_years))]

# Plot unified map
fig = px.scatter(
    filtered,
    x="tsne_x", y="tsne_y",
    color=color_col,
    hover_data=["year", "season", "collection_id", "era"],
    color_discrete_sequence=color_seq,
    title=title,
    width=950, height=700
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**How to interpret:**  
Clusters that appear close together correspond to collections sharing a similar *visual DNA* (silhouette, color palette, texture language, ...)
AI is not told what “romantic” or “futuristic” means here. It purely groups based on *visual resonance*.
""")


# Add vertical spacing
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

# ================================================
# Image preview on click
# ================================================

st.markdown("### 👗 Look Preview")

st.markdown("""
Available collections by era:

**Maria Grazia Chiuri (2016-2024)**:
- FW2023, FW2022, SS2021, FW2019, FW2018, SS2017

**Raf Simons (2012-2015)**:
- FW2015, FW2014, SS2013, FW2012

**John Galliano (1998-2011)**:
- FW2010, SS2007, FW2005, SS2002, FW1998
""")

clicked = st.text_input("Enter Collection to preview a few random looks (e.g. FW2019):", "")

df_original_metadata = pd.read_csv("data/metadata_dior_womenswear.csv")

# Map collection name to collection_id
if clicked:
    try:
        # Create the collection identifier (e.g., "FW2019_Chiuri")
        collection_query = clicked.upper()  # Convert to uppercase to match folder names
        matching_collections = df_original_metadata[
            df_original_metadata['folder_path'].str.contains(collection_query, case=False)
        ]
        
        if not matching_collections.empty:
            collection_id = matching_collections['collection_id'].iloc[0]
        else:
            st.error(f"Collection '{clicked}' not found. Please enter a valid collection (e.g., FW2019)")
            st.stop()
    except Exception as e:
        st.error(f"Error finding collection: {e}")
        st.stop()

def resize_image_for_display(image, max_size=(300, 450)):
    """Resize image while maintaining aspect ratio."""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

if clicked and 'collection_id' in locals():
    folder = df.loc[df["collection_id"] == collection_id, "folder_path"].iloc[0]
    st.write(f"Showing images from: `{folder}`")

    # List all image paths
    image_paths = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Randomly select 12 images to show
    image_paths = random.sample(image_paths, min(12, len(image_paths)))

    # Create grid layout (3 columns)
    num_cols = 3
    cols = st.columns(num_cols)

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            img = resize_image_for_display(img)
            with cols[i % num_cols]:
                st.image(img, caption=os.path.basename(img_path), use_container_width=False)
        except Exception as e:
            st.write(f"Could not open image {img_path}: {e}")


# Add vertical spacing
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

# ================================================
# 6. Color Palette Evolution by Collection
# ================================================
import numpy as np
from sklearn.cluster import KMeans

st.markdown("### 🎨 Color Palette Evolution")

st.markdown("""
Each point represents one **dominant color** extracted from a collection.  
""")

def extract_palette(image_folder, n_colors=5, sample_size=500):
    """Extract dominant colors using KMeans."""
    pixels = []
    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            img = Image.open(os.path.join(image_folder, img_name)).convert("RGB")
            img = img.resize((100, 100))  # small for speed
            arr = np.array(img).reshape(-1, 3)
            pixels.append(arr)
        except Exception:
            continue
    if not pixels:
        return []
    pixels = np.vstack(pixels)
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = np.clip(kmeans.cluster_centers_.astype(int), 0, 255)
    return [tuple(map(int, c)) for c in colors]

# Compute palettes per collection (cache to avoid recomputing)
@st.cache_data
def compute_palettes(metadata_df):
    palette_data = []
    for _, row in metadata_df.iterrows():
        folder = row["folder_path"]
        if os.path.exists(folder):
            colors = extract_palette(folder)
            for color in colors:
                palette_data.append({
                    "collection_id": row["collection_id"],
                    "era": row["era"],
                    "year": row["year"],
                    "season": row["season"],
                    "color_hex": f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                })
    return pd.DataFrame(palette_data)

palette_df = compute_palettes(df_original_metadata)

# Visualize palette timeline
palette_df_sorted = palette_df.sort_values(by=["year", "season"])
fig_palette = px.scatter(
    palette_df_sorted,
    x="year",
    y="era",
    color="color_hex",
    color_discrete_sequence=palette_df_sorted["color_hex"].tolist(),
    size=[8]*len(palette_df_sorted),
    title="Color Palette Evolution of Dior (Dominant Runway Tones)",
)
fig_palette.update_traces(marker=dict(symbol="square"))
st.plotly_chart(fig_palette, use_container_width=True)


# Add vertical spacing
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")


# ================================================
# 💬 Concept Similarity Explorer (CLIP-based)
# ================================================
st.markdown("## 💬 Concept Similarity Explorer")

st.markdown("""
Explore how AI understands **aesthetic concepts** like “romantic”, “minimalist”, or “avant-garde” 
in Dior’s collections.
The model compares the runway image with the **text description** of that concept,  
highlighting which collections most embody each aesthetic.
""")

concepts = ["romantic", "minimalist", "avant-garde", "military", "futuristic", "baroque", "ethereal", "structured"]
selected_concept = st.selectbox("Choose a fashion concept:", concepts)

df_emb = pd.DataFrame(clip_embeddings)
df_emb["image_path"] = df_meta["image_path"]
df_emb = df_emb.dropna(subset=["image_path"])
df_meta = df_meta[df_meta["image_path"].isin(df_emb["image_path"])].reset_index(drop=True)
clip_embeddings = df_emb.drop(columns=["image_path"]).to_numpy()

def compute_text_similarity(concept, model, image_embs):
    """Compute cosine similarity between text concept and all image embeddings"""
    with torch.no_grad():
        text_tokens = clip.tokenize([concept])
        text_emb = model.encode_text(text_tokens).cpu().numpy()
        text_emb = normalize(text_emb)
        sims = np.dot(image_embs, text_emb.T).squeeze()
    return sims

if selected_concept:
    sims = compute_text_similarity(selected_concept, model, clip_embeddings)

    # Get top 10 most similar looks
    top_idx = np.argsort(sims)[::-1][:12]
    top_rows = df_meta.iloc[top_idx]

    st.markdown(f"### 🔍 Top looks matching **{selected_concept}** aesthetics")

    # Display top results in grid
    num_cols = 3
    cols = st.columns(num_cols)
    for i, (_, row) in enumerate(top_rows.iterrows()):
        img_path = row["image_path"] if "image_path" in row else row["folder_path"]
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((300, 400))
                with cols[i % num_cols]:
                    st.image(img, caption=f"{row['season']} {row['year']} ({row['director']})", use_container_width=False)
            except Exception as e:
                st.write(f"Could not load {img_path}: {e}")

st.write("")
st.markdown("### 📈 Aesthetic Language Trends")

df_concepts = pd.read_csv("data/dior_concept_scores.csv")

# Average by year (or by era)
yearly = df_concepts.groupby("year")[["romantic", "minimalist", "avant-garde", "feminine", "futuristic"]].mean().reset_index()

st.markdown("""
This timeline shows how Dior’s **stylistic alignment** with key concepts evolves over time.  
Peaks and dips in each line reflect the brand’s changing priorities.
The shaded backgrounds indicate the tenure of each creative director.
""")

director_spans = [
    {"director": "John Galliano (1998-2011)", "start": 1998, "end": 2011, "color": "rgba(255,200,200,0.8)"},
    {"director": "Raf Simons (2012-2015)", "start": 2012, "end": 2015, "color": "rgba(200,255,200,0.8)"},
    {"director": "Maria Grazia Chiuri (2016-2024)", "start": 2016, "end": 2024, "color": "rgba(200,200,255,0.8)"}
]

fig = px.line(
    yearly,
    x="year",
    y=["romantic", "minimalist", "avant-garde", "feminine", "futuristic"],
    labels={"value": "Similarity", "year": "Year"},
    title="Dior Aesthetic Language Over Time"
)

for span in director_spans:
    fig.add_vrect(
        x0=span["start"],
        x1=span["end"],
        fillcolor=span["color"],
        opacity=0.8,
        layer="below",
        line_width=0,
        annotation_text=span["director"],
        annotation_position="top left",
        annotation=dict(font_size=12, font_color="black")
    )

st.plotly_chart(fig, use_container_width=True)


# ================================================
# Footer
# ================================================
st.write("")
st.markdown("---")
st.markdown("""
*Created by Axel Heussner*  
Built with **Python, CLIP, t-SNE, and Streamlit**  
Project: *“Mapping the DNA of Dior: AI Analysis of Aesthetic Evolution”*
""")
