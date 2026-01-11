"""
Generate Color Embeddings met Trained Multilingual Model
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

# Auto-detect latest model
def get_latest_model_path(models_dir='./trained_models'):
    if not os.path.exists(models_dir):
        return None
    model_folders = [
        os.path.join(models_dir, d) 
        for d in os.listdir(models_dir) 
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    if not model_folders:
        return None
    return max(model_folders, key=os.path.getmtime)

# Laad nieuwste trained model
MODEL_PATH = get_latest_model_path()
print(f"🚀 Loading model: {MODEL_PATH}\n")

model = SentenceTransformer(MODEL_PATH)
print(f"✓ Model dimension: {model.get_sentence_embedding_dimension()}D")

# Laad color data
df = pd.read_csv('data/colors_with_combined_text.csv')
print(f"✓ Loaded {len(df)} colors\n")

# Genereer embeddings voor ALLE kleuren met het nieuwe model
print("Generating embeddings with multilingual model...")
embeddings = model.encode(
    df['combined_text'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=32
)

print(f"\n✓ Embeddings shape: {embeddings.shape}")
print(f"✓ Dimension: {embeddings.shape[1]}D")

# Sla op (OVERSCHRIJFT oude embeddings!)
output_file = 'color_embeddings_trained.npy'
np.save(output_file, embeddings)
print(f"\n✅ Saved to: {output_file}")
print(f"✅ Ready to use with test_model.py and api.py!\n")
