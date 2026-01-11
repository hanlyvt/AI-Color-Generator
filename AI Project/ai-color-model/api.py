from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os


app = FastAPI()


# CORS voor Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# AUTO-DETECT LATEST TRAINED MODEL
# ============================================
def get_latest_model_path(models_dir='./trained_models'):
    """Automatisch detecteer het meest recente trained model"""
    if not os.path.exists(models_dir):
        return None
    
    model_folders = [
        os.path.join(models_dir, d) 
        for d in os.listdir(models_dir) 
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    
    if not model_folders:
        return None
    
    latest_model = max(model_folders, key=os.path.getmtime)
    return latest_model


# ============================================
# LAAD MODEL BIJ OPSTARTEN
# ============================================
print("🚀 Loading AI Color Generator Model...")

# Probeer trained model te laden, anders fallback naar base model
MODEL_PATH = get_latest_model_path()

if MODEL_PATH:
    print(f"✓ Using trained model: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
else:
    print("⚠ No trained model found, using base multilingual model")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(f"✓ Model dimension: {model.get_sentence_embedding_dimension()}D")


# ============================================
# LAAD EMBEDDINGS EN DATA
# ============================================

# Probeer trained embeddings te laden
EMBEDDINGS_FILE = 'color_embeddings_trained.npy'
if not os.path.exists(EMBEDDINGS_FILE):
    EMBEDDINGS_FILE = 'color_embeddings.npy'  # Fallback

color_embeddings = np.load('embeddings/color_embeddings_trained.npy')
print(f"✓ Loaded embeddings: {color_embeddings.shape}")

# Laad kleur data
colors_df = pd.read_csv('data/colors_dataframe_cleaned_keywords.csv')
print(f"✓ Loaded {len(colors_df)} colors")

# Check dimensie match
model_dim = model.get_sentence_embedding_dimension()
embeddings_dim = color_embeddings.shape[1]

if model_dim != embeddings_dim:
    print(f"⚠️ WARNING: Dimension mismatch!")
    print(f"   Model: {model_dim}D, Embeddings: {embeddings_dim}D")
    print(f"   → Please run 'python train_color_model.py' to regenerate embeddings!")
else:
    print(f"✓ Dimensions match: {model_dim}D")

print("\n🎨 API Ready! Visit http://localhost:8000/docs\n")


# ============================================
# REQUEST MODEL
# ============================================
class TextInput(BaseModel):
    text: str
    top_k: int = 5


# ============================================
# ENDPOINTS
# ============================================
@app.post("/match-colors")
def match_colors(data: TextInput):
    """
    Match colors based on text input (supports NL + EN)
    
    Example:
        POST /match-colors
        {
            "text": "innovative tech startup",
            "top_k": 5
        }
    """
    try:
        # Genereer embedding voor input tekst
        text_embedding = model.encode([data.text])
        
        # Bereken similarities
        scores = cosine_similarity(text_embedding, color_embeddings)[0]
        
        # Top K matches
        top_indices = np.argsort(scores)[-data.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            row = colors_df.iloc[idx]
            results.append({
                "color_hex": row['HEX Code'],
                "color_name": row['Color Name'],
                "description": str(row.get('Description', '')) if pd.notna(row.get('Description')) else None,
                "emotion": str(row.get('Emotion', '')) if pd.notna(row.get('Emotion')) else None,
                "use_case": str(row.get('Use Case', '')) if pd.notna(row.get('Use Case')) else None,
                "personality": str(row.get('Personality', '')) if pd.notna(row.get('Personality')) else None,
                "mood": str(row.get('Mood', '')) if pd.notna(row.get('Mood')) else None,
                "score": float(scores[idx])
            })
        
        return {"matches": results}
    
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/")
def root():
    return {
        "message": "AI Color Generator API - Multilingual (NL + EN)",
        "model": MODEL_PATH or "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": model.get_sentence_embedding_dimension(),
        "colors": len(colors_df),
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "embeddings_loaded": color_embeddings is not None,
        "colors_loaded": len(colors_df) > 0
    }


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

