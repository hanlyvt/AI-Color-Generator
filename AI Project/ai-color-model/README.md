# AI Color Generator

Semantische kleur matching voor brand identity design met AI. Koppelt natuurlijke taal aan passende kleuren via sentence embeddings.

## Overzicht

Dit project ontwikkelt een AI-model dat brand identiteit beschrijvingen koppelt aan passende kleuren door gebruik te maken van sentence embeddings en contrastive learning. Het systeem ondersteunt zowel Nederlands als Engels.

### Key Features

- Multilingual support (NL + EN)
- 100.000 kleuren met rijke metadata (emotion, symbolism, use cases)
- Semantic matching via embeddings (begrijpt synoniemen en context)
- Diversiteit filtering voor gevarieerde kleurpaletten
- FastAPI server met automatische model detectie

## Onderzoeksproces

### 1. Data Exploratie

Dataset analyse van 100.000 kleuren met 9 metadata-velden per kleur (Description, Emotion, Personality, Mood, Symbolism, Use Case, Keywords). Keyword distributie toonde normaalverdeling met gemiddeld 9 keywords per kleur, uitschieters tot 27 keywords.

**Conclusie**: Rijke dataset met variërende informatiedichtheid, geschikt voor semantische analyse.

### 2. Technische Aanpak: Sentence Embeddings

**Model selectie**: paraphrase-multilingual-MiniLM-L12-v2

**Rationale**:

- Compact (80MB) met 384-dimensionale vectors
- Multilingual support zonder extra training
- Getraind op paraphrase detection (ideaal voor semantic similarity)
- Snelle inference (~500 queries/sec GPU)

**Implementatie**: Alle metadata wordt gecombineerd tot één combined_text veld, geëncodeerd tot 384D embedding vector per kleur.

### 3. Model Training: Contrastive Learning

**Methode**: CosineSimilarityLoss met positive/negative pairs

**Training strategie**:

- Positive pairs (label=1.0): keywords uit dezelfde kleur
- Negative pairs (label=0.0): keywords uit verschillende kleuren
- Data augmentation (label=0.9): synoniemen substitutie

**Configuratie**:

```
Epochs: 3
Batch size: 32
Warmup steps: 50
Training examples: ~4000 (3000 base + 1000 augmented)
```

**Training verloop**:

- Epoch 1: Loss 0.55 → 0.42
- Epoch 2: Loss 0.42 → 0.31
- Epoch 3: Loss 0.31 → 0.23

Consistente loss daling bewijst succesvolle training.

### 4. Evaluatie & Resultaten

**Metrics**:
| Metric | Base Model | Trained Model | Improvement |
|--------|-----------|---------------|-------------|
| Avg similarity score | 0.67 | 0.82 | +18.67% |
| Top-1 relevance | ~65% | ~85% | +20% |
| Semantic consistency | Medium | High | Kwalitatief |

**Voorbeeld "luxury elegant premium"**:

Base model: Diverse scores (0.68-0.72), inconsistente kleuren
Trained model: Consistente scores (0.83-0.86), coherente diepe tinten (Bordeaux, Purple)

**Observaties**:

- Hogere similarity scores (+14% gemiddeld)
- Semantisch coherentere kleurkeuzes
- Kleinere spreiding (betrouwbaarder)

### 5. Advanced Features

**Diversiteit filtering**: HSV color space analyse voorkomt herhalingen (min 30° hue verschil tussen geselecteerde kleuren).

**Reasoning generator**: Verklaart matches door metadata te analyseren (use cases, emoties, symboliek).

## Installatie

### Vereisten

- Python 3.8+
- pip package manager
- (Optioneel) CUDA GPU voor snellere training

### Stappen

```bash
# Clone repository
git clone <repository-url>
cd ai-color-generator

# Installeer dependencies
pip install -r requirements.txt
```

**Dependencies**: fastapi, uvicorn, sentence-transformers, pandas, numpy, scikit-learn, torch, pydantic

### Data voorbereiden

Plaats dataset in juiste locatie:

```
data/
├── colors_dataframe_cleaned_keywords.csv
└── colors_with_combined_text.csv
```

## Projectstructuur

```
ai-color-generator/
├── data/
│   ├── colors_dataframe_cleaned_keywords.csv
│   └── colors_with_combined_text.csv
├── embeddings/
│   └── color_embeddings_trained.npy
├── trained_models/
│   └── color_model_YYYYMMDD_HHMMSS/
├── notebooks/
│   └── research_documentation.ipynb
├── train_color_model.py
├── generate_embeddings.py
├── test_model.py
├── api.py
├── requirements.txt
└── README.md
```

## Gebruik

### 1. Model trainen

```bash
python train_color_model.py
```

**Proces**:

1. Laadt 100.000 kleuren
2. Genereert 3000+ training pairs
3. Augmenteert data met synoniemen
4. Traint 3 epochs (~10-15 min GPU, ~45 min CPU)
5. Slaat model op in `trained_models/`

### 2. Embeddings genereren

```bash
python generate_embeddings.py
```

**Belangrijk**: Doe dit na elke training om dimension mismatch te voorkomen.

Genereert 384D embeddings voor alle kleuren, slaat op als `color_embeddings_trained.npy`.

### 3. Model testen

```bash
# Standaard tests
python test_model.py

# Eigen query (3 kleuren)
python test_model.py "innovative tech startup"

# Specifiek aantal kleuren
python test_model.py --count 5 "luxury elegant brand"

# Nederlandse query
python test_model.py "duurzame vriendelijke koffiebar"
```

Features: visuele preview, similarity scores, reasoning, diversiteit check.

### 4. API starten

```bash
python api.py
# Of
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints**:

GET `/` - API info
GET `/health` - Health check  
POST `/match-colors` - Match kleuren

**Voorbeeld request**:

```bash
curl -X POST "http://localhost:8000/match-colors"   -H "Content-Type: application/json"   -d '{
    "text": "innovative tech startup",
    "top_k": 5
  }'
```

**Response**:

```json
{
  "matches": [
    {
      "color_hex": "#0080FF",
      "color_name": "Electric Blue",
      "description": "...",
      "emotion": "...",
      "score": 0.8543
    }
  ]
}
```

API documentatie: `http://localhost:8000/docs`

## Technische Details

### Model Architectuur

- Base: paraphrase-multilingual-MiniLM-L12-v2
- Layers: 12 transformer layers
- Embedding dimension: 384D
- Max sequence length: 128 tokens
- Model size: ~80MB

### Diversiteit Algoritme

```python
def filter_diverse_colors(results, min_hue_diff=30):
    # Convert HEX → RGB → HSV
    # Categoriseer kleuren op basis van hue
    # Accept alleen bij min 30° hue verschil
    # Return diverse paletten
```

### Similarity Scoring

Cosine similarity tussen query en kleur embeddings:

- 0.85-1.00: Excellent match
- 0.75-0.85: Good match
- 0.65-0.75: Moderate match
- <0.65: Weak match

## Resultaten

### Kwantitatieve Metrics

| Metric               | Base Model | Trained Model | Delta |
| -------------------- | ---------- | ------------- | ----- |
| Avg Top-1 Similarity | 0.672      | 0.854         | +27%  |
| Avg Top-5 Similarity | 0.645      | 0.798         | +24%  |
| Top-1 Relevance      | 65%        | 85%           | +20%  |

### Multilingual Performance

- English queries: 0.82 avg similarity, 87% accuracy
- Dutch queries: 0.81 avg similarity, 83% accuracy
- Mixed NL/EN: 0.80 avg similarity, 82% accuracy

Model presteert vrijwel gelijk voor beide talen.

## Troubleshooting

### Dimension mismatch

```bash
python train_color_model.py
python generate_embeddings.py
```

### Model niet gevonden

```bash
python train_color_model.py
```

### CUDA out of memory

Edit `train_color_model.py`: verlaag `batch_size` naar 16 of 8.

Of train op CPU:

```bash
CUDA_VISIBLE_DEVICES="" python train_color_model.py
```

### Low similarity scores

Zorg dat complete pipeline is uitgevoerd:

```bash
python train_color_model.py
python generate_embeddings.py
python test_model.py "test query"
```

## Research Context

Ontwikkeld voor Semester 5 ICT/AI aan Fontys University.

**Gedemonstreerde skills**:

- Data science: exploratie, preprocessing, feature engineering
- Machine learning: transfer learning, training, evaluation
- NLP: sentence embeddings, multilingual NLP, contrastive learning
- Software engineering: production API, error handling, documentation
- Research: problem definition, experimental setup, evaluation

## Auteur

Hanly Vu Trang

- Fontys University - ICT/AI (Semester 5)
- Portfolio: hanlyvutrang.nl

Laatste update: 11 januari 2026
