# AI Color Generator - App Setup

Handleiding voor het starten van de volledige AI Color Generator applicatie.

## Projectstructuur

```
ai-color-model/              # Backend (FastAPI + ML model)
├── api.py
├── train_color_model.py
├── generate_embeddings.py
├── test_model.py
├── requirements.txt
├── data/
├── embeddings/
└── trained_models/

color-generator-app/         # Frontend (Next.js)
├── app/
├── components/
├── package.json
└── README.md
```

## Vereisten

### Backend

- Python 3.8+
- pip package manager

### Frontend

- Node.js 16+
- npm of yarn

## Installatie

### 1. Backend Setup (ai-color-model)

```bash
cd ai-color-model

# Installeer Python dependencies
pip install -r requirements.txt
```

**Zorg dat je hebt:**

- Getraind model in `trained_models/`
- Embeddings in `embeddings/color_embeddings_trained.npy`
- Dataset in `data/colors_dataframe_cleaned_keywords.csv`

Als deze bestanden ontbreken, train eerst het model:

```bash
python train_color_model.py
python generate_embeddings.py
```

### 2. Frontend Setup (color-generator-app)

```bash
cd color-generator-app

# Installeer Node dependencies
npm install
```

## App Starten

### Stap 1: Start de Backend API

Open een terminal venster en navigeer naar de backend folder:

```bash
cd ai-color-model

# Start de FastAPI server
python api.py
```

**Output:**

```
🚀 Loading AI Color Generator Model...
✓ Using trained model: ./trained_models/color_model_YYYYMMDD_HHMMSS
✓ Model dimension: 384D
✓ Loaded embeddings: (100000, 384)
✓ Loaded 100000 colors
✓ Dimensions match: 384D

🎨 API Ready! Visit http://localhost:8000/docs

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

De API draait nu op `http://localhost:8000`

### Stap 2: Start de Frontend

Open een **tweede** terminal venster en navigeer naar de frontend folder:

```bash
cd color-generator-app

# Start de Next.js development server
npm run dev
```

**Output:**

```
▲ Next.js 14.x.x
- Local:        http://localhost:3000
- Network:      http://192.168.x.x:3000

✓ Ready in 2.3s
```

De webapp draait nu op `http://localhost:3000`

### Stap 3: Open de App

Open je browser en ga naar:

```
http://localhost:3000
```

Je kunt nu brand beschrijvingen invoeren en real-time kleurmatches ontvangen!

## Gebruik

1. Voer een brand beschrijving in (bijvoorbeeld "innovative tech startup")
2. Klik op "Generate Colors" of druk op Enter
3. Bekijk de top 5 semantisch passende kleuren met metadata

**Ondersteunt Nederlands en Engels:**

- "innovative tech startup"
- "duurzame vriendelijke koffiebar"
- "luxury elegant premium"

## Troubleshooting

### Backend problemen

**Error: "Model niet gevonden"**

```bash
# Train eerst het model
cd ai-color-model
python train_color_model.py
python generate_embeddings.py
```

**Error: "Port 8000 already in use"**

```bash
# Stop andere processen op poort 8000, of wijzig poort in api.py
uvicorn api:app --port 8001
```

### Frontend problemen

**Error: "Failed to fetch"**

- Check of de backend API draait op `http://localhost:8000`
- Controleer CORS configuratie in `api.py`

**Error: "Port 3000 already in use"**

```bash
# Start op andere poort
npm run dev -- -p 3001
```

## API Endpoints

De backend biedt de volgende endpoints:

### GET `/`

API informatie en status

### POST `/match-colors`

Match kleuren op basis van tekst input

**Request:**

```json
{
  "text": "innovative tech startup",
  "top_k": 5
}
```

**Response:**

```json
{
  "matches": [
    {
      "color_hex": "#0080FF",
      "color_name": "Electric Blue",
      "description": "...",
      "emotion": "Trust, innovation",
      "score": 0.8543
    }
  ]
}
```

### GET `/health`

Health check endpoint

## Development

### Backend hot reload

```bash
cd ai-color-model
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend hot reload

Next.js heeft automatisch hot reload bij `npm run dev`

## Productie Deployment

### Backend

```bash
cd ai-color-model
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

```bash
cd color-generator-app
npm run build
npm start
```

## Stoppen

Stop beide servers met `CTRL+C` in hun respectievelijke terminal vensters.

## Demo Video

Zie `App-color-generator.mp4` voor een volledige demonstratie van de werkende applicatie.

---

**Veel succes met de AI Color Generator!**
