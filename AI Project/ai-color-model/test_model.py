"""
Test Model - AI Color Generator (VISUAL VERSION met REASONING + DIVERSITEIT)
Diverse kleuren zonder herhalingen in hetzelfde kleurenschema
MULTILINGUAL SUPPORT (NL + EN)
"""



from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Back, Style
import sys
import colorsys
import os



# Initialiseer colorama
init(autoreset=True)



# ============================================
# 1. CONFIGURATIE (AUTO-DETECT LATEST MODEL)
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
    
    # Sorteer op modificatie tijd (nieuwste eerst)
    latest_model = max(model_folders, key=os.path.getmtime)
    return latest_model


# Probeer automatisch het nieuwste model te vinden
MODEL_PATH = get_latest_model_path()

# Fallback naar hardcoded path
if MODEL_PATH is None:
    MODEL_PATH = "./trained_models/color_model_20251217_182402"

EMBEDDINGS_FILE = 'embeddings/color_embeddings_trained.npy'
DATA_FILE = 'data/colors_with_combined_text.csv'


# ============================================
# 2. LAAD MODEL EN DATA
# ============================================
def load_model_and_data():
    """Laad getraind model, embeddings en dataset"""
    print(f"{Fore.YELLOW}Model en data laden...{Style.RESET_ALL}")
    
    # Check of model bestaat
    if not os.path.exists(MODEL_PATH):
        print(f"{Fore.RED}ERROR: Model niet gevonden: {MODEL_PATH}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Tip: Train eerst je model met 'python train_color_model.py'{Style.RESET_ALL}")
        sys.exit(1)
    
    model = SentenceTransformer(MODEL_PATH)
    df = pd.read_csv(DATA_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    
    # Check embedding dimensies
    model_dim = model.get_sentence_embedding_dimension()
    embeddings_dim = embeddings.shape[1]
    
    print(f"{Fore.GREEN}✓ Model geladen: {MODEL_PATH}")
    print(f"✓ Dataset geladen: {len(df)} kleuren")
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Model dimensie: {model_dim}")
    
    if model_dim != embeddings_dim:
        print(f"{Fore.RED}⚠ WARNING: Dimensie mismatch!")
        print(f"  Model: {model_dim}D, Embeddings: {embeddings_dim}D")
        print(f"  → Run 'python train_color_model.py' opnieuw!{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✓ Dimensies matchen: {model_dim}D{Style.RESET_ALL}\n")
    
    return model, df, embeddings



# ============================================
# 3. KLEUR ANALYSE HELPERS
# ============================================
def hex_to_rgb(hex_code):
    """Convert HEX naar RGB tuple"""
    try:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (128, 128, 128)  # Fallback grijs



def rgb_to_hsv(rgb):
    """Convert RGB naar HSV voor kleuranalyse"""
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hsv(r, g, b)



def get_color_category(hex_code):
    """Bepaal kleur categorie (rood, blauw, groen, etc.)"""
    rgb = hex_to_rgb(hex_code)
    h, s, v = rgb_to_hsv(rgb)
    
    # Als saturatie laag is -> grijs/wit/zwart
    if s < 0.15:
        if v < 0.3:
            return 'black'
        elif v > 0.8:
            return 'white'
        else:
            return 'grey'
    
    # Bepaal kleur op basis van hue
    h_degree = h * 360
    
    if h_degree < 15 or h_degree >= 345:
        return 'red'
    elif 15 <= h_degree < 45:
        return 'orange'
    elif 45 <= h_degree < 75:
        return 'yellow'
    elif 75 <= h_degree < 165:
        return 'green'
    elif 165 <= h_degree < 195:
        return 'cyan'
    elif 195 <= h_degree < 255:
        return 'blue'
    elif 255 <= h_degree < 285:
        return 'purple'
    elif 285 <= h_degree < 345:
        return 'magenta'
    
    return 'unknown'



def hex_to_ansi_bg(hex_code):
    """Convert HEX naar ANSI terminal achtergrondkleur"""
    try:
        hex_code = hex_code.lstrip('#')
        r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return f'\033[48;2;{r};{g};{b}m'
    except:
        return Back.WHITE



# ============================================
# 4. DIVERSITEIT FILTER (GEFIXED)
# ============================================
def filter_diverse_colors(results_df, top_n=3, min_hue_diff=30):
    """
    Filter kleuren voor diversiteit - geen dubbele kleurenschema's
    
    Args:
        results_df: DataFrame met kleuren en scores
        top_n: Aantal diverse kleuren om te returnen
        min_hue_diff: Minimale hue verschil in graden (30 = goed onderscheid)
    
    Returns:
        Gefilterde DataFrame met diverse kleuren
    """
    if len(results_df) == 0:
        return results_df
    
    diverse_results = []
    used_categories = set()
    used_hues = []
    
    for _, row in results_df.iterrows():
        if len(diverse_results) >= top_n:
            break
        
        hex_code = row['HEX Code']
        category = get_color_category(hex_code)
        rgb = hex_to_rgb(hex_code)
        h, s, v = rgb_to_hsv(rgb)
        hue_degree = h * 360
        
        # Check of deze kleur categorie al gebruikt is
        if category in used_categories and category not in ['grey', 'white', 'black']:
            # Extra check: is de hue voldoende verschillend?
            too_similar = False
            for used_hue in used_hues:
                hue_diff = min(abs(hue_degree - used_hue), 360 - abs(hue_degree - used_hue))
                if hue_diff < min_hue_diff:
                    too_similar = True
                    break
            
            if too_similar:
                continue
        
        # Voeg toe aan diverse resultaten
        diverse_results.append(row)
        used_categories.add(category)
        used_hues.append(hue_degree)
    
    # Als we niet genoeg diverse kleuren hebben, vul aan met best scorende
    if len(diverse_results) < top_n:
        for _, row in results_df.iterrows():
            if len(diverse_results) >= top_n:
                break
            # FIX: Vergelijk HEX codes als strings, niet als Series
            hex_already_used = any(r['HEX Code'] == row['HEX Code'] for r in diverse_results)
            if not hex_already_used:
                diverse_results.append(row)
    
    return pd.DataFrame(diverse_results).reset_index(drop=True)



# ============================================
# 5. REASONING GENERATOR
# ============================================
def generate_reasoning(row, query):
    """Genereer uitleg waarom deze kleur past bij de query"""
    
    use_case = str(row.get('Use Case', ''))
    emotion = str(row.get('Emotion', ''))
    personality = str(row.get('Personality', ''))
    symbolism = str(row.get('Symbolism', ''))
    mood = str(row.get('Mood', ''))
    
    reasons = []
    
    if use_case and use_case != 'nan':
        reasons.append(f"Geschikt voor: {use_case[:100]}")
    
    if emotion and emotion != 'nan':
        reasons.append(f"Emotie: {emotion[:80]}")
    
    if personality and personality != 'nan':
        reasons.append(f"Persoonlijkheid: {personality[:80]}")
    
    if symbolism and symbolism != 'nan':
        reasons.append(f"Symboliek: {symbolism[:80]}")
    
    if mood and mood != 'nan':
        reasons.append(f"Sfeer: {mood[:80]}")
    
    return reasons if reasons else ["Geen specifieke use case beschikbaar"]



# ============================================
# 6. ZOEKFUNCTIE MET DIVERSITEIT
# ============================================
def find_colors_diverse(query, model, df, embeddings, top_n=3, search_pool=20):
    """
    Vind diverse kleuren - geen herhalingen in kleurenschema
    
    Args:
        top_n: Aantal diverse kleuren (default 3)
        search_pool: Zoek in top X resultaten voor diversiteit (default 20)
    """
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Haal top X resultaten op
    top_indices = similarities.argsort()[-search_pool:][::-1]
    
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
    # Filter voor diversiteit
    diverse_results = filter_diverse_colors(results, top_n=top_n)
    
    return diverse_results



# ============================================
# 7. VISUELE EVALUATIE MET REASONING + DIVERSITEIT
# ============================================
def evaluate_query_visual(query, model, df, embeddings, top_n=3):
    """Evalueer query met diverse kleuren en reasoning"""
    
    # Header
    print(f"\n{'='*90}")
    print(f"{Fore.CYAN}{Style.BRIGHT}Query: '{query}'{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Toon {top_n} diverse kleuren (geen dubbele kleurenschema's){Style.RESET_ALL}")
    print(f"{'='*90}\n")
    
    # Zoek diverse kleuren
    results = find_colors_diverse(query, model, df, embeddings, top_n=top_n)
    
    # Toon kleur categorieën
    color_categories = [get_color_category(row['HEX Code']) for _, row in results.iterrows()]
    print(f"{Fore.MAGENTA}Kleur diversiteit: {', '.join(color_categories).title()}{Style.RESET_ALL}\n")
    
    # Toon resultaten
    for idx, (_, row) in enumerate(results.iterrows(), 1):
        color_name = row['Color Name']
        hex_code = row['HEX Code']
        score = row['similarity_score']
        category = get_color_category(hex_code)
        
        # Kleur blok
        color_ansi = hex_to_ansi_bg(hex_code)
        color_block = f"{color_ansi}{'      '}{Style.RESET_ALL}"
        
        # Score bar
        bar_length = int(score * 40)
        score_bar = f"{Fore.GREEN}{'█' * bar_length}{Style.DIM}{'░' * (40-bar_length)}{Style.RESET_ALL}"
        
        # Print header met categorie
        print(f"{Fore.YELLOW}{Style.BRIGHT}#{idx} {color_name[:45]}{Style.RESET_ALL} "
              f"{Fore.WHITE}[{category.upper()}]{Style.RESET_ALL}")
        print(f"  {color_block} {Fore.WHITE}{hex_code:<10}{Style.RESET_ALL} {score_bar} {Fore.YELLOW}{score:.4f}{Style.RESET_ALL}")
        
        # Genereer en toon reasoning
        reasons = generate_reasoning(row, query)
        print(f"\n  {Fore.MAGENTA}{Style.BRIGHT}Waarom deze kleur?{Style.RESET_ALL}")
        
        for reason in reasons[:3]:  # Max 3 redenen voor leesbaarheid
            if len(reason) > 80:
                words = reason.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 <= 80:
                        line += word + " "
                    else:
                        print(f"     • {Fore.CYAN}{line.strip()}{Style.RESET_ALL}")
                        line = word + " "
                if line:
                    print(f"     • {Fore.CYAN}{line.strip()}{Style.RESET_ALL}")
            else:
                print(f"     • {Fore.CYAN}{reason}{Style.RESET_ALL}")
        
        print()
    
    print(f"{'='*90}\n")



# ============================================
# 8. BATCH EVALUATIE
# ============================================
def batch_evaluate(queries, model, df, embeddings, top_n=3):
    """Evalueer meerdere queries"""
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*90}")
    print("BATCH EVALUATIE")
    print(f"{'='*90}{Style.RESET_ALL}")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{Fore.CYAN}[{i}/{len(queries)}]{Style.RESET_ALL}")
        evaluate_query_visual(query, model, df, embeddings, top_n)



# ============================================
# 9. TEST CASES (NL + EN)
# ============================================
TEST_QUERIES = [
    # English
    "innovative tech startup",
    "eco-friendly sustainable brand",
    "luxury elegant premium",
    "artificial intelligence machine learning",
    "mystical magical enchanting",
    "playful joyful fun",
    # Dutch
    "duurzame vriendelijke koffiebar",
    "moderne technologie startup",
    "rustige spa wellness",
    "energieke sportschool fitness",
    "professionele zakelijke corporate"
]



# ============================================
# 10. MAIN FUNCTIE
# ============================================
def main():
    """Hoofdfunctie"""
    
    # Header
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}╔{'═'*88}╗")
    print(f"║{' '*5}AI COLOR GENERATOR - MULTILINGUAL (NL+EN) - 3 diverse kleuren{' '*5}║")
    print(f"╚{'═'*88}╝{Style.RESET_ALL}\n")
    
    # Laad model en data
    model, df, embeddings = load_model_and_data()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Check voor --count flag
        top_n = 3
        args = sys.argv[1:]
        
        if '--count' in args:
            idx = args.index('--count')
            try:
                top_n = int(args[idx + 1])
                args = args[:idx] + args[idx+2:]
            except (IndexError, ValueError):
                print(f"{Fore.RED}Gebruik: --count [getal]{Style.RESET_ALL}")
        
        query = ' '.join(args)
        evaluate_query_visual(query, model, df, embeddings, top_n)
        return
    
    # Test 1: English query
    print(f"{Fore.CYAN}{Style.BRIGHT}### TEST 1: ENGLISH QUERY ###{Style.RESET_ALL}")
    evaluate_query_visual("innovative modern startup", model, df, embeddings, top_n=3)
    
    # Test 2: Dutch query
    print(f"\n{Fore.CYAN}{Style.BRIGHT}### TEST 2: NEDERLANDSE QUERY ###{Style.RESET_ALL}")
    evaluate_query_visual("duurzame vriendelijke koffiebar", model, df, embeddings, top_n=3)
    
    # Test 3: Mixed NL/EN
    print(f"\n{Fore.CYAN}{Style.BRIGHT}### TEST 3: GEMIXTE QUERY (NL+EN) ###{Style.RESET_ALL}")
    evaluate_query_visual("sustainable tech modern groen", model, df, embeddings, top_n=3)
    
    # Test 4: Meer diversiteit
    print(f"\n{Fore.CYAN}{Style.BRIGHT}### TEST 4: MEER DIVERSITEIT (5 kleuren) ###{Style.RESET_ALL}")
    evaluate_query_visual("mystical magical enchanting", model, df, embeddings, top_n=5)
    
    # Test 5: Interactieve mode
    print(f"\n{Fore.CYAN}{Style.BRIGHT}### TEST 5: INTERACTIEVE MODE ###{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Tip: Werkt met Nederlands én Engels!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}     Standaard 3 diverse kleuren{Style.RESET_ALL}\n")
    
    while True:
        try:
            custom_query = input(f"{Fore.CYAN}Query (NL/EN, of 'quit'): {Style.RESET_ALL}").strip()
            
            if custom_query.lower() in ['quit', 'exit', 'q', '']:
                break
            
            # Check voor aantal in query (bijv. "5 innovative startup")
            parts = custom_query.split(maxsplit=1)
            if parts[0].isdigit():
                top_n = int(parts[0])
                custom_query = parts[1] if len(parts) > 1 else ""
            else:
                top_n = 3
            
            if custom_query:
                evaluate_query_visual(custom_query, model, df, embeddings, top_n)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    # Footer
    print(f"\n{Fore.GREEN}{'='*90}")
    print("TESTS VOLTOOID - MULTILINGUAL MODEL GETEST!")
    print(f"{'='*90}{Style.RESET_ALL}\n")



# ============================================
# 11. RUN
# ============================================
if __name__ == "__main__":
    main()
