"""
AI Color Generator - Model Training Pipeline
=============================================
Train een custom Sentence Transformer model op kleur-brand matching data
"""


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import torch
import os
from datetime import datetime
import json



class ColorModelTrainer:
    """Training pipeline voor custom kleur matching model"""
    
    def __init__(self, base_model='paraphrase-multilingual-MiniLM-L12-v2', output_dir='./trained_models'):
        """
        Args:
            base_model: Pre-trained model om te fine-tunen
            output_dir: Directory voor opslaan trained models
        """
        self.base_model_name = base_model
        self.model = SentenceTransformer(base_model)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Base model geladen: {base_model}")
        print(f"Embedding dimensies: {self.model.get_sentence_embedding_dimension()}")
        print(f"Output directory: {output_dir}")
    
    def load_color_data(self, csv_path='colors_dataframe_cleaned_keywords.csv'):
        """Laad kleur dataset"""
        self.df = pd.read_csv(csv_path)
        
        # Maak combined text zoals in notebook
        self.df['combined_text'] = self.df.apply(
            lambda row: ' '.join([
                str(row['Description']) if pd.notna(row['Description']) else '',
                str(row['Emotion']) if pd.notna(row['Emotion']) else '',
                str(row['Personality']) if pd.notna(row['Personality']) else '',
                str(row['Mood']) if pd.notna(row['Mood']) else '',
                str(row['Symbolism']) if pd.notna(row['Symbolism']) else '',
                str(row['Use Case']) if pd.notna(row['Use Case']) else '',
                str(row['Keywords']) if pd.notna(row['Keywords']) else ''
            ]).strip(),
            axis=1
        )
        
        print(f"Dataset geladen: {len(self.df)} kleuren")
        return self.df
    
    def create_training_examples_from_keywords(self, num_examples=3000):
        """
        Creëer training pairs door keywords te matchen met kleuren
        
        Strategy: Positieve pairs = query met keywords die in color zitten
                  Negatieve pairs = query met keywords die niet matchen
        """
        print(f"\nGenereer {num_examples} training examples...")
        
        examples = []
        
        # Sample kleuren voor training
        sampled_colors = self.df.sample(min(num_examples, len(self.df)))
        
        for idx, row in sampled_colors.iterrows():
            color_text = row['combined_text']
            keywords = str(row['Keywords']).split()
            
            if len(keywords) < 2:
                continue
            
            # POSITIEF PAIR: Query met gerelateerde keywords
            # Neem 1-3 random keywords uit deze kleur
            n_keywords = np.random.randint(1, min(4, len(keywords) + 1))
            positive_query = ' '.join(np.random.choice(keywords, n_keywords, replace=False))
            
            # Score = 1.0 (perfecte match)
            examples.append(InputExample(
                texts=[positive_query, color_text],
                label=1.0
            ))
            
            # NEGATIEF PAIR: Query met totaal andere keywords
            # Neem random kleur met andere keywords
            random_color = self.df.sample(1).iloc[0]
            other_keywords = str(random_color['Keywords']).split()
            
            if len(other_keywords) >= 2:
                n_other = np.random.randint(1, min(4, len(other_keywords) + 1))
                negative_query = ' '.join(np.random.choice(other_keywords, n_other, replace=False))
                
                # Score = 0.0 (geen match)
                examples.append(InputExample(
                    texts=[negative_query, color_text],
                    label=0.0
                ))
        
        print(f"{len(examples)} training examples gegenereerd")
        return examples
    
    def create_training_examples_from_user_feedback(self, feedback_csv='user_feedback.csv'):
        """
        Creëer training data uit user feedback
        
        Expected CSV format:
        query,color_name,hex_code,rating
        "innovative startup",Electric Blue,#0080FF,5
        "calm nature",Forest Green,#228B22,4
        ...
        
        Args:
            feedback_csv: Path naar CSV met user ratings
        """
        if not os.path.exists(feedback_csv):
            print(f"Feedback file niet gevonden: {feedback_csv}")
            return []
        
        print(f"\nLaad user feedback uit {feedback_csv}...")
        feedback_df = pd.read_csv(feedback_csv)
        
        examples = []
        
        for _, row in feedback_df.iterrows():
            query = row['query']
            color_name = row['color_name']
            rating = row['rating']  # 1-5 sterren
            
            # Vind de kleur in dataset
            color_row = self.df[self.df['Color Name'] == color_name]
            
            if color_row.empty:
                continue
            
            color_text = color_row.iloc[0]['combined_text']
            
            # Converteer rating (1-5) naar score (0-1)
            # 5 stars = 1.0, 1 star = 0.0
            score = (rating - 1) / 4.0
            
            examples.append(InputExample(
                texts=[query, color_text],
                label=score
            ))
        
        print(f"{len(examples)} feedback examples geladen")
        return examples
    
    def create_augmented_queries(self, num_augmentations=1):
        """
        Augmenteer training data met synoniemen en variaties
        
        Ondersteunt nu ook Nederlandse synoniemen!
        """
        print(f"\nAugmenteer training data...")
        
        # Synoniem dictionary (NL + EN)
        synonyms = {
            # English
            'innovative': ['cutting-edge', 'modern', 'progressive', 'forward-thinking'],
            'luxury': ['premium', 'elegant', 'sophisticated', 'high-end'],
            'calm': ['peaceful', 'serene', 'tranquil', 'relaxing'],
            'energetic': ['dynamic', 'vibrant', 'lively', 'active'],
            'eco-friendly': ['sustainable', 'green', 'natural', 'organic'],
            'tech': ['technology', 'digital', 'cyber', 'electronic'],
            'warm': ['cozy', 'comfortable', 'inviting', 'welcoming'],
            'bold': ['strong', 'powerful', 'striking', 'confident'],
            'creative': ['artistic', 'imaginative', 'original', 'inventive'],
            'professional': ['corporate', 'business', 'formal', 'executive'],
            # Dutch
            'duurzaam': ['milieuvriendelijk', 'groen', 'natuurlijk', 'ecologisch'],
            'vriendelijk': ['warm', 'gezellig', 'uitnodigend', 'toegankelijk'],
            'modern': ['eigentijds', 'hedendaags', 'actueel', 'innovatief'],
            'rustig': ['kalm', 'sereen', 'vredig', 'ontspannen'],
            'energiek': ['dynamisch', 'levendig', 'actief', 'krachtig']
        }
        
        examples = []
        
        # Sample kleuren
        sampled_colors = self.df.sample(min(1000, len(self.df)))
        
        for _, row in sampled_colors.iterrows():
            color_text = row['combined_text']
            keywords = str(row['Keywords']).lower().split()
            
            # Check of er synoniemen beschikbaar zijn
            for keyword in keywords:
                if keyword in synonyms:
                    # Maak variaties met synoniemen
                    for _ in range(num_augmentations):
                        synonym = np.random.choice(synonyms[keyword])
                        
                        # Vervang keyword door synoniem
                        augmented_query = ' '.join([
                            synonym if k == keyword else k 
                            for k in keywords[:3]  # Max 3 keywords
                        ])
                        
                        examples.append(InputExample(
                            texts=[augmented_query, color_text],
                            label=0.9  # Hoge score, maar niet perfect
                        ))
        
        print(f"{len(examples)} augmented examples gegenereerd")
        return examples
    
    def train(self, 
              num_epochs=3,
              batch_size=32,
              warmup_steps=50,
              use_augmentation=True,
              use_feedback=False,
              feedback_csv='user_feedback.csv'):
        """
        Train het model
        
        Args:
            num_epochs: Aantal training epochs
            batch_size: Batch size voor training
            warmup_steps: Warmup steps voor optimizer
            use_augmentation: Gebruik data augmentation
            use_feedback: Gebruik user feedback data
            feedback_csv: Path naar feedback CSV
        """
        print("\n" + "="*80)
        print("START TRAINING")
        print("="*80)
        
        # Genereer training data
        train_examples = self.create_training_examples_from_keywords(num_examples=3000)
        
        if use_augmentation:
            augmented = self.create_augmented_queries(num_augmentations=1)
            train_examples.extend(augmented)
        
        if use_feedback and os.path.exists(feedback_csv):
            feedback_examples = self.create_training_examples_from_user_feedback(feedback_csv)
            train_examples.extend(feedback_examples)
        
        print(f"\nTotaal training examples: {len(train_examples)}")
        
        # Maak DataLoader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # Loss function: CosineSimilarityLoss
        # Perfect voor regression tasks (scores van 0-1)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Training configuratie
        print(f"\nTraining config:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Loss: CosineSimilarityLoss")
        
        # Train!
        print(f"\nTraining gestart...\n")
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=None,  # We slaan later handmatig op
            show_progress_bar=True
        )
        
        print(f"\nTraining voltooid!")
        
        return self.model
    
    def evaluate(self, test_queries=None):
        """
        Evalueer het model met test queries (NL + EN)
        
        Args:
            test_queries: List van (query, expected_keywords) tuples
        """
        if test_queries is None:
            test_queries = [
                # English
                ("innovative tech startup", ["innovative", "modern", "tech"]),
                ("eco-friendly sustainable", ["eco", "green", "natural"]),
                ("luxury elegant brand", ["luxury", "elegant", "premium"]),
                ("calm peaceful nature", ["calm", "peaceful", "serene"]),
                ("energetic bold sports", ["energetic", "dynamic", "bold"]),
                # Dutch
                ("duurzame vriendelijke koffiebar", ["duurzaam", "groen", "vriendelijk"]),
                ("moderne tech startup", ["modern", "innovatief", "tech"]),
                ("rustige spa wellness", ["rustig", "kalm", "ontspannen"])
            ]
        
        print("\n" + "="*80)
        print("MODEL EVALUATIE")
        print("="*80)
        
        # Encode alle kleuren
        color_embeddings = self.model.encode(
            self.df['combined_text'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        for query, expected_keywords in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Bereken similarities
            similarities = cosine_similarity(query_embedding, color_embeddings)[0]
            
            # Top 5
            top5_indices = similarities.argsort()[-5:][::-1]
            
            print("   Top 5 resultaten:")
            for i, idx in enumerate(top5_indices, 1):
                color_name = self.df.iloc[idx]['Color Name']
                hex_code = self.df.iloc[idx]['HEX Code']
                score = similarities[idx]
                keywords = self.df.iloc[idx]['Keywords']
                
                # Check relevantie
                relevant = any(kw in str(keywords).lower() for kw in expected_keywords)
                marker = "[✓ RELEVANT]" if relevant else "[✗ NOT RELEVANT]"
                
                print(f"   {i}. {marker} {color_name:30s} {hex_code:10s} Score: {score:.3f}")
        
        print("\n" + "="*80)
    
    def save_model(self, model_name=None):
        """Sla trained model op"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"color_model_{timestamp}"
        
        save_path = os.path.join(self.output_dir, model_name)
        self.model.save(save_path)
        
        # Sla ook metadata op
        metadata = {
            'base_model': self.base_model_name,
            'training_date': datetime.now().isoformat(),
            'num_colors': len(self.df),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'model_path': save_path
        }
        
        metadata_path = os.path.join(save_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel opgeslagen: {save_path}")
        return save_path
    
    def load_trained_model(self, model_path):
        """Laad een eerder trained model"""
        self.model = SentenceTransformer(model_path)
        print(f"Trained model geladen: {model_path}")
        return self.model




# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================



if __name__ == "__main__":
    print("AI Color Generator - Model Training\n")
    
    # Initialiseer trainer met MULTILINGUAL model
    trainer = ColorModelTrainer(
        base_model='paraphrase-multilingual-MiniLM-L12-v2',
        output_dir='./trained_models'
    )
    
    # Laad data
    trainer.load_color_data('data/colors_dataframe_cleaned_keywords.csv')
    
    # Train model (GEOPTIMALISEERD + MULTILINGUAL)
    trained_model = trainer.train(
        num_epochs=3,          # Sneller dan 4-8 epochs
        batch_size=32,         # 2x sneller dan 16
        warmup_steps=50,       # Sneller dan 100
        use_augmentation=True,
        use_feedback=False     # Zet True als je user_feedback.csv hebt
    )
    
    # Evalueer met NL + EN queries
    trainer.evaluate()
    
    # Sla op
    model_path = trainer.save_model()
    
    print("\nTraining pipeline voltooid!")
    print(f"Model opgeslagen in: {model_path}")
    print("\nGebruik in je applicatie:")
    print(f"  model = SentenceTransformer('{model_path}')")
