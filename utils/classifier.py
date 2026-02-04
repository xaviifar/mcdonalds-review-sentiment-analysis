"""
Script para clasificar automáticamente palabras del dataset usando NLTK Opinion Lexicon
"""

import pandas as pd
import re
from collections import Counter
from nltk.corpus import opinion_lexicon
import nltk
import sys
import os

# Configurar NLTK
venv_path = sys.prefix 
nltk_data_dir = os.path.join(venv_path, "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Descargar si no está disponible
try:
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
except:
    print("Descargando NLTK Opinion Lexicon...")
    nltk.download("opinion_lexicon", download_dir=nltk_data_dir)
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())

# Stopwords en inglés
ENGLISH_STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his',
    'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
    'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
    'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like',
    'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year',
    'your', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
    'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
    'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is',
    'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having', 'may',
    'should', 'am', 'being', 'very', 's', 't', 've', 're', 'll', 'd', 'm'
}

def extract_and_classify_words():
    """Extrae palabras del dataset y las clasifica automáticamente con NLTK."""
    
    print("\n" + "="*70)
    print("🔍 CLASIFICACIÓN AUTOMÁTICA DE PALABRAS CON NLTK")
    print("="*70)
    
    # Leer dataset
    print("\n📂 Leyendo dataset...")
    df = pd.read_csv('./data/clean_dataset.csv')
    print(f"   ✓ Cargadas {len(df)} reseñas")
    
    # Combinar todo el texto
    print("\n📝 Extrayendo palabras...")
    all_text = ' '.join(df['review_translated_text'].dropna().astype(str))
    
    # Tokenizar y limpiar
    words = re.findall(r'\b[a-z]+\b', all_text.lower())
    
    # Filtrar stopwords y palabras muy cortas
    words = [w for w in words if w not in ENGLISH_STOPWORDS and len(w) >= 3]
    
    # Contar frecuencias
    word_freq = Counter(words)
    
    print(f"   ✓ Encontradas {len(word_freq)} palabras únicas")
    
    # Clasificar con NLTK
    print("\n🏷️  Clasificando con NLTK Opinion Lexicon...")
    
    classified_words = []
    stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for word, freq in word_freq.items():
        if word in positive_words:
            polarity = 'positive'
            stats['positive'] += 1
        elif word in negative_words:
            polarity = 'negative'
            stats['negative'] += 1
        else:
            polarity = 'neutral'
            stats['neutral'] += 1
        
        classified_words.append({
            'word': word,
            'polarity': polarity,
            'frequency': freq
        })
    
    # Crear DataFrame y ordenar por frecuencia
    df_dict = pd.DataFrame(classified_words)
    df_dict = df_dict.sort_values('frequency', ascending=False)
    
    # Guardar
    output_file = './dictionary/custom_dictionary.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_dict.to_csv(output_file, index=False)
    
    # Mostrar estadísticas
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS DE CLASIFICACIÓN")
    print("="*70)
    print(f"   Total de palabras:     {len(word_freq):,}")
    print(f"   ✓ Positivas:           {stats['positive']:,} ({stats['positive']/len(word_freq)*100:.1f}%)")
    print(f"   ✗ Negativas:           {stats['negative']:,} ({stats['negative']/len(word_freq)*100:.1f}%)")
    print(f"   ○ Neutras:             {stats['neutral']:,} ({stats['neutral']/len(word_freq)*100:.1f}%)")
    
    # Mostrar las 10 palabras más frecuentes de cada categoría
    print("\n" + "="*70)
    print("🔝 TOP 10 PALABRAS MÁS FRECUENTES POR CATEGORÍA")
    print("="*70)
    
    for polarity in ['positive', 'negative', 'neutral']:
        top_words = df_dict[df_dict['polarity'] == polarity].head(10)
        print(f"\n{polarity.upper()}:")
        for _, row in top_words.iterrows():
            print(f"   • {row['word']:20s} → {row['frequency']:,} veces")
    
    print("\n" + "="*70)
    print(f"✅ Diccionario guardado en: {output_file}")
    print("="*70 + "\n")
    
    return df_dict

if __name__ == '__main__':
    df_result = extract_and_classify_words()
