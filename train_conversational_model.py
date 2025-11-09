import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import os

INTENTS_FILE = 'intents_conversacionales.json'
MODEL_FILE = 'models/conversational_model.pkl'

def train_intent_classifier():
    """
    Entrena un clasificador de intenciones (Cerebro Recepcionista)
    usando scikit-learn y lo guarda en un archivo .pkl.
    """
    print(f"Cargando intenciones desde {INTENTS_FILE}...")
    try:
        with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error al cargar {INTENTS_FILE}: {e}")
        return

    patterns = [] # Frases del usuario (X)
    tags = []     # Intenciones (y)
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])

    if not patterns:
        print("No se encontraron patrones en el JSON.")
        return

    print(f"Encontrados {len(patterns)} patrones para {len(set(tags))} intenciones.")
    
    # --- Crear el Pipeline de IA ---
    # 1. TfidfVectorizer: Convierte texto en números (vectores)
    # 2. LinearSVC: Es un clasificador muy rápido y eficiente
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='word', stop_words=None)),
        ('classifier', LinearSVC(C=1.0, class_weight='balanced'))
    ])

    print("Entrenando el modelo 'Recepcionista'...")
    # Entrenar el modelo
    model_pipeline.fit(patterns, tags)

    print("Entrenamiento completado.")
    
    # Guardar el modelo entrenado
    os.makedirs('models', exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print(f"✅ Modelo 'Recepcionista' guardado en {MODEL_FILE}")

if __name__ == "__main__":
    train_intent_classifier()