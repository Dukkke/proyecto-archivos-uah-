from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
import random

# --- Inicializar Flask ---
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

# --- Configuración ---
EMBEDDINGS_FILE = 'models/document_embeddings.pkl'
SEARCH_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
search_model = None
DOC_DATA = []
DOC_EMBEDDINGS = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONVERSATIONAL_MODEL_FILE = 'models/conversational_model.pkl'
conversational_model = None
INTENTS_RESPONSES = {}

# --- 1. Cargar modelos ---
def load_all_models():
    global search_model, DOC_DATA, DOC_EMBEDDINGS, conversational_model, INTENTS_RESPONSES, DEVICE
    
    # Cargar Cerebro 1 (Recepcionista)
    print(f"Cargando 'Cerebro 1 (Recepcionista)' desde {CONVERSATIONAL_MODEL_FILE}...")
    try:
        with open(CONVERSATIONAL_MODEL_FILE, 'rb') as f:
            conversational_model = pickle.load(f)
        with open('intents_conversacionales.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
            for intent in intents_data['intents']:
                INTENTS_RESPONSES[intent['tag']] = intent['responses']
        print("✅ Cerebro 1 cargado.")
    except Exception as e:
        print(f"❌ Error cargando Cerebro 1: {e}")
        return False

    # Cargar Cerebro 2 (Archivista)
    print(f"Cargando 'Cerebro 2 (Archivista)' ({SEARCH_MODEL_NAME})...")
    try:
        search_model = SentenceTransformer(SEARCH_MODEL_NAME)
        search_model.to(DEVICE)
        print(f"✅ Modelo de búsqueda cargado en {DEVICE}.")
    except Exception as e:
        print(f"❌ Error cargando modelo de búsqueda: {e}")
        return False

    # Cargar embeddings
    print(f"Cargando 'Memoria del Archivista' desde {EMBEDDINGS_FILE}...")
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            DOC_DATA = data['data']
            DOC_EMBEDDINGS = data['embeddings'].to(DEVICE)
        print(f"✅ {len(DOC_DATA)} documentos listos.")
    except Exception as e:
        print(f"❌ Error cargando embeddings: {e}")
        return False

    return True

# --- 2. Funciones ---
def get_intent(query):
    if conversational_model is None:
        return "buscar"
    intent = conversational_model.predict([query])[0]
    confidence = conversational_model.decision_function([query]).max()
    if confidence < 0.3:
        return "buscar"
    return intent

def get_conversational_response(intent_tag):
    return random.choice(INTENTS_RESPONSES.get(intent_tag, ["No entendí eso."]))

def semantic_search(query, top_k=5):
    if DOC_EMBEDDINGS is None or search_model is None:
        return []
    query_embedding = search_model.encode(query, convert_to_tensor=True, device=DEVICE)
    cos_scores = util.pytorch_cos_sim(query_embedding, DOC_EMBEDDINGS)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(DOC_DATA)))
    resultados = []
    for score, idx in zip(top_results[0], top_results[1]):
        if score > 0.4:
            resultados.append({
                'title': DOC_DATA[idx].get('title', 'Sin Título'),
                'url': DOC_DATA[idx].get('url', '#'),
                'relevance_score': score.item()
            })
    return resultados

# --- 3. Endpoint principal ---
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chatbot_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    if request.method == 'POST':
        try:
            data = request.get_json()
            query = data.get('message', '').strip()
            if not query:
                return jsonify({'error': 'Mensaje vacío'}), 400

            print(f"--- Query recibida: '{query}' ---")
            intent = get_intent(query.lower())

            if intent == 'buscar':
                resultados = semantic_search(query)
                if resultados:
                    response_text = f"📄 Resultados para '{query}':<br><br>"
                    for i, result in enumerate(resultados):
                        response_text += f"{i+1}. {result['title']}<br>"
                        response_text += f"   🔗 <a href='{result['url']}' target='_blank'>Acceder al documento</a><br><br>"
                else:
                    response_text = f"Lo siento, no encontré resultados para '{query}'."
            else:
                response_text = get_conversational_response(intent)

            return jsonify({'response': response_text})

        except Exception as e:
            print(f"Error procesando la solicitud: {e}")
            return jsonify({'error': 'Error interno del servidor'}), 500

    return jsonify({'error': 'Método no permitido'}), 405

# --- 4. Main ---
if __name__ == '__main__':
    if load_all_models():
        print("🚀 Servidor Flask iniciado en http://0.0.0.0:5000")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("--- Error: No se pudo iniciar el servidor. ---")
