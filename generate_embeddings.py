import json
import pickle
from sentence_transformers import SentenceTransformer
import os

# --- Configuración ---
DATA_FILE = 'scraped_data/archivo_uah_consolidado_completo.json'
EMBEDDINGS_FILE = 'models/document_embeddings.pkl'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # Usamos el modelo estándar

def formatear_resultados(documentos, consulta):
    """
    Convierte la lista de documentos en un texto ordenado con títulos y links.
    """
    salida = f"## 📄 Resultados para \"{consulta}\"\n\n"
    for i, doc in enumerate(documentos, start=1):
        titulo = doc.get('title', 'Sin título')
        link = doc.get('url', '#')
        doc_id = doc.get('id', i)

        salida += f"{i}. **Documento {doc_id}**\n"
        salida += f"   *{titulo}*\n"
        salida += f"   🔗 [Acceder al documento]({link})\n\n"
    return salida

def generate_embeddings():
    """
    Lee el JSON de scraping y genera embeddings semánticos para los documentos.
    ¡Ahora incluye título, descripción y contenido!
    """
    print(f"Cargando modelo de IA (Archivista): {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error: No se pudo cargar el modelo SentenceTransformer: {e}")
        return

    print(f"Cargando documentos desde {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            data = json.load(file)
            documentos = data.get('documentos', [])
    except Exception as e:
        print(f"Error al leer el JSON: {e}")
        return

    if not documentos:
        print("Error: No se encontró la clave 'documentos' en el JSON.")
        return

    corpus = []
    corpus_data = []
    
    for doc in documentos:
        titulo = doc.get('title', '')
        descripcion = doc.get('description', '')
        contenido = doc.get('content', '').replace('\n', ' ').strip()
        
        texto_para_ia = f"{titulo}. {descripcion}. {contenido}"
        
        if texto_para_ia.strip() != ". .":
            corpus.append(texto_para_ia)
            corpus_data.append(doc)

    if not corpus:
        print("Error: No se pudo generar un corpus de texto (¿JSON vacío?).")
        return

    print(f"Se prepararon {len(corpus)} documentos para la IA (con contenido completo).")

    print("Generando embeddings (vectores de significado)...")
    try:
        corpus_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True)
    except Exception as e:
        print(f"Error durante la codificación de embeddings: {e}")
        return

    os.makedirs('models', exist_ok=True)
    
    print(f"Guardando 'memoria' de la IA en {EMBEDDINGS_FILE}...")
    try:
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({
                'data': corpus_data, 
                'embeddings': corpus_embeddings
            }, f)
    except Exception as e:
        print(f"Error al guardar el archivo pickle: {e}")
        return

    print("--- ¡Proceso completado! ---")
    print(f"Archivo '{EMBEDDINGS_FILE}' creado exitosamente.")

    # Ejemplo de salida ordenada para validar
    print("\nEjemplo de resultados ordenados:\n")
    print(formatear_resultados(corpus_data[:5], "dictadura"))

if __name__ == "__main__":
    generate_embeddings()
