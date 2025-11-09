#!/usr/bin/env python3
"""
Script para consolidar toda la información scrapeada en un solo archivo JSON
Combina datos del archivo_uah_data.json original con archivo_uah_completo_data.json
"""

import json
import os
from datetime import datetime

def consolidar_datos():
    """Consolida todos los datos en un solo archivo JSON completo"""

    print("🔄 Iniciando consolidación de datos...")

    # Archivos de entrada
    archivo_original = 'scraped_data/archivo_uah_data.json'
    archivo_completo = 'scraped_data/archivo_uah_completo_data.json'
    archivo_saludos = 'scraped_data/saludos_e_informacion.json'

    # Archivo de salida
    archivo_consolidado = 'scraped_data/archivo_uah_consolidado_completo.json'

    datos_consolidados = {
        'metadata': {
            'fecha_consolidacion': datetime.now().isoformat(),
            'version': '2.0',
            'descripcion': 'Archivo consolidado con toda la información del Archivo Patrimonial UAH',
            'fuente': 'https://archivopatrimonial.uahurtado.cl'
        },
        'estadisticas': {},
        'documentos': [],
        'intents_chatbot': [],
        'configuracion_chatbot': {}
    }

    # Cargar datos completos (prioridad alta)
    if os.path.exists(archivo_completo):
        print("📖 Cargando datos completos...")
        with open(archivo_completo, 'r', encoding='utf-8') as f:
            documentos_completos = json.load(f)
            datos_consolidados['documentos'] = documentos_completos
        print(f"   ✅ {len(documentos_completos)} documentos completos cargados")
    else:
        print("⚠️  Archivo de datos completos no encontrado")

    # Cargar datos originales si faltan algunos
    if os.path.exists(archivo_original):
        print("📖 Cargando datos originales...")
        with open(archivo_original, 'r', encoding='utf-8') as f:
            documentos_originales = json.load(f)

        # Crear mapa de URLs para evitar duplicados
        urls_existentes = {doc['url'] for doc in datos_consolidados['documentos']}

        # Agregar documentos que no estén en los completos
        nuevos_documentos = []
        for doc in documentos_originales:
            if doc['url'] not in urls_existentes:
                nuevos_documentos.append(doc)

        if nuevos_documentos:
            datos_consolidados['documentos'].extend(nuevos_documentos)
            print(f"   ✅ {len(nuevos_documentos)} documentos adicionales agregados")

    # Cargar datos de chatbot (saludos e información)
    if os.path.exists(archivo_saludos):
        print("🤖 Cargando datos del chatbot...")
        with open(archivo_saludos, 'r', encoding='utf-8') as f:
            datos_chatbot = json.load(f)

        # Extraer intents si existen
        if 'intents' in datos_chatbot:
            datos_consolidados['intents_chatbot'] = datos_chatbot['intents']
            print(f"   ✅ {len(datos_chatbot['intents'])} intents de chatbot cargados")

        # Extraer configuración si existe
        if 'configuracion' in datos_chatbot:
            datos_consolidados['configuracion_chatbot'] = datos_chatbot['configuracion']

    # Calcular estadísticas
    print("📊 Calculando estadísticas...")
    documentos = datos_consolidados['documentos']

    estadisticas = {
        'total_documentos': len(documentos),
        'total_caracteres_contenido': sum(len(doc.get('full_content', '')) for doc in documentos),
        'total_imagenes': sum(len(doc.get('images', [])) for doc in documentos),
        'total_enlaces': sum(len(doc.get('links', [])) for doc in documentos),
        'total_fechas': sum(len(doc.get('dates', [])) for doc in documentos),
        'documentos_con_isad': sum(1 for doc in documentos if doc.get('areas_isad')),
        'documentos_con_imagenes': sum(1 for doc in documentos if doc.get('images')),
        'documentos_con_enlaces': sum(1 for doc in documentos if doc.get('links')),
        'temas_principales': {},
        'fechas_cubiertas': set()
    }

    # Analizar temas principales
    temas_counter = {}
    for doc in documentos:
        areas_isad = doc.get('areas_isad', {})
        for area_name, area_data in areas_isad.items():
            if isinstance(area_data, dict):
                for key, value in area_data.items():
                    if 'materia' in key.lower() and isinstance(value, str):
                        tema = value.lower().strip()
                        if len(tema) > 3:  # Solo temas significativos
                            temas_counter[tema] = temas_counter.get(tema, 0) + 1

    # Top 10 temas
    estadisticas['temas_principales'] = dict(sorted(temas_counter.items(), key=lambda x: x[1], reverse=True)[:10])

    # Rango de fechas
    for doc in documentos:
        fechas = doc.get('dates', [])
        for fecha in fechas:
            if isinstance(fecha, str) and len(fecha) >= 4:  # Al menos año
                try:
                    # Extraer año
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', fecha)
                    if year_match:
                        estadisticas['fechas_cubiertas'].add(int(year_match.group()))
                except:
                    pass

    estadisticas['fechas_cubiertas'] = sorted(list(estadisticas['fechas_cubiertas']))

    datos_consolidados['estadisticas'] = estadisticas

    # Crear intents mejorados para el chatbot
    print("🎯 Generando intents mejorados para chatbot...")
    intents_mejorados = []

    # Intent de saludo
    intents_mejorados.append({
        'tag': 'saludo',
        'patterns': ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'hey', 'hi', 'saludos'],
        'responses': [
            '¡Hola! Soy el asistente del Archivo Patrimonial UAH. ¿En qué puedo ayudarte?',
            '¡Buen día! Estoy aquí para ayudarte con información del archivo patrimonial.',
            '¡Hola! ¿Qué información necesitas sobre el Archivo Patrimonial UAH?'
        ]
    })

    # Intent de búsqueda general
    intents_mejorados.append({
        'tag': 'busqueda_general',
        'patterns': ['buscar', 'encontrar', 'necesito informacion', 'quiero saber', 'dime sobre'],
        'responses': [
            'Claro, puedo ayudarte a buscar información. ¿Qué tema específico te interesa?',
            'Estoy aquí para ayudarte con búsquedas. ¿Qué información necesitas?'
        ]
    })

    # Intent sobre iglesia católica
    documentos_iglesia = [doc for doc in documentos if 'iglesia' in doc.get('title', '').lower() or
                         any('iglesia' in str(value).lower() for area in doc.get('areas_isad', {}).values()
                             for value in area.values() if isinstance(value, str))]

    if documentos_iglesia:
        intents_mejorados.append({
            'tag': 'iglesia_catolica',
            'patterns': ['iglesia catolica', 'iglesia', 'religion', 'catolico', 'fe cristiana'],
            'responses': [
                f'La Iglesia Católica es un tema importante en el archivo. Tengo información sobre {len(documentos_iglesia)} documentos relacionados. ¿Qué aspecto específico te interesa?',
                f'Encontré {len(documentos_iglesia)} documentos sobre la Iglesia Católica. ¿Quieres que te muestre algunos?'
            ]
        })

    # Intent sobre derechos humanos
    documentos_ddhh = [doc for doc in documentos if 'derechos humanos' in doc.get('full_content', '').lower() or
                      'violacion' in doc.get('full_content', '').lower()]

    if documentos_ddhh:
        intents_mejorados.append({
            'tag': 'derechos_humanos',
            'patterns': ['derechos humanos', 'ddhh', 'violaciones', 'represion', 'dictadura'],
            'responses': [
                f'Tengo información sobre derechos humanos en {len(documentos_ddhh)} documentos. Este es un tema central del archivo.',
                f'Los derechos humanos son un tema recurrente. Encontré {len(documentos_ddhh)} documentos relacionados.'
            ]
        })

    # Intent de estadísticas
    intents_mejorados.append({
        'tag': 'estadisticas',
        'patterns': ['estadisticas', 'cuantos documentos', 'que hay en el archivo', 'informacion general'],
        'responses': [
            f'El archivo contiene {estadisticas["total_documentos"]} documentos con {estadisticas["total_caracteres_contenido"]:,} caracteres de contenido. ¿Quieres saber más sobre algún tema específico?',
            f'Tenemos {estadisticas["total_documentos"]} documentos procesados, {estadisticas["documentos_con_imagenes"]} con imágenes y {estadisticas["documentos_con_isad"]} con metadatos ISAD(G) completos.'
        ]
    })

    # Intent de ayuda
    intents_mejorados.append({
        'tag': 'ayuda',
        'patterns': ['ayuda', 'help', 'que puedes hacer', 'como usar', 'instrucciones'],
        'responses': [
            'Puedo ayudarte a buscar información en el Archivo Patrimonial UAH. Puedes preguntarme sobre temas específicos, buscar documentos, o pedir estadísticas del archivo.',
            'Estoy aquí para ayudarte con consultas sobre el archivo. Prueba preguntando sobre "iglesia católica", "derechos humanos", o cualquier tema que te interese.'
        ]
    })

    datos_consolidados['intents_chatbot'] = intents_mejorados

    # Guardar archivo consolidado
    print(f"💾 Guardando archivo consolidado: {archivo_consolidado}")
    with open(archivo_consolidado, 'w', encoding='utf-8') as f:
        json.dump(datos_consolidados, f, ensure_ascii=False, indent=2)

    # Verificar tamaño del archivo
    file_size = os.path.getsize(archivo_consolidado)
    print(f"✅ Archivo consolidado guardado: {file_size:,} bytes")

    # Mostrar resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE CONSOLIDACIÓN")
    print("="*60)
    print(f"📄 Total de documentos: {estadisticas['total_documentos']}")
    print(f"📝 Total de caracteres: {estadisticas['total_caracteres_contenido']:,}")
    print(f"🖼️  Total de imágenes: {estadisticas['total_imagenes']}")
    print(f"🔗 Total de enlaces: {estadisticas['total_enlaces']}")
    print(f"📅 Total de fechas: {estadisticas['total_fechas']}")
    print(f"🏷️  Documentos con ISAD(G): {estadisticas['documentos_con_isad']}")
    print(f"🎯 Intents generados: {len(intents_mejorados)}")
    print(f"📅 Rango de fechas: {min(estadisticas['fechas_cubiertas']) if estadisticas['fechas_cubiertas'] else 'N/A'} - {max(estadisticas['fechas_cubiertas']) if estadisticas['fechas_cubiertas'] else 'N/A'}")

    if estadisticas['temas_principales']:
        print(f"🎯 Temas principales: {', '.join(list(estadisticas['temas_principales'].keys())[:5])}")

    print("="*60)

    return archivo_consolidado

if __name__ == "__main__":
    consolidar_datos()
