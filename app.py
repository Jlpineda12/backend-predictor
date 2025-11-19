import os
import pickle
import numpy as np
import tensorflow as tf
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MAX_SEQUENCE_LEN = 50
model = None
tokenizer = None
RESOURCES_LOADED = False
LOAD_ERROR = ""

# --- DEBUGGER DE ARCHIVOS (Esto nos dirá la verdad) ---
print("📂 [DIAGNÓSTICO] Listando archivos en la carpeta actual:")
try:
    files = os.listdir(os.getcwd())
    for f in files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"   - {f} ({size:.2f} MB)")
except Exception as e:
    print(f"   Error listando archivos: {e}")
# -----------------------------------------------------

def load_resources():
    global model, tokenizer, RESOURCES_LOADED, LOAD_ERROR
    print("⏳ [INICIO] Intentando cargar recursos...")
    
    try:
        # Verificación explícita antes de cargar
        if not os.path.exists('tokenizer.pickle'):
            raise FileNotFoundError("Falta el archivo 'tokenizer.pickle'")
        if not os.path.exists('modelo_prediccion.h5'):
            raise FileNotFoundError("Falta el archivo 'modelo_prediccion.h5'")

        gc.collect()
        
        print("... Cargando Tokenizer")
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        print("... Cargando Modelo .h5")
        model = tf.keras.models.load_model('modelo_prediccion.h5', compile=False)
        
        RESOURCES_LOADED = True
        print("✅ [ÉXITO] ¡Todo cargado correctamente!")
        
    except Exception as e:
        LOAD_ERROR = str(e)
        print(f"❌ [ERROR FATAL] No se pudo cargar: {e}")
        # NO lanzamos error para permitir que el servidor arranque y nos cuente qué pasó

# Intentamos cargar, pero si falla, el servidor arranca igual
load_resources()

@app.route('/')
def home():
    status = "✅ Cargado" if RESOURCES_LOADED else f"❌ Falló: {LOAD_ERROR}"
    return f"🤖 Estado del Servidor: {status}<br>Archivos detectados: {str(os.listdir())}"

@app.route('/prediccion', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    # Si la carga falló al inicio, devolvemos el error ahora
    if not RESOURCES_LOADED:
        return jsonify({
            'error': 'Error de Carga en Servidor', 
            'details': f'El backend no pudo cargar el modelo. Causa: {LOAD_ERROR}'
        }), 500

    try:
        data = request.get_json()
        input_text = data.get('texto', '')
        
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN-1, padding='pre')
        
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        
        return _corsify_actual_response(jsonify({'prediccion': input_text + " " + output_word}))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _build_cors_preflight_response():
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
