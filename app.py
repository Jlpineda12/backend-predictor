import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# --- CONFIGURACIÓN ---
# Ajusta esto al largo que usaste en tu entrenamiento en Colab
MAX_SEQUENCE_LEN = 50  # Ejemplo: si usaste input_length=50
model = None
tokenizer = None

def load_resources():
    global model, tokenizer
    print("⏳ Cargando modelo y tokenizer...")
    
    # 1. Cargar el Tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # 2. Cargar el Modelo
    model = tf.keras.models.load_model('modelo_prediccion.h5')
    print("✅ ¡Modelo y Tokenizer cargados correctamente!")

# Cargamos todo al iniciar
load_resources()

@app.route('/')
def home():
    return "🤖 El servidor de IA está funcionando."

@app.route('/prediccion', methods=['POST'])
def predict():
    try:
        # 1. Recibir el texto de la página web
        data = request.get_json()
        input_text = data.get('texto', '')

        if not input_text:
            return jsonify({'error': 'No enviaste texto'}), 400

        # 2. Preprocesar el texto (Igual que en Colab)
        # Convertir texto a secuencia de números
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        # Aplicar padding (relleno)
        token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN-1, padding='pre')

        # 3. Predecir
        predicted = model.predict(token_list, verbose=0)
        
        # 4. Obtener la palabra con mayor probabilidad
        predicted_word_index = np.argmax(predicted, axis=-1)[0] # A veces es axis=1
        
        # 5. Convertir el número de vuelta a palabra
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        
        # Devolver resultado
        full_prediction = input_text + " " + output_word
        return jsonify({'prediccion': full_prediction, 'palabra_siguiente': output_word})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render asigna un puerto automáticamente, usamos ese o el 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
