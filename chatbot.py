import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

# Descargar recursos de NLTK si aún no están descargados
nltk.download('punkt')
nltk.download('wordnet')

# Cargar datos y modelo previamente generados
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Limpiar la oración y convertirla en un conjunto de palabras lematizadas
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Crear una bolsa de palabras binaria según las palabras presentes en la oración
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecir la clase a la que pertenece la oración ingresada
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    return category

# Obtener una respuesta aleatoria de las respuestas asociadas a una categoría
def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Lo siento, no sé cómo ayudarte con eso. ¿Podrías intentar con otra pregunta?"

# Función principal de interacción con el usuario
def chatbot_response(message):
    predicted_class = predict_class(message)
    response = get_response(predicted_class, intents)
    return response

# Loop para la interacción continua con el usuario
while True:
    user_message = input("Tú: ")
    if user_message.lower() == 'salir':
        break
    
    bot_response = chatbot_response(user_message)
    print("Bot:", bot_response)
