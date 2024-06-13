import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Descargar recursos necesarios de NLTK si aún no están descargados
nltk.download('punkt')
nltk.download('wordnet')

# Cargar archivo JSON de intents
intents = json.loads(open('intents.json').read())

# Inicialización de lematizador
lemmatizer = WordNetLemmatizer()

# Listas para almacenar palabras, clases y documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Procesar intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar palabras en el patrón
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Agregar el documento al corpus
        documents.append((word_list, intent["tag"]))
        # Agregar el tag a la lista de clases si no está presente
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizar y normalizar palabras
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardar palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar aleatoriamente y convertir a arrays numpy
random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Definir el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save("chatbot_model.h5")

print("Modelo entrenado y guardado como 'chatbot_model.h5'")
