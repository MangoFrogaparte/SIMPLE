import json
import random
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

nltk.download('punkt')

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Preprocessing
stemmer = PorterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w.isalnum()]
words = sorted(list(set(words)))
labels = sorted(labels)

# Bag of Words
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1 if w in wrds else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Neural Network
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(8, activation="relu"))
model.add(Dense(len(output[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(training, output, epochs=200, batch_size=8, verbose=1)

# Save model and data
model.save("chatbot_model.h5")
pickle.dump((words, labels, training, output), open("chatbot_data.pkl", "wb"))
