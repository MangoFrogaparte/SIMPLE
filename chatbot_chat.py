import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model

nltk.download('punkt')

stemmer = PorterStemmer()
model = load_model("chatbot_model.h5")
words, labels, _, _ = pickle.load(open("chatbot_data.pkl", "rb"))

with open("intents.json") as file:
    data = json.load(file)

def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("ChatBot: Hello! (type 'quit' to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            print("ChatBot:", random.choice(responses))
        else:
            print("ChatBot: I'm not sure I understand. Can you rephrase?")

if __name__ == "__main__":
    chat()
