import json
import random
import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Function to make sure NLTK data is available
def ensure_nltk_data():
    required_packages = ["punkt", "punkt_tab", "wordnet"]
    for pkg in required_packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

# Ensure all necessary NLTK data is installed
ensure_nltk_data()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get safe file path for intents.json
file_path = os.path.join(os.path.dirname(__file__), "intents.json")

# Load intents
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Preprocess patterns into a corpus
corpus = []
tags = []
responses_map = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        lemmatized = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        corpus.append(" ".join(lemmatized))
        tags.append(intent["tag"])
    responses_map[intent["tag"]] = intent["responses"]

# Vectorize corpus
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

def preprocess_input(user_input):
    tokens = nltk.word_tokenize(user_input)
    lemmatized = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    return " ".join(lemmatized)

def get_intent(user_input):
    processed = preprocess_input(user_input)
    user_vec = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vec, X)

    best_match = np.argmax(similarities)

    if similarities[0][best_match] < 0.3:
        return "default"

    return tags[best_match]

def chatbot():
    print("ChatBot is running! Type 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ")

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Bot:", random.choice(responses_map["goodbye"]))
                break

            intent = get_intent(user_input)
            response = random.choice(responses_map[intent])
            print("Bot:", response)

        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    chatbot()
