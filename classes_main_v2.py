# Standard imports for our project
import string
import random
import re

# imports like numpy, pandas and torch
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Gensim imports for word2vec model
from gensim.models import Word2Vec

# import for sentence transformers
from sentence_transformers import SentenceTransformer

# Spacy imports for NER
import spacy
from spacy.language import Language
from spacy.tokens import Span

# Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

SYMPT = "SYMPTOM"
ANAT = "ANATOMY"

@Language.component("medical_ner")
def medical_ner(doc):
    """
    This is custom NER pipeline made using spacy to detect
    SYMPTOMS and ANATOMY patterns
    """
    symptom_patterns = ["pain", "ache", "fever", "cough", "fatigue"]
    anatomy_patterns = ["head", "chest", "stomach", "back", "leg", "arm"]

    new_ents = []
    for token in doc:
        if token.lower_ in symptom_patterns:
            new_ents.append(Span(doc, token.i, token.i + 1, label=SYMPT))
        elif token.lower_ in anatomy_patterns:
            new_ents.append(Span(doc, token.i, token.i + 1, label=ANAT))

    doc.ents = list(doc.ents) + new_ents
    return doc

nlp.add_pipe("medical_ner", last=True)

# Extracting the medical entities from the text passed
def extract_medical_entities(text):
    doc = nlp(text)
    entities = {
        "SYMPTOM": [],
        "ANATOMY": [],
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": []
    }

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    return entities

# Text preprocessing
def text_preprocessing(text):
    # Lowercasing the text
    text = text.lower()
    # Removing punctuation marks from the text
    text = ''.join([char for char in text if char not in string.punctuation])
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    # Tokenizing the text
    tokens = word_tokenize(text)
    # Removal of stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Data preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['processed_text'] = data['text'].apply(text_preprocessing)
    return data

# SBERT feature extraction
def extract_sbert_features(texts):
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return sbert_model.encode(texts)

def train_classifier(data):
    X = extract_sbert_features(data['processed_text'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(C=1, penalty='l2', solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)

    return clf