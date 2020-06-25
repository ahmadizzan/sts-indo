import argparse
import io
import logging
import json
import time

import nltk
import numpy as np
import pandas as pd
import scipy

from model import compute_pearson, compute_spearman
from train_model import FASTEXT_PATH, STS_DATA_BASE_PATH


class TextVectorizer:

    def __init__(self, embedding_path):
        word_vec = {}
        with io.open(embedding_path, 'r', encoding='utf-8') as f:
            # if word2vec or fasttext file : skip first line "next(f)"
            next(f)
            for line in f:
                word, vec = line.split(' ', 1)
                word_vec[word] = np.fromstring(vec, sep=' ')
        self.word_vec = word_vec

    def encode_to_vectors(self, text, max_len=30):
        words = nltk.word_tokenize(text)
        embeddings = []

        for word in words:
            if word in self.word_vec:
                embeddings.append(self.word_vec[word])
            if len(embeddings) >= max_len:
                break

        embeddings = np.array(embeddings).astype(np.float32)
        return embeddings

    def encode(self, text, max_len=30):
        embeddings = self.encode_to_vectors(text, max_len)
        return np.average(embeddings, axis=0)

class VectorSimilarityModel:

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def predict(self, X):
        X1, X2 = X

        predictions = []
        for text1, text2 in zip(X1, X2):
            vec1 = self.vectorizer.encode(text1)
            vec2 = self.vectorizer.encode(text2)

            cosine_similarity = 1 - scipy.spatial.distance.cosine(vec1, vec2)
            predictions.append(cosine_similarity)

        return predictions

def preprocess_data(sts_df):
    data = sts_df
    text1 = data['text1'].apply(lambda x: x.lower()).values
    text2 = data['text2'].apply(lambda x: x.lower()).values
    scores = data['score'].values

    scaled_scores = scores / 5

    return text1, text2, scaled_scores

if __name__ == '__main__':
    vectorizer = TextVectorizer(embedding_path=FASTEXT_PATH)
    model = VectorSimilarityModel(vectorizer=vectorizer)

    sts = pd.read_csv(f'{STS_DATA_BASE_PATH}/test.tsv', sep='\t')
    X1, X2, y = preprocess_data(sts)
    
    predictions = model.predict([X1, X2])

    print('Pearson:', compute_pearson(y, predictions))
    print('Spearman:', compute_spearman(y, predictions))
