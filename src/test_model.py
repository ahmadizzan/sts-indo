import argparse
import logging

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import layers, \
                             models, \
                             Input

from model import compute_pearson, compute_spearman, pearson_correlation, spearman_correlation, correlation_coefficient_loss
from train_model import encode, load_word_embedding, FASTEXT_PATH, STS_DATA_BASE_PATH, preprocess_data

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='STS training')
parser.add_argument("--model-path", type=str, default=None, help="Model path")
params, _ = parser.parse_known_args()

def compute_similarity(sent1, sent2, word_vec, model):
    sent1_encoded = encode(sent1, word_vec, feature_size=feature_size)
    sent2_encoded = encode(sent2, word_vec, feature_size=feature_size)

    x = np.array([sent1_encoded, sent2_encoded])
    predictions = intermediate_model.predict(x)

    sent1_embedding = predictions[0]
    sent2_embedding = predictions[1]

    cosine_sim = (sent1_embedding @ sent2_embedding.T) / (np.linalg.norm(sent1_embedding) * np.linalg.norm(sent2_embedding))

    return cosine_sim

def compute_pearson(y_true, y_pred):
    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    fs_pred = y_pred - np.mean(y_pred)
    fs_true = y_true - np.mean(y_true)
    covariance = np.mean(fs_true * fs_pred)

    stdv_true = np.std(y_true)
    stdv_pred = np.std(y_pred)

    return covariance / (stdv_true * stdv_pred)

if __name__ == '__main__':
    logging.info('Loading word embedding data')
    feature_size = 300
    word_vec = load_word_embedding(FASTEXT_PATH)

    logging.info('Loading pre-trained model')
    model = models.load_model(
        params.model_path,
        custom_objects={
            'correlation_coefficient_loss': correlation_coefficient_loss,
            'pearson_correlation': pearson_correlation,
            'spearman_correlation': spearman_correlation
        })

    sts = pd.read_csv(f'{STS_DATA_BASE_PATH}/test.tsv', sep='\t')
    X1, X2, y = preprocess_data(sts, word_vec, 300)

    logging.info('Predicting labels...')
    predictions = model.predict([X1, X2]).reshape(-1)

    print('Pearson:', compute_pearson(y, predictions))
    print('Spearman:', compute_spearman(y, predictions))
