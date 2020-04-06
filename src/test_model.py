import logging

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import layers, \
                             models, \
                             Input

from model import pearson_correlation
from main import encode, load_word_embedding, FASTEXT_PATH, STS_DATA_BASE_PATH

logging.basicConfig(level=logging.INFO)

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
        'models/id_v5.h5',
        custom_objects={'pearson_correlation': pearson_correlation})

    layer_input = model.get_layer('left_input').input
    layer_output = model.get_layer('bidirectional').output
    intermediate_model = models.Model(inputs=layer_input, outputs=layer_output)

    # sentence_pairs = [
    #     ('Berapa lama Anda bisa menyimpan cokelat di dalam freezer?', 'Berapa lama saya bisa menyimpan adonan roti di lemari es?'),
    #     ('restoran itu adalah tempat saya makan', 'saya makan siang di restoran'),
    #     ('nilai saya jelek', 'ipk tidak menentukan kesuksesan'),
    #     ('aku bermain bola', 'kucing mengejar seekor tikus')
    # ]

    # logging.info('Testing samples')

    # for sent1, sent2 in sentence_pairs:
    #     cosine_sim = compute_similarity(sent1, sent2, word_vec, intermediate_model)
    #     logging.info('=' * 50)
    #     logging.info(f'sent1: {sent1}')
    #     logging.info(f'sent2: {sent2}')
    #     logging.info(f'Cosine similarity: {cosine_sim}')    

    sts = pd.read_csv(f'{STS_DATA_BASE_PATH}/test.tsv', sep='\t')
    # sts = sts.reindex(np.random.permutation(sts.index))

    sent1_list = sts['text1_id'].values
    sent2_list = sts['text2_id'].values
    scores = sts['score'].values
    predictions = []
    for sent1, sent2 in zip(sent1_list, sent2_list):
        sim = compute_similarity(sent1, sent2, word_vec, intermediate_model)
        predictions.append(sim)
    
    pearson = compute_pearson(scores, predictions)
    logging.info(f'Pearson correlation: {pearson}')
