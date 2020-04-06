import io
import logging

import nltk
import numpy as np
import pandas as pd

from model import build_siamese_model

logging.basicConfig(level=logging.INFO)

WORD_EMBEDDING_PATH = '/Users/ahmadizzan/data/word-embeddings'
FASTEXT_PATH = f'{WORD_EMBEDDING_PATH}/fastext-wiki.id.vec'

GLOBAL_DATA_PATH = '/Users/ahmadizzan/data'
GLOVE_6B_PATH = f'{GLOBAL_DATA_PATH}/glove.6B/glove.6B.100d.txt'

# STS_DATA_PATH = '/Users/ahmadizzan/Academics/ta/data/sts-all'
STS_DATA_PATH = '/Users/ahmadizzan/Academics/ta/data/final-data'
MODELS_PATH = '/Users/ahmadizzan/Academics/ta/models'


def encode(sentence, word_vec, feature_size=50, max_len=50):
    words = nltk.word_tokenize(sentence)
    embeddings = []

    for word in words:
        if word in word_vec:
            embeddings.append(word_vec[word])
        if len(embeddings) >= max_len:
            break

    embeddings = [[0] * feature_size
                  for _ in range(max(0, max_len - len(embeddings)))
                  ] + embeddings
    embeddings = np.array(embeddings).astype(np.float32)
    return embeddings


def load_word_embedding(embedding_path):
    word_vec = {}
    with io.open(embedding_path, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        # next(f)
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')
    return word_vec


def preprocess_data(sts_df, word_vec, feature_size):
    data = sts_df
    text1 = data['text1_id'].apply(lambda x: x.lower()).values
    text2 = data['text2_id'].apply(lambda x: x.lower()).values
    scores = data['score'].values

    logging.info('Encoding text into embeddings')
    text1_encoded = np.array(
        [encode(x, word_vec, feature_size=feature_size) for x in text1])
    text2_encoded = np.array(
        [encode(x, word_vec, feature_size=feature_size) for x in text2])
    scaled_scores = scores / 5

    return text1_encoded, text2_encoded, scaled_scores


def print_train_history(history):
    logging.info('Evaluation metrics')
    logging.info('====================')
    for h_key, h_val in history.history.items():
        logging.info(f'Key: {h_key}')
        logging.info(f'Val: {h_val}')
        logging.info('---------------')


def main():
    logging.info('Loading STS dataset')
    # sts = pd.read_csv(f'{STS_DATA_PATH}/sts-all/sts-all.tsv', sep='\t')
    # sts = pd.read_csv(f'{STS_DATA_PATH}/translated_sts.csv', sep='\t')
    train_data = pd.read_csv(f'{STS_DATA_PATH}/train.tsv', sep='\t')
    test_data = pd.read_csv(f'{STS_DATA_PATH}/test.tsv', sep='\t')

    logging.info('Loading word embedding data')
    feature_size = 300
    word_vec = load_word_embedding(FASTEXT_PATH)

    logging.info(f'Data preprocessing')
    train_text1_encoded, train_text2_encoded, train_scaled_scores = preprocess_data(
        train_data, word_vec, feature_size)
    test_text1_encoded, test_text2_encoded, test_scaled_scores = preprocess_data(
        test_data, word_vec, feature_size)


    logging.info('Building siamese ML model')
    model = build_siamese_model(input_dim=text1_encoded.shape[2], n_hidden=100)

    logging.info('Training model')
    history = model.fit([train_text1_encoded, train_text2_encoded],
                        scaled_scores,
                        validation_data=([test_text1_encoded, test_text2_encoded], test_scaled_scores),
                        epochs=200,
                        batch_size=128)

    logging.info('Saving trained model')
    model.save(f'{MODELS_PATH}/id_v4.h5')

    print_train_history(history)


if __name__ == '__main__':
    main()
