import argparse
import io
import logging
import json
import time

import nltk
import numpy as np
import pandas as pd

from model import build_siamese_model, compute_pearson, compute_spearman
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)

WORD_EMBEDDING_PATH = '/Users/ahmadizzan/data/word-embeddings'
# FASTEXT_PATH = f'{WORD_EMBEDDING_PATH}/fastext-wiki.id.vec'
FASTEXT_PATH = f'{WORD_EMBEDDING_PATH}/wiki-news-300d-1M.vec'

STS_DATA_BASE_PATH = '/Users/ahmadizzan/Academics/ta/data/final-data'
STS_DATA_PATH = f'{STS_DATA_BASE_PATH}/train.tsv'
MODELS_PATH = '/Users/ahmadizzan/Academics/ta/models'

parser = argparse.ArgumentParser(description='STS training')

# Paths.
parser.add_argument("--datapath", type=str, default=STS_DATA_PATH, help="STS data path")
parser.add_argument("--outputdir", type=str, default='experiments/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.h5')
parser.add_argument("--word_emb_path", type=str, default=FASTEXT_PATH, help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
# parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
# parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="adam", help="adam or sgd")
# parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
# parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
# parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='LSTM', help="see list of encoders")
parser.add_argument("--encoder_bidirectional", type=int, default=0, help="see list of encoders")
parser.add_argument("--encoder_attention", type=int, default=0, help="see list of encoders")
parser.add_argument("--enc_dim", type=int, default=100, help="encoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=32, help="nhid of fc layers")
parser.add_argument("--n_fc_layers", type=int, default=3, help="num of fc layers")
parser.add_argument("--pool_type", type=str, default='ave', help="ave or sum or mul")

# validation split
parser.add_argument("--seed", type=int, default=8898, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

configs = {
    # Paths.
    'datapath': params.datapath,
    'outputdir': params.outputdir,
    'outputmodelname': params.outputmodelname,
    'word_emb_path': params.word_emb_path,
    # Training.
    'n_epochs': params.n_epochs,
    'batch_size': params.batch_size,
    'dpout_fc': params.dpout_fc,
    # 'nonlinear_fc': params.nonlinear_fc,
    'optimizer': params.optimizer,
    # 'decay': params.decay,
    # 'minlr': params.minlr,
    # Model.
    'encoder_type': params.encoder_type,
    'encoder_bidirectional': params.encoder_bidirectional,
    'encoder_attention': params.encoder_attention,
    'enc_dim': params.enc_dim,
    'fc_dim': params.fc_dim,
    'n_fc_layers': params.n_fc_layers,
    'pool_type': params.pool_type,
    # Random seed.
    'seed': params.seed,
    # Data.
    'word_emb_dim': params.word_emb_dim
}

def encode(sentence, word_vec, feature_size=50, max_len=30):
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
        next(f)
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')
    return word_vec


def preprocess_data(sts_df, word_vec, feature_size):
    data = sts_df
    text1 = data['text1'].apply(lambda x: x.lower()).values
    text2 = data['text2'].apply(lambda x: x.lower()).values
    scores = data['score'].values

    logging.info('Encoding text into embeddings')
    text1_encoded = np.array(
        [encode(x, word_vec, feature_size=feature_size) for x in text1])
    text2_encoded = np.array(
        [encode(x, word_vec, feature_size=feature_size) for x in text2])
    scaled_scores = scores / 5

    return text1_encoded, text2_encoded, scaled_scores


def embed_history(configs, history):
    configs['metrics'] = {}
    configs['history'] = {}
    for h_key, h_val in history.history.items():
        configs['metrics'][h_key] = float(h_val[-1])
        configs['history'][h_key] = [float(x) for x in h_val]

def print_train_history(history):
    logging.info('Evaluation metrics')
    logging.info('====================')
    for h_key, h_val in history.history.items():
        logging.info(f'Key: {h_key}')
        logging.info(f'Val: {h_val}')
        logging.info('---------------')


def main():
    time_log = {}

    logging.info('Loading STS dataset')
    load_sts_start_time = time.time()
    train_data = pd.read_csv(params.datapath, sep='\t')
    load_sts_start_time = time.time() - load_sts_start_time
    time_log['load_sts'] = load_sts_start_time

    logging.info('Loading word embedding data')
    load_word_embed = time.time()
    word_vec = load_word_embedding(params.word_emb_path)
    load_word_embed = time.time() - load_word_embed
    time_log['load_word_emb'] = load_word_embed

    logging.info(f'Data preprocessing')
    load_preproc_time = time.time()
    train_text1_encoded, train_text2_encoded, train_scaled_scores = preprocess_data(
        train_data, word_vec, params.word_emb_dim)
    load_preproc_time = time.time() - load_preproc_time
    time_log['load_preproc_time'] = load_preproc_time

    split_data_time = time.time()
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        train_text1_encoded, train_text2_encoded, train_scaled_scores,
        test_size=0.2, shuffle=False)
    split_data_time = time.time() - split_data_time
    time_log['split_data_time'] = split_data_time

    logging.info('Building siamese ML model')
    build_model_time = time.time()
    model = build_siamese_model(configs)
    print(model.summary())
    build_model_time = time.time() - build_model_time
    time_log['build_model_time'] = build_model_time

    logging.info('Training model')
    train_model_time = time.time()
    es = EarlyStopping(
        monitor='val_loss', mode='min',
        restore_best_weights=True, min_delta=0.001, patience=20
    )
    history = model.fit([X1_train, X2_train],
                        y_train,
                        validation_data=([X1_val, X2_val], y_val),
                        epochs=params.n_epochs,
                        batch_size=params.batch_size,
                        callbacks=[es])
    train_model_time = time.time() - train_model_time


    logging.info('Computing final metrics')

    predictions = model.predict([X1_train, X2_train]).reshape(-1)
    final_metrics = {}
    final_metrics['train'] = {
        'pearson': float(compute_pearson(y_train, predictions)),
        'spearman': float(compute_spearman(y_train, predictions))
    }

    predictions = model.predict([X1_val, X2_val]).reshape(-1)
    final_metrics['validation'] = {
        'pearson': float(compute_pearson(y_val, predictions)),
        'spearman': float(compute_spearman(y_val, predictions))
    }
    configs['final_metrics'] = final_metrics
    time_log['train_model_time'] = train_model_time

    
    logging.info('Saving trained model')
    save_model_time = time.time()
    model.save(f'{params.outputdir}/{params.outputmodelname}')
    save_model_time = time.time() - save_model_time
    time_log['save_model_time'] = save_model_time


    time_log['total_time'] = sum(time_log.values())


    configs['time_log'] = time_log
    embed_history(configs, history)


    with open(f'{params.outputdir}/{params.outputmodelname}.metadata', 'w') as f:
        json.dump(configs, f, indent=2)

    print_train_history(history)
    




if __name__ == '__main__':
    main()
