import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, \
                             models, \
                             Input


def pearson_correlation(y_true, y_pred):
    # # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    # fs_pred = y_pred - K.mean(y_pred)
    # fs_true = y_true - K.mean(y_true)
    # covariance = K.mean(fs_true * fs_pred)

    # stdv_true = K.std(y_true)
    # stdv_pred = K.std(y_pred)

    # return covariance / (stdv_true * stdv_pred)
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(1 - r) # TRY: square this.


def build_siamese_model(configs):
    word_emb_dim = configs['word_emb_dim']

    encoder_type = configs['encoder_type']
    encoder_bidirectional = configs['encoder_bidirectional']
    enc_dim = configs['enc_dim']
    pool_type = configs['pool_type']
    
    n_fc_layers = configs['n_fc_layers']
    fc_dim = configs['fc_dim']
    dpout_fc = configs['dpout_fc']

    optimizer = configs['optimizer']

    seed = configs['seed']

    left_input = Input(shape=(None, word_emb_dim), name='left_input')
    right_input = Input(shape=(None, word_emb_dim), name='right_input')

    rnn = None
    if encoder_type == 'LSTM':
        rnn = layers.LSTM(enc_dim,
                          unit_forget_bias=True,
                          kernel_initializer='he_normal',
                          kernel_regularizer='l2',
                          name='rnn')
    elif rnn_type == 'GRU':
        rnn = layers.GRU(enc_dim,
                         kernel_initializer='he_normal',
                         kernel_regularizer='l2',
                         name='rnn')
    else:
        raise Exception(f'Encoder type ({encoder_type}) not supported!')

    if encoder_bidirectional:
        rnn = layers.Bidirectional(rnn, merge_mode=pool_type, name='bidirectional')

    left_output = rnn(left_input)
    right_output = rnn(right_input)

    l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
    dot_product = lambda x: x[0] * x[1]

    merged_abs_diff = layers.Lambda(function=l1_norm,
                                    output_shape=lambda x: x[0],
                                    name='l1_distance')(
                                        [left_output, right_output])
    merged_dot_product = layers.Lambda(function=dot_product,
                                       output_shape=lambda x: x[0],
                                       name='dot_product')(
                                           [left_output, right_output])

    merged = layers.concatenate(
        [left_output, right_output, merged_abs_diff, merged_dot_product])

    for i in range(n_fc_layers):
        merged = layers.Dropout(dpout_fc, seed=seed)(merged)
        merged = layers.Dense(fc_dim, activation='relu', name=f'inner_dense{i}')(merged)

    predictions = layers.Dense(1,
                               activation='sigmoid',
                               name='similarity_layer')(merged)

    model = models.Model([left_input, right_input], predictions)
    model.compile(
        loss=correlation_coefficient_loss,
        optimizer=optimizer,
        metrics=[pearson_correlation])

    return model


if __name__ == '__main__':
    pass
