import numpy as np
import tensorflow as tf
keras = tf.keras

from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from enum import Enum

import joblib
import pickle

import os
import json
from core.utils.logger import logger

import tensorflow_addons as tfa

class LossType(Enum):
    binary: str = "binary_crossentropy"
    categorical: str = "categorical_crossentropy"

class ClsModel:
    def __init__(self, path: str = None, verbose = True):
        if path is None:
            self.reset()
        else:
            self.load(path)

        self.verbose = verbose

    def create_feature_conv(
        self,
        filters: int,
        kernel_size: int,
        layer, 
        kernel_initializer: tf.keras.initializers.Initializer, 
    ):
        l_conv = keras.layers.Conv1D(filters, kernel_size, kernel_initializer=kernel_initializer, padding="same", activation="relu")(layer)
        max_pool = keras.layers.GlobalMaxPooling1D()(l_conv)
        ave_pool = keras.layers.GlobalAveragePooling1D()(l_conv)
        feature_conv = keras.layers.Concatenate(axis=1)([max_pool, ave_pool])

        return feature_conv

    def create_fully_connect(
        self, 
        units: int,
        layer, 
        kernel_initializer: tf.keras.initializers.Initializer, 
        l2: float=0.001, 
        dropout: float=0.25
    ):
        dense = keras.layers.Dense(
            units, 
            activation='relu', 
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(l2=l2)
        )(layer)

        fully_connect = keras.layers.Dropout(dropout)(dense)
        return fully_connect
    
    def create_model(self) -> tf.keras.Model:
        embedding_layer = keras.layers.Embedding(
            self.config['vocab_size'],
            self.config['embedding_dim'], 
            input_length=self.config['max_length'], 
            trainable=True
        )
        embedding_layer.build((None, ))
        embedding_layer.set_weights([self.embedding_matrix])

        he_normal_initializer = keras.initializers.he_normal(seed=42)

        sequence_input = keras.layers.Input(shape=(self.config['max_length'],), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        features_convs = [
            self.create_feature_conv(64, i, embedded_sequences, he_normal_initializer)
            for i in range(2, 5)
        ]
        feature_sent = keras.layers.Concatenate(axis=1)(features_convs)

        fully_connected_1 = self.create_fully_connect(256, feature_sent, he_normal_initializer)
        fully_connected_2 = self.create_fully_connect(128, fully_connected_1, he_normal_initializer)
        
        outputs = keras.layers.Dense(self.config['num_class'], activation='softmax')(fully_connected_2)

        model = keras.Model(sequence_input, outputs)
        return model

    def train_word2vec(self, X_data):
        logger.info("Training word2vec model")
        
        sentences = [row.strip().split(" ") for row in X_data]
        w2v_model = Word2Vec(
            sentences,
            vector_size=50,
            window=3,
            min_count=1,
            sg= 1,
            workers=4,
            seed = 42,
            epochs =100
        )
        
        return w2v_model

    def train_tokenizer(self, X_data):
        logger.info("Training tokenizer")
        
        tokenizer = keras.preprocessing.text.Tokenizer(filters="", lower=True, split=' ', oov_token="UNK")
        tokenizer.fit_on_texts(X_data)

        vocab_size = len(tokenizer.word_index) + 1

        return tokenizer, vocab_size
    
    def reset(self):
        self.model = None
        self.tokenizer = None
        self.w2v_model = None
        self.config = {}

    def prepare_X(self, X_data):
        if isinstance(X_data, str):
            X_data = [X_data]

        X_data_vector = self.tokenizer.texts_to_sequences(X_data)
        X_data_padded = keras.preprocessing.sequence.pad_sequences(X_data_vector, maxlen=self.config['max_length'], padding='post')

        return X_data_padded

    def prepare_y(self, y_data):
        return tf.keras.utils.to_categorical(y_data, num_classes=self.config['num_class'])
    
    def prepare_data(self, X_data, y_data):
        
        X_data_encode = self.prepare_X(X_data)
        y_data_encode = self.prepare_y(y_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data_encode, y_data_encode, 
            stratify=y_data_encode,
            test_size=0.1, 
            random_state=42
        )

        return X_train, X_test, y_train, y_test

    def get_embedding_matrix(self):
        embedding_matrix = np.asarray([np.random.uniform(-0.01, 0.01, self.config['embedding_dim']) for _ in range(self.config['vocab_size'])])
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.word2vec_model.wv.get_vector(word, norm=True)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            except:
                pass

        return embedding_matrix

    def compile(
        self,
        lr: float,
        optimizer: tf.keras.optimizers = None, 
        loss: LossType = LossType.binary,
    ):
        optimizer = optimizer if optimizer else keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss=loss.value, optimizer=optimizer, metrics=[
            keras.metrics.BinaryAccuracy() if loss == LossType.binary else keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ])
    
    def train(
        self, 
        data,
        embedding_dim: int = 50,
        name = "cls_model",
        lr: float = 0.001, 
        optimizer: tf.keras.optimizers = None, 
        loss: LossType = LossType.binary,
        reset: bool = False,
        save_path: str = "Models"
    ):        
        X_data = data['x']
        y_data = data['y']

        logger.info(f"Training {name} model")
        logger.debug(f"Data Total: {len(X_data)}")

        self.config['embedding_dim'] = embedding_dim
        self.config['num_class'] = data['num_class']
        self.config['max_length'] = data['max_length']
        self.config['class'] = data['class']
        self.config['name'] = name

        if reset:
            self.reset()

        self.w2v_model = self.train_word2vec(X_data)
        self.tokenizer, vocab_size = self.train_tokenizer(X_data)

        self.config['vocab_size'] = vocab_size
        
        self.embedding_matrix = self.get_embedding_matrix()
        
        if self.model is None:
            self.model = self.create_model()

        logger.info("Compile model")
        self.compile(lr, optimizer, loss)

        es_callback = keras.callbacks.EarlyStopping(monitor='val_binary_accracy' if loss == LossType.binary else 'val_categorical_accuracy', patience=15, mode='max', restore_best_weights=True)

        X_train, X_test, y_train, y_test = self.prepare_data(X_data, y_data)

        logger.debug(f"Train : {len(X_data)}")
        logger.debug(f"Test  : {len(X_test)}")

        logger.info("Training model")
        hist = self.model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=32, shuffle=True, callbacks=[es_callback])

        logger.info("Evaluate model")
        score = self.model.evaluate(X_test, y_test, batch_size=32)

        logger.info(f"Score: {score}")

        self.save(name, save_path)

    def save(self, name, path):
        save_path = os.path.join(path, name)

        os.makedirs(save_path, exist_ok=True)
            
        self.model.save(os.path.join(save_path, "model.keras"))

        joblib.dump(self.tokenizer, os.path.join(save_path, "tokenizer.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(self.w2v_model, os.path.join(save_path, "w2v_model.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(self.embedding_matrix, os.path.join(save_path, "embedding_matrix.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
        
        json.dump(
            self.config, 
            open(os.path.join(save_path, "config.json"), "w", encoding="utf-8"), 
            ensure_ascii=False,
            indent=4,
            sort_keys=True
        )

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} not found! Please train model first!")
        
        self.config = json.load(open(os.path.join(path, "config.json"), "r", encoding="utf8"))
        logger.info(f"Loading {self.config['name']} model from: {path}")
        
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"), custom_objects={"Addons>F1Score": tfa.metrics.F1Score})
        self.tokenizer = joblib.load(os.path.join(path, "tokenizer.pkl"))
        self.w2v_model = joblib.load(os.path.join(path, "w2v_model.pkl"))
        self.embedding_matrix = joblib.load(os.path.join(path, "embedding_matrix.pkl"))
    
    async def predict(self, query):
        x_feature = self.prepare_X(query)

        with tf.device('/cpu:0'):
            pred_prob = self.model.predict(x_feature, verbose=self.verbose)[0]
            
        class_prob = [
            {
                'class': self.config['class'][idx],
                'prob': prob
            }
            for idx, prob in enumerate(pred_prob)
        ]

        class_prob = sorted(class_prob, key = lambda x: x['prob'], reverse=True)

        return class_prob