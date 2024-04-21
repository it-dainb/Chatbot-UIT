import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/ProgramData/miniconda3/envs/cb_uit"

import numpy as np
import tensorflow as tf
keras = tf.keras
import tensorflow_addons as tfa

import regex as re
import unidecode    
import itertools
import string
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from core.utils.logger import logger
from core.config.config import get_config
from core.utils.preprocessing import clean_text
import unicodedata

from fast_map import fast_map

import os
import json
import joblib, pickle
LONGEST_LENGTH = 6

MAX_LINES = 500_000

tf.config.optimizer.set_jit('autoclustering')
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "auto_mixed_precision": True,
})


class AccentRestoreModel:
    def __init__(self, path = None, database = None):
        if path is not None:
            self.load(path)
        self.database = database
        self.accented_chars_vietnamese = [
            'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
            'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
            'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
            'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
            'í', 'ì', 'ỉ', 'ĩ', 'ị',
            'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
            'đ',
        ]
        self.accented_chars_vietnamese.extend([c.upper() for c in self.accented_chars_vietnamese])
        self.pad_token = '\x00'
        self.alphabet = list((f'{self.pad_token} _' + string.ascii_letters + string.digits + ''.join(self.accented_chars_vietnamese)))

        self.config = {}
        self.model = None

    def create_model(self, units=256) -> tf.keras.Model:        
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units = units, input_shape=(self.max_length, len(self.alphabet)), return_sequences=True))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = units, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(self.alphabet))))
        model.add(keras.layers.Activation('softmax'))

        return model

    def compile(
        self,
        lr: float,
        optimizer: tf.keras.optimizers = None, 
    ):
        optimizer = optimizer if optimizer else keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            tfa.metrics.F1Score(num_classes = len(self.alphabet), average = 'micro'),
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ])

    def remove_accent(self, text):
        return unidecode.unidecode(text)

    def extract_phrases(self, text):
        pattern = r'\w[\w ]*|\s\W+|\W+'
        return re.findall(pattern, text)

    @tf.autograph.experimental.do_not_convert
    def gen_ngrams(self, words):
        with tf.device('/gpu:0'):
            ngrams = tf.strings.ngrams(words.split(), self.ngram).numpy().tolist()
        
        for i in np.char.decode(ngrams, encoding='utf-8'):
            yield str(i)

    def process_phrase(self, p):
        ngrams = []

        for ngr in self.gen_ngrams(p):
            if len(ngr) < self.max_length:
                ngrams.append(ngr)
        
        return ngrams

    def clean_text(self, text):
        return clean_text(text, self.database.synonyms_dictionary, tokenizer=False)

    def create_data(self, path):
        ngrams_path = os.path.join(get_config("Path", "data"), f"{self.ngram}_ngrams.pkl")  
        
        if os.path.exists(ngrams_path):
            list_ngrams = joblib.load(ngrams_path)
        else:
            with open(path, "r", encoding="utf8") as f_r:
                lines = f_r.read().split("\n")
            
            phrases = itertools.chain.from_iterable(self.extract_phrases(text) for text in lines)
            phrases = [p.strip() for p in phrases if len(p.strip()) >= self.ngram]
            phrases = list(set(phrases))
            max_p = int(MAX_LINES * 1.2)
            print(max_p)
            phrases = phrases[:max_p]

            clean_phrases = []

            for phrase in tqdm(fast_map(self.clean_text, phrases, threads_limit = 10), total=len(phrases)):
                if len(phrase.split()) >= self.ngram:
                    clean_phrases.append(phrase)
        

            clean_phrases = list(set(clean_phrases))
            clean_phrases = clean_phrases[:MAX_LINES]

            list_ngrams = []

            for ngrams in tqdm(fast_map(self.process_phrase, clean_phrases, threads_limit = 10), total=len(clean_phrases)):
                list_ngrams.extend(ngrams)

            list_ngrams = list(set(list_ngrams))
            joblib.dump(list_ngrams, ngrams_path, protocol=pickle.HIGHEST_PROTOCOL)
            
        train, test = train_test_split(list_ngrams, test_size=0.2, random_state=42)

        return train, test


    def padding(self, text):
        text = self.pad_token + text + self.pad_token * max(0, self.max_length - len(text) - 1)
        return text

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        text = self.padding(text)

        x = np.zeros((self.max_length, len(self.alphabet)))

        for i, c in enumerate(text[:self.max_length]):
            x[i, self.alphabet.index(c)] = 1
        
        return x

    def decode(self, x):
        x = x.argmax(axis=-1)
        
        return ''.join(self.alphabet[i] for i in x if i != 0)

    def generate_data(self, data, batch_size=128):
        cur_index = 0

        while True:
            x, y = [], []
            for _ in range(batch_size):  
                y.append(self.encode(data[cur_index]))
                x.append(self.encode(unidecode.unidecode(data[cur_index])))
                cur_index += 1
                
                if cur_index > len(data)-1:
                    cur_index = 0
            
            yield np.array(x), np.array(y)

    def train(self, save_path= "Models", name = "accent", data_path = "Data/train_accent.txt", ngram = 5, batch_size=1024, lr = 0.001):
        save_path = os.path.join(save_path, name)
        os.makedirs(save_path, exist_ok=True)

        self.ngram = ngram
        self.max_length = LONGEST_LENGTH * ngram
        self.name = name
        
        self.config['max_length'] = self.max_length
        self.config['ngram'] = self.ngram
        self.config['name'] = self.name
        
        train, test = self.create_data(data_path)

        train_generator = self.generate_data(train, batch_size=batch_size)
        test_generator = self.generate_data(test, batch_size=batch_size)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, 'checkpoint_model.keras'),
            save_best_only=True, 
            verbose=1,
            monitor='val_f1_score',
            mode='max',
        )
        es_callback = keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=2, mode='max', restore_best_weights=True)

        with tf.device('/gpu:0'):
            self.model = self.create_model()
            self.compile(lr=lr)

            self.model.fit(
                train_generator, 
                epochs=100,
                steps_per_epoch=len(train)//batch_size, 
                callbacks=[checkpoint_callback, es_callback],
            )

            self.save(save_path)

            score = self.model.evaluate(test_generator, steps=len(test)//batch_size)
            logger.info(f"Score: {score}")

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save(os.path.join(save_path, "model.keras"))
        
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
        
        self.model = keras.models.load_model(os.path.join(path, "model.keras"))